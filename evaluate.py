"""
BLEU评估脚本 - 修正版 (Corpus-level BLEU)
"""
import argparse
import math
import os
import collections
from typing import List

import torch
from tqdm import tqdm

# 引入项目依赖
from data_utils import load_jsonl, Vocab, SPMVocabWrapper, tokenize_zh, tokenize_en
from models.models_rnn import EncoderRNN, DecoderRNN, Seq2Seq
from models.models_transformer import TransformerNMT


def compute_corpus_bleu(candidates: List[List[str]], references: List[List[List[str]]], max_n: int = 4) -> float:
    """
    计算 Corpus-level BLEU 分数
    
    :param candidates: 预测结果列表，每个元素是一个token列表 [ ['hello', 'world'], ... ]
    :param references: 参考答案列表，每个元素是参考答案列表的列表 [ [['hello', 'world']], ... ]
    :param max_n: 最大 N-gram
    :return: BLEU score
    """
    p_numerators = collections.defaultdict(int)  # 匹配的 n-gram 数量（clipped）
    p_denominators = collections.defaultdict(int)  # 预测的 n-gram 总数
    hyp_lengths = 0
    ref_lengths = 0
    
    for cand, refs in zip(candidates, references):
        # 1. 长度惩罚统计
        hyp_lengths += len(cand)
        # 找到长度最接近 prediction 的 reference
        best_ref_len = min((len(ref) for ref in refs), key=lambda x: abs(x - len(cand)))
        ref_lengths += best_ref_len
        
        # 2. 对每个 n-gram 级别进行统计
        for n in range(1, max_n + 1):
            # 生成 candidate n-grams
            cand_ngrams = [tuple(cand[i:i+n]) for i in range(len(cand) - n + 1)]
            cand_cnt = collections.Counter(cand_ngrams)
            
            # 累加分母（预测的 n-gram 总数）
            p_denominators[n] += len(cand_ngrams)
            
            # 生成 references n-grams 并找最大重叠
            # 对于每个 n-gram，它在 candidate 中的计数不能超过它在任意单一 reference 中的最大计数
            max_ref_cnt = collections.defaultdict(int)
            for ref in refs:
                ref_ngrams = [tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)]
                ref_cnt = collections.Counter(ref_ngrams)
                for gram, count in ref_cnt.items():
                    max_ref_cnt[gram] = max(max_ref_cnt[gram], count)
            
            # 累加分子 (Clipped Count)
            for gram, count in cand_cnt.items():
                p_numerators[n] += min(count, max_ref_cnt.get(gram, 0))
    
    # 计算 BLEU
    # 1. Brevity Penalty (BP)
    if hyp_lengths > ref_lengths:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_lengths / hyp_lengths) if hyp_lengths > 0 else 0.0
    
    # 2. Geometric Mean of Precisions (带平滑处理)
    weights = [0.25] * 4  # BLEU-4 的权重
    log_precisions = []
    
    for n in range(1, max_n + 1):
        if p_denominators[n] == 0:
            # 如果没有预测任何 n-gram，precision 为 0
            log_precisions.append(float('-inf'))
        else:
            if p_numerators[n] == 0:
                # 平滑处理：如果匹配数为 0，使用平滑值避免 log(0)
                # 使用简单的平滑策略：1 / (2 * denominator)
                p_n = 1.0 / (2.0 * p_denominators[n])
            else:
                p_n = p_numerators[n] / p_denominators[n]
            
            log_precisions.append(math.log(p_n))
    
    # 如果所有 precision 都是 -inf，返回 0
    if all(p == float('-inf') for p in log_precisions):
        return 0.0
    
    # 计算加权几何平均
    s = sum(w * lp for w, lp in zip(weights, log_precisions))
    bleu = bp * math.exp(s)
    
    return bleu


def load_model(checkpoint_path: str, model_type: str):
    """加载模型"""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]
    args = ckpt.get("args", {})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "rnn":
        embed_dim = args.get("embed_dim", 256)
        hidden_size = args.get("hidden_size", 512)
        num_layers = args.get("num_layers", 2)
        dropout = args.get("dropout", 0.1)
        attn_type = args.get("attn_type", "dot")
        
        encoder = EncoderRNN(len(src_vocab.itos), embed_dim, hidden_size, num_layers, dropout)
        decoder = DecoderRNN(len(tgt_vocab.itos), embed_dim, hidden_size, num_layers, dropout, attn_type)
        model = Seq2Seq(encoder, decoder)
        model.load_state_dict(ckpt["model_state"])
    else:
        model = TransformerNMT(
            src_vocab_size=len(src_vocab.itos),
            tgt_vocab_size=len(tgt_vocab.itos),
            d_model=args.get("d_model", 512),
            num_heads=args.get("num_heads", 8),
            num_encoder_layers=args.get("num_encoder_layers", 3),
            num_decoder_layers=args.get("num_decoder_layers", 3),
            dim_ff=args.get("dim_ff", 2048),
            dropout=args.get("dropout", 0.3),
            max_len=args.get("max_len", 128),
            pos_encoding=args.get("pos_encoding", "sinusoidal"),
            norm_type=args.get("norm_type", "layernorm"),
        )
        model.load_state_dict(ckpt["model_state"])
    
    model.to(device)
    model.eval()
    return model, src_vocab, tgt_vocab, device


def evaluate_corpus(
    model,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    test_data: List[dict],
    device: torch.device,
    beam_size: int = 1,
    max_len: int = 64,
):
    """
    收集所有预测和参考，计算 Corpus BLEU
    """
    candidates = []
    references = []
    
    print(f"Starting evaluation on {len(test_data)} sentences...")
    
    for item in tqdm(test_data, desc="Evaluating", disable=True):
        src_text = item["zh"]
        tgt_text = item["en"]
        
        # 1. 预处理输入
        if isinstance(src_vocab, SPMVocabWrapper):
            src_ids = src_vocab.encode(src_text, add_sos_eos=True)
        else:
            src_tokens = tokenize_zh(src_text)
            src_ids = src_vocab.encode(src_tokens, add_sos_eos=True)
            
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = src_tensor.ne(0)
        
        # 2. 模型生成
        with torch.no_grad():
            if beam_size > 1:
                pred_ids = model.beam_search(src_tensor, src_mask, beam_size=beam_size, max_len=max_len)
            else:
                pred_ids = model.greedy_decode(src_tensor, src_mask, max_len=max_len)
        
        # 3. 解码预测结果 (Token级别)
        if isinstance(tgt_vocab, SPMVocabWrapper):
            # SPM: ID -> String -> Token List (tokenize_en)
            pred_str = tgt_vocab.decode_to_sentence(pred_ids[0].tolist())
            pred_tokens = tokenize_en(pred_str)
        else:
            # Vocab: ID -> Token List
            pred_tokens = tgt_vocab.decode(pred_ids[0].tolist(), remove_special=True)
        
        # 4. 处理参考答案 (Token级别)
        # 注意：References 应该是一个 list of lists (支持多参考)，这里我们只有一个参考
        ref_tokens = tokenize_en(tgt_text)
        
        candidates.append(pred_tokens)
        references.append([ref_tokens])  # 包裹一层，格式为 [[ref1_tokens], [ref2_tokens], ...]
    
    # 计算 Corpus BLEU
    bleu_score = compute_corpus_bleu(candidates, references)
    return bleu_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="transformer", choices=["rnn", "transformer"])
    parser.add_argument("--test_file", type=str, default="data_raw/AP0004_Midterm&Final_translation_dataset_zh_en/test.jsonl")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=80)
    args = parser.parse_args()
    
    if not os.path.exists(args.test_file):
        print(f"Test file not found: {args.test_file}")
        return
    
    test_data = load_jsonl(args.test_file)
    model, src_vocab, tgt_vocab, device = load_model(args.checkpoint, args.model_type)
    
    bleu = evaluate_corpus(model, src_vocab, tgt_vocab, test_data, device, args.beam_size, args.max_len)
    
    print(f"\n{'='*40}")
    print(f"Evaluation Result")
    print(f"Model: {args.checkpoint}")
    print(f"BLEU-4 Score: {bleu:.4f} ({bleu*100:.2f}%)")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
