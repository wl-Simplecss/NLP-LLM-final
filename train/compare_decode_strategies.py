"""
对比解码策略: Greedy Search vs Beam Search
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data_utils import load_jsonl, Vocab, SPMVocabWrapper
from models.models_rnn import EncoderRNN, DecoderRNN, Seq2Seq
from models.models_transformer import TransformerNMT
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 为了兼容性，创建别名
SentencePieceVocab = SPMVocabWrapper


def load_model(checkpoint_path, model_type, device):
    """加载模型"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {})
    
    # 尝试从checkpoint加载词表，如果不存在则报错
    if "src_vocab" not in ckpt or "tgt_vocab" not in ckpt:
        raise KeyError(
            f"Checkpoint中缺少词表信息。请确保checkpoint包含'src_vocab'和'tgt_vocab'。\n"
            f"这可能是因为使用了旧版本的训练脚本。请使用最新版本重新训练模型。"
        )
    
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]
    
    # 加载模型
    if model_type == "rnn":
        embed_dim = args.get("embed_dim", 256)
        hidden_size = args.get("hidden_size", 512)
        num_layers = args.get("num_layers", 2)
        dropout = args.get("dropout", 0.3)
        attn_type = args.get("attn_type", "additive")
        
        encoder = EncoderRNN(len(src_vocab.itos), embed_dim, hidden_size, num_layers, dropout)
        decoder = DecoderRNN(len(tgt_vocab.itos), embed_dim, hidden_size, num_layers, dropout, attn_type)
        model = Seq2Seq(encoder, decoder)
    else:  # transformer
        d_model = args.get("d_model", 512)
        num_heads = args.get("num_heads", 8)
        num_encoder_layers = args.get("num_encoder_layers", 3)
        num_decoder_layers = args.get("num_decoder_layers", 3)
        dim_ff = args.get("dim_ff", 2048)
        dropout = args.get("dropout", 0.3)
        pos_encoding = args.get("pos_encoding", "sinusoidal")
        norm_type = args.get("norm_type", "layernorm")
        
        model = TransformerNMT(
            len(src_vocab.itos), len(tgt_vocab.itos),
            d_model=d_model, num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_ff=dim_ff, dropout=dropout,
            pos_encoding=pos_encoding, norm_type=norm_type
        )
    
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    return model, src_vocab, tgt_vocab


def evaluate_decode_strategy(model, test_data, src_vocab, tgt_vocab, device, 
                            strategy="greedy", beam_size=5, max_len=80, model_type="rnn"):
    """评估解码策略"""
    model.eval()
    smoothing = SmoothingFunction().method1
    
    total_bleu = 0.0
    results = []
    
    with torch.no_grad():
        for item in tqdm(test_data, desc=f"Evaluating {strategy}"):
            src_text = item["zh"]
            tgt_text = item["en"]
            
            # 编码源句子
            if isinstance(src_vocab, SPMVocabWrapper):
                # SentencePiece 直接处理字符串
                src_ids = src_vocab.encode(src_text)
            else:
                # 传统 Vocab 需要先分词
                src_ids = [src_vocab.stoi.get(tok, src_vocab.stoi["<unk>"]) for tok in src_text.split()]
            
            src_tensor = torch.tensor([src_ids], device=device)
            # 获取 pad token id（兼容 SPMVocabWrapper 和 Vocab）
            if isinstance(src_vocab, SPMVocabWrapper):
                # SentencePiece 使用 0 作为 pad_id
                pad_id = 0
            else:
                # 传统 Vocab 从 stoi 获取
                pad_id = src_vocab.stoi.get("<pad>", 0)
            # RNN模型的mask应该是 (B, S) 形状，注意力机制内部会处理维度
            src_mask = (src_tensor != pad_id)  # (B, S)
            
            # 解码
            if strategy == "greedy":
                if model_type == "rnn":
                    pred_ids = model.greedy_decode(src_tensor, src_mask, max_len=max_len)
                else:
                    pred_ids = model.greedy_decode(src_tensor, src_mask, max_len=max_len)
            else:  # beam search
                if model_type == "rnn":
                    pred_ids = model.beam_search(src_tensor, src_mask, beam_size=beam_size, max_len=max_len)
                else:
                    pred_ids = model.beam_search(src_tensor, src_mask, beam_size=beam_size, max_len=max_len)
            
            # 解码预测结果
            if isinstance(tgt_vocab, SPMVocabWrapper):
                # SentencePiece 解码：使用 decode_to_sentence 直接获取文本
                pred_text = tgt_vocab.decode_to_sentence(pred_ids[0].tolist())
                # 转换为token列表用于BLEU计算
                pred_tokens = pred_text.split()
            else:
                # 传统 Vocab 解码
                pad_id = tgt_vocab.stoi.get("<pad>", 0)
                sos_id = tgt_vocab.stoi.get("<sos>", 1)
                eos_id = tgt_vocab.stoi.get("<eos>", 2)
                pred_tokens = [tgt_vocab.itos[idx] for idx in pred_ids[0].tolist() 
                             if idx not in [pad_id, sos_id, eos_id] and 0 <= idx < len(tgt_vocab.itos)]
                pred_text = " ".join(pred_tokens)
            
            # 计算BLEU
            reference = tgt_text.split()
            bleu = sentence_bleu([reference], pred_tokens, smoothing_function=smoothing)
            total_bleu += bleu
            
            results.append({
                "src": src_text,
                "tgt": tgt_text,
                "pred": pred_text,
                "bleu": bleu
            })
    
    avg_bleu = total_bleu / len(test_data)
    return avg_bleu, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--model_type", type=str, choices=["rnn", "transformer"], required=True)
    parser.add_argument("--test_file", type=str, required=True, help="测试文件路径")
    parser.add_argument("--beam_size", type=int, default=5, help="束搜索大小")
    parser.add_argument("--max_len", type=int, default=80, help="最大生成长度")
    parser.add_argument("--output_file", type=str, default="decode_comparison.txt", help="输出文件")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}...")
    model, src_vocab, tgt_vocab = load_model(args.checkpoint, args.model_type, device)
    
    # 加载测试数据
    test_data = load_jsonl(args.test_file)
    print(f"Loaded {len(test_data)} test examples")
    
    # 评估贪婪搜索
    print("\n" + "="*80)
    print("评估 Greedy Search")
    print("="*80)
    greedy_bleu, greedy_results = evaluate_decode_strategy(
        model, test_data, src_vocab, tgt_vocab, device,
        strategy="greedy", max_len=args.max_len, model_type=args.model_type
    )
    
    # 评估束搜索
    print("\n" + "="*80)
    print(f"评估 Beam Search (beam_size={args.beam_size})")
    print("="*80)
    beam_bleu, beam_results = evaluate_decode_strategy(
        model, test_data, src_vocab, tgt_vocab, device,
        strategy="beam", beam_size=args.beam_size, max_len=args.max_len, model_type=args.model_type
    )
    
    # 输出结果
    print("\n" + "="*80)
    print("解码策略对比结果")
    print("="*80)
    print(f"Greedy Search BLEU-4: {greedy_bleu:.4f}")
    print(f"Beam Search (beam_size={args.beam_size}) BLEU-4: {beam_bleu:.4f}")
    print(f"性能提升: {(beam_bleu - greedy_bleu):.4f} ({(beam_bleu - greedy_bleu) / greedy_bleu * 100:.2f}%)")
    
    # 保存详细结果
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("解码策略对比结果\n")
        f.write("="*80 + "\n\n")
        f.write(f"Greedy Search BLEU-4: {greedy_bleu:.4f}\n")
        f.write(f"Beam Search (beam_size={args.beam_size}) BLEU-4: {beam_bleu:.4f}\n")
        f.write(f"性能提升: {(beam_bleu - greedy_bleu):.4f} ({(beam_bleu - greedy_bleu) / greedy_bleu * 100:.2f}%)\n\n")
        
        f.write("="*80 + "\n")
        f.write("详细对比 (前10个例子)\n")
        f.write("="*80 + "\n\n")
        
        for i in range(min(10, len(test_data))):
            f.write(f"\n例子 {i+1}:\n")
            f.write(f"源文本: {greedy_results[i]['src']}\n")
            f.write(f"参考翻译: {greedy_results[i]['tgt']}\n")
            f.write(f"Greedy预测: {greedy_results[i]['pred']} (BLEU: {greedy_results[i]['bleu']:.4f})\n")
            f.write(f"Beam预测: {beam_results[i]['pred']} (BLEU: {beam_results[i]['bleu']:.4f})\n")
    
    print(f"\n详细结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()

