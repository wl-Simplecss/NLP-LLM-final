import argparse
import os
import time
import sys
from datetime import datetime, timedelta
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_jsonl, TranslationDataset, collate_fn, Vocab
from models.models_rnn import EncoderRNN, DecoderRNN, Seq2Seq

# 导入本地BLEU计算函数
from run_mt_pipeline import bleu_compute_corpus_bleu
from data_utils import tokenize_en


class Logger:
    """同时输出到控制台和日志文件"""
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.file = open(log_path, 'w', encoding='utf-8')
    
    def log(self, msg):
        print(msg)
        self.file.write(msg + '\n')
        self.file.flush()
    
    def close(self):
        self.file.close()


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    max_len: int = 128,
    use_small: bool = True,
    use_spm: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, object, object]:
    train_file = "train_10k.jsonl" if use_small else "train_100k.jsonl"
    train_path = os.path.join(data_dir, train_file)
    valid_path = os.path.join(data_dir, "valid.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")

    train_data = load_jsonl(train_path)
    valid_data = load_jsonl(valid_path)
    test_data = load_jsonl(test_path)

    train_ds = TranslationDataset(train_data, src_lang="zh", tgt_lang="en", max_len=max_len, use_spm=use_spm)
    valid_ds = TranslationDataset(
        valid_data, src_lang="zh", tgt_lang="en", max_len=max_len,
        src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab, use_spm=use_spm,
    )
    test_ds = TranslationDataset(
        test_data, src_lang="zh", tgt_lang="en", max_len=max_len,
        src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab, use_spm=use_spm,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, valid_loader, test_loader, train_ds.src_vocab, train_ds.tgt_vocab


def _scheduled_sampling_forward(
    model: Seq2Seq,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt: torch.Tensor,
    tf_ratio: float,
) -> torch.Tensor:
    """
    Scheduled sampling: 每步以 tf_ratio 概率用 gold token
    """
    enc_out, hidden = model.encoder(src, src_lengths)
    B, T = tgt.shape
    # 根据encoder outputs的实际长度生成mask
    enc_seq_len = enc_out.size(1)
    src_mask = torch.arange(enc_seq_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)
    
    input_tok = tgt[:, 0]  # <sos>
    logits_list = []
    for t in range(1, T):
        step_logits, hidden, _ = model.decoder.forward_step(input_tok, hidden, enc_out, src_mask)
        logits_list.append(step_logits.unsqueeze(1))
        pred_tok = torch.argmax(step_logits, dim=-1)
        gold_tok = tgt[:, t]
        use_teacher = (torch.rand(B, device=tgt.device) < tf_ratio)
        input_tok = torch.where(use_teacher, gold_tok, pred_tok)
    return torch.cat(logits_list, dim=1)


def train_epoch(model: Seq2Seq, loader: DataLoader, criterion, optimizer, device, tf_ratio: float):
    model.train()
    total_loss = 0.0
    for src, tgt, src_lengths, tgt_lengths, src_mask, tgt_mask in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_lengths = src_lengths.to(device)
        
        optimizer.zero_grad()
        logits = _scheduled_sampling_forward(model, src, src_lengths, tgt, tf_ratio=tf_ratio)
        tgt_y = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model: Seq2Seq, loader: DataLoader, criterion, device):
    model.eval()
    total_loss = 0.0
    for src, tgt, src_lengths, tgt_lengths, src_mask, tgt_mask in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_lengths = src_lengths.to(device)
        
        logits = _scheduled_sampling_forward(model, src, src_lengths, tgt, tf_ratio=1.0)
        tgt_y = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def compute_bleu(model: Seq2Seq, loader: DataLoader, tgt_vocab, device, max_samples=500):
    """
    使用本地bleu_compute_corpus_bleu计算BLEU-4分数
    返回值范围: 0.0 ~ 1.0 (乘100后为百分比)
    """
    model.eval()
    candidates = []  # List[List[str]] - 预测的token列表
    references = []  # List[List[List[str]]] - 参考的token列表（支持多参考）
    count = 0
    
    for src, tgt, src_lengths, tgt_lengths, src_mask, tgt_mask in loader:
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        
        for i in range(src.size(0)):
            if count >= max_samples:
                break
            
            # 单个样本推理
            single_src = src[i:i+1]
            single_len = src_lengths[i:i+1]
            
            # greedy decode
            pred_ids = model.greedy_decode(single_src, single_len, max_len=64)
            pred_ids = pred_ids[0].cpu().tolist()
            
            # 参考译文
            ref_ids = tgt[i].tolist()
            
            # 解码为字符串，然后用tokenize_en分词
            if hasattr(tgt_vocab, 'decode_to_sentence'):
                pred_str = tgt_vocab.decode_to_sentence(pred_ids)
                ref_str = tgt_vocab.decode_to_sentence(ref_ids)
            else:
                pred_tokens = tgt_vocab.decode(pred_ids, remove_special=True)
                ref_tokens_raw = tgt_vocab.decode(ref_ids, remove_special=True)
                pred_str = ' '.join(pred_tokens)
                ref_str = ' '.join(ref_tokens_raw)
            
            # 使用tokenize_en进行标准化分词（与run_mt_pipeline一致）
            pred_tokens = tokenize_en(pred_str)
            ref_tokens = tokenize_en(ref_str)
            
            candidates.append(pred_tokens)
            references.append([ref_tokens])  # 每个参考是一个列表
            count += 1
        
        if count >= max_samples:
            break
    
    if not candidates:
        return 0.0
    
    # 使用本地BLEU计算函数（返回0~1的值）
    bleu_score = bleu_compute_corpus_bleu(candidates, references, max_n=4)
    # 转换为百分比形式（0~100）
    return bleu_score * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_raw/AP0004_Midterm&Final_translation_dataset_zh_en")
    parser.add_argument("--use_small", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)  # 与报告一致，减少过拟合
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attn_type", type=str, default="general", choices=["general", "additive", "dot", "multiplicative"])
    parser.add_argument("--use_spm", action="store_true", default=False)  # 默认使用Jieba+空格分词，与报告一致
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="checkpoints/checkpoints_rnn")
    parser.add_argument("--log_dir", type=str, default="outputs")
    # scheduled sampling (报告中使用纯Teacher Forcing)
    parser.add_argument("--tf_start", type=float, default=1.0)
    parser.add_argument("--tf_end", type=float, default=1.0)  # 纯TF，不衰减
    parser.add_argument("--tf_decay_epochs", type=int, default=100)
    # 早停
    parser.add_argument("--patience", type=int, default=30)
    # BLEU评估间隔
    parser.add_argument("--eval_bleu_every", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"rnn_training_{timestamp}.log")
    logger = Logger(log_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印配置
    logger.log("=" * 60)
    logger.log("RNN Training Configuration")
    logger.log("=" * 60)
    for k, v in vars(args).items():
        logger.log(f"  {k}: {v}")
    logger.log(f"  device: {device}")
    logger.log("=" * 60)

    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        args.data_dir, args.batch_size, args.max_len, args.use_small, use_spm=args.use_spm
    )
    
    logger.log(f"Train samples: {len(train_loader.dataset)}")
    logger.log(f"Valid samples: {len(valid_loader.dataset)}")
    logger.log(f"Test samples: {len(test_loader.dataset)}")
    logger.log(f"Source vocab size: {len(src_vocab.itos)}")
    logger.log(f"Target vocab size: {len(tgt_vocab.itos)}")
    logger.log("=" * 60)

    encoder = EncoderRNN(len(src_vocab.itos), args.embed_dim, args.hidden_size, args.num_layers, args.dropout)
    decoder = DecoderRNN(len(tgt_vocab.itos), args.embed_dim, args.hidden_size, args.num_layers, args.dropout, args.attn_type)
    model = Seq2Seq(encoder, decoder).to(device)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Total parameters: {total_params:,}")
    logger.log(f"Trainable parameters: {trainable_params:,}")
    logger.log("=" * 60)

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    # 使用更温和的学习率衰减策略
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=15, min_lr=1e-5)

    best_valid = float("inf")
    best_bleu = 0.0
    best_epoch = 0
    no_improve = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Teacher Forcing ratio衰减
        if epoch <= args.tf_decay_epochs:
            frac = (epoch - 1) / max(args.tf_decay_epochs - 1, 1)
            tf_ratio = args.tf_start + frac * (args.tf_end - args.tf_start)
        else:
            tf_ratio = args.tf_end

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, tf_ratio=tf_ratio)
        valid_loss = eval_epoch(model, valid_loader, criterion, device)
        scheduler.step(valid_loss)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = timedelta(seconds=int(time.time() - start_time))

        # 定期计算BLEU
        bleu_str = ""
        if epoch % args.eval_bleu_every == 0 or epoch == 1:
            bleu_score = compute_bleu(model, valid_loader, tgt_vocab, device, max_samples=500)
            bleu_str = f" | BLEU {bleu_score:.2f}"
            if bleu_score > best_bleu:
                best_bleu = bleu_score

        log_msg = f"Epoch {epoch:>3}/{args.epochs} | Train {train_loss:.4f} | Valid {valid_loss:.4f} | LR {lr:.6f} | TF {tf_ratio:.2f}{bleu_str} | Elapsed {elapsed}"
        logger.log(log_msg)

        is_best = valid_loss < best_valid
        if is_best:
            best_valid = valid_loss
            best_epoch = epoch
            no_improve = 0
            
            # 保存最佳模型
            best_path = os.path.join(args.save_dir, "best_rnn_model.pt")
            torch.save({
                "model_state": model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "args": vars(args),
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }, best_path)
            logger.log(f"  -> New best model saved at epoch {epoch}")
        else:
            no_improve += 1

        # 每50轮保存checkpoint
        if epoch % 50 == 0:
        ckpt_path = os.path.join(args.save_dir, f"rnn_epoch{epoch}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "args": vars(args),
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }, ckpt_path)

        if no_improve >= args.patience:
            logger.log(f"Early stop at epoch {epoch}. Best epoch={best_epoch}, Best valid loss={best_valid:.4f}")
            break

    # 训练结束，在测试集上评估
    logger.log("=" * 60)
    logger.log("Training completed. Evaluating on test set...")
    
    # 加载最佳模型
    best_ckpt = torch.load(os.path.join(args.save_dir, "best_rnn_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    
    test_loss = eval_epoch(model, test_loader, criterion, device)
    test_bleu = compute_bleu(model, test_loader, tgt_vocab, device, max_samples=1000)
    
    logger.log(f"Test Loss: {test_loss:.4f}")
    logger.log(f"Test BLEU: {test_bleu:.2f}")
    logger.log("=" * 60)
    logger.log(f"Best epoch: {best_epoch}")
    logger.log(f"Best valid loss: {best_valid:.4f}")
    logger.log(f"Best valid BLEU: {best_bleu:.2f}")
    logger.log("=" * 60)
    
    logger.close()
    print(f"\nTraining log saved to: {log_path}")


if __name__ == "__main__":
    main()
