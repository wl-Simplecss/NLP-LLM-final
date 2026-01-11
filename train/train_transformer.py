"""
Transformer NMT训练脚本 - 改进版
添加BLEU评估、日志输出、测试集评估
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_jsonl, TranslationDataset, collate_fn, Vocab, tokenize_en
from models.models_transformer import TransformerNMT
from run_mt_pipeline import bleu_compute_corpus_bleu


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
    max_len: int,
    use_small: bool,
    use_spm: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, object, object, list, list]:
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
    
    return train_loader, valid_loader, test_loader, train_ds.src_vocab, train_ds.tgt_vocab, valid_data, test_data


def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    for src, tgt, src_lengths, tgt_lengths, src_mask, tgt_mask in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_mask = src_mask.to(device)
        
        optimizer.zero_grad()
        tgt_inp = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        logits = model(src, src_mask, tgt_inp)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    
    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
    return total_loss / len(loader), current_lr


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for src, tgt, src_lengths, tgt_lengths, src_mask, tgt_mask in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_mask = src_mask.to(device)
        
        tgt_inp = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        logits = model(src, src_mask, tgt_inp)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def compute_bleu(model, loader, src_vocab, tgt_vocab, device, max_samples=500, max_len=128):
    """计算BLEU分数"""
    model.eval()
    candidates = []
    references = []
    count = 0
    
    for src, tgt, src_lengths, tgt_lengths, src_mask, tgt_mask in loader:
        src = src.to(device)
        src_mask = src_mask.to(device)
        
        for i in range(src.size(0)):
            if count >= max_samples:
                break
            
            single_src = src[i:i+1]
            single_mask = src_mask[i:i+1]
            
            # greedy decode
            pred_ids = model.greedy_decode(single_src, single_mask, max_len=max_len)
            pred_ids = pred_ids[0].cpu().tolist()
            ref_ids = tgt[i].tolist()
            
            # 解码
            if hasattr(tgt_vocab, 'decode_to_sentence'):
                pred_str = tgt_vocab.decode_to_sentence(pred_ids)
                ref_str = tgt_vocab.decode_to_sentence(ref_ids)
            else:
                pred_tokens = tgt_vocab.decode(pred_ids, remove_special=True)
                ref_tokens_raw = tgt_vocab.decode(ref_ids, remove_special=True)
                pred_str = ' '.join(pred_tokens)
                ref_str = ' '.join(ref_tokens_raw)
            
            # 使用tokenize_en标准化
            pred_tokens = tokenize_en(pred_str)
            ref_tokens = tokenize_en(ref_str)
            
            candidates.append(pred_tokens)
            references.append([ref_tokens])
            count += 1
        
        if count >= max_samples:
            break
    
    if not candidates:
        return 0.0
    
    bleu_score = bleu_compute_corpus_bleu(candidates, references, max_n=4)
    return bleu_score * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_raw/AP0004_Midterm&Final_translation_dataset_zh_en")
    parser.add_argument("--use_small", action="store_true", help="use 10k training set")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=128)
    # 模型参数 - 最优配置
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--dim_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)  # 增加dropout减少过拟合
    parser.add_argument("--use_spm", action="store_true", default=False)  # 默认使用Jieba分词
    parser.add_argument("--pos_encoding", type=str, default="sinusoidal", choices=["sinusoidal", "learned", "relative"])
    parser.add_argument("--norm_type", type=str, default="rmsnorm", choices=["layernorm", "rmsnorm"])  # RMSNorm效果更好
    # 训练参数 - 充分训练
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--lr_scale", type=float, default=1.0)  # Noam学习率缩放系数
    parser.add_argument("--save_dir", type=str, default="checkpoints/checkpoints_transformer")
    parser.add_argument("--log_dir", type=str, default="outputs")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--eval_bleu_every", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"transformer_training_{timestamp}.log")
    logger = Logger(log_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印配置
    logger.log("=" * 60)
    logger.log("Transformer Training Configuration")
    logger.log("=" * 60)
    for k, v in vars(args).items():
        logger.log(f"  {k}: {v}")
    logger.log(f"  device: {device}")
    logger.log("=" * 60)

    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab, valid_data, test_data = build_dataloaders(
        args.data_dir, args.batch_size, args.max_len, args.use_small, use_spm=args.use_spm
    )

    logger.log(f"Train samples: {len(train_loader.dataset)}")
    logger.log(f"Valid samples: {len(valid_loader.dataset)}")
    logger.log(f"Test samples: {len(test_loader.dataset)}")
    logger.log(f"Source vocab size: {len(src_vocab.itos)}")
    logger.log(f"Target vocab size: {len(tgt_vocab.itos)}")
    logger.log("=" * 60)

    model = TransformerNMT(
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pos_encoding=args.pos_encoding,
        norm_type=args.norm_type,
    ).to(device)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Total parameters: {total_params:,}")
    logger.log(f"Trainable parameters: {trainable_params:,}")
    logger.log("=" * 60)

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    # Noam学习率调度
    def _noam_lambda(step: int):
        step = max(step, 1)
        return args.lr_scale * (args.d_model ** -0.5) * min(step ** -0.5, step * args.warmup_steps ** -1.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_noam_lambda)

    best_valid_loss = float('inf')
    best_bleu = 0.0
    best_epoch = 0
    no_improve_count = 0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train_loss, current_lr = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        valid_loss = eval_epoch(model, valid_loader, criterion, device)
        
        elapsed = timedelta(seconds=int(time.time() - start_time))
        
        # 计算BLEU
        bleu_str = ""
        if epoch % args.eval_bleu_every == 0 or epoch == 1:
            bleu_score = compute_bleu(model, valid_loader, src_vocab, tgt_vocab, device, 
                                       max_samples=min(300, len(valid_loader.dataset)), max_len=args.max_len)
            bleu_str = f" | BLEU {bleu_score:.2f}"
            if bleu_score > best_bleu:
                best_bleu = bleu_score
        
        # 检查最佳模型
        is_best = valid_loss < best_valid_loss
        best_mark = ""
        if is_best:
            best_valid_loss = valid_loss
            best_epoch = epoch
            no_improve_count = 0
            best_mark = " [BEST]"
            
            # 保存最佳模型
            best_path = os.path.join(args.save_dir, "best_transformer_model.pt")
            torch.save({
                "model_state": model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "args": vars(args),
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }, best_path)
        else:
            no_improve_count += 1
        
        log_msg = f"Epoch {epoch:>3}/{args.epochs} | Train {train_loss:.4f} | Valid {valid_loss:.4f} | LR {current_lr:.6f}{bleu_str} | Elapsed {elapsed}{best_mark}"
        logger.log(log_msg)
        
        # 每20轮保存checkpoint
        if epoch % 20 == 0:
            ckpt_path = os.path.join(args.save_dir, f"transformer_epoch{epoch}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "args": vars(args),
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }, ckpt_path)
        
        # Early stopping
        if no_improve_count >= args.patience:
            logger.log(f"Early stop at epoch {epoch}. Best epoch={best_epoch}, Best valid loss={best_valid_loss:.4f}")
            break
    
    # 测试集评估
    logger.log("=" * 60)
    logger.log("Training completed. Evaluating on test set...")
    
    # 加载最佳模型
    best_ckpt = torch.load(os.path.join(args.save_dir, "best_transformer_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    
    test_loss = eval_epoch(model, test_loader, criterion, device)
    test_bleu = compute_bleu(model, test_loader, src_vocab, tgt_vocab, device, 
                              max_samples=len(test_loader.dataset), max_len=args.max_len)
    
    logger.log(f"Test Loss: {test_loss:.4f}")
    logger.log(f"Test BLEU: {test_bleu:.2f}")
    logger.log("=" * 60)
    logger.log(f"Best epoch: {best_epoch}")
    logger.log(f"Best valid loss: {best_valid_loss:.4f}")
    logger.log(f"Best valid BLEU: {best_bleu:.2f}")
    logger.log("=" * 60)
    
    logger.close()
    print(f"\nTraining log saved to: {log_path}")


if __name__ == "__main__":
    main()
