import argparse
import os
import time
from datetime import datetime, timedelta
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_jsonl, TranslationDataset, collate_fn, Vocab
from models.models_transformer import TransformerNMT


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    max_len: int,
    use_small: bool,
    use_spm: bool = True,
) -> Tuple[DataLoader, DataLoader, Vocab, Vocab]:
    train_file = "train_10k.jsonl" if use_small else "train_100k.jsonl"
    train_path = os.path.join(data_dir, train_file)
    valid_path = os.path.join(data_dir, "valid.jsonl")

    train_data = load_jsonl(train_path)
    valid_data = load_jsonl(valid_path)

    train_ds = TranslationDataset(train_data, src_lang="zh", tgt_lang="en", max_len=max_len, use_spm=use_spm)
    valid_ds = TranslationDataset(
        valid_data,
        src_lang="zh",
        tgt_lang="en",
        max_len=max_len,
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
        use_spm=use_spm,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader, train_ds.src_vocab, train_ds.tgt_vocab


def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    for src, tgt, src_mask, tgt_mask in loader:
        src, tgt, src_mask, tgt_mask = src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device)
        optimizer.zero_grad()

        # shift target: input is [sos, ..., last-1], predict [first, ..., eos]
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
    for src, tgt, src_mask, tgt_mask in loader:
        src, tgt, src_mask, tgt_mask = src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device)
        tgt_inp = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        logits = model(src, src_mask, tgt_inp)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_raw/AP0004_Midterm&Final_translation_dataset_zh_en")
    parser.add_argument("--use_small", action="store_true", help="use 10k training set instead of 100k")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)  # 增加模型容量
    parser.add_argument("--num_heads", type=int, default=8)  # 增加注意力头数
    parser.add_argument("--num_encoder_layers", type=int, default=3)  # 减少层数到3，适合100k数据量
    parser.add_argument("--num_decoder_layers", type=int, default=3)  # 减少层数到3，适合100k数据量
    parser.add_argument("--dim_ff", type=int, default=2048)  # 增加FFN维度（通常是d_model的4倍）
    parser.add_argument("--dropout", type=float, default=0.3)  # 提升dropout到0.3，防止过拟合
    parser.add_argument("--use_spm", action="store_true", default=True, help="使用SentencePiece分词")
    parser.add_argument("--pos_encoding", type=str, default="sinusoidal", choices=["sinusoidal", "learned", "relative"])
    parser.add_argument("--norm_type", type=str, default="layernorm", choices=["layernorm", "rmsnorm"])
    parser.add_argument("--epochs", type=int, default=60)  # 增加到60轮，确保模型充分收敛
    parser.add_argument("--warmup_steps", type=int, default=8000)  # 增加Warmup步数
    parser.add_argument("--lr", type=float, default=1.0, help="Noam lr scale factor (peak lr≈lr*d_model^-0.5*warmup^-0.5)")
    parser.add_argument("--save_dir", type=str, default="checkpoints/checkpoints_transformer")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, src_vocab, tgt_vocab = build_dataloaders(
        args.data_dir, args.batch_size, args.max_len, args.use_small, use_spm=args.use_spm
    )

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

    # 使用Label Smoothing减少过拟合，提升泛化能力
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    # 添加weight_decay进行L2正则化，使用更标准的Adam参数
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)

    def _noam_lambda(step: int):
        step = max(step, 1)
        return args.lr * (args.d_model ** -0.5) * min(step ** -0.5, step * args.warmup_steps ** -1.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_noam_lambda)

    # 训练历史记录
    train_losses = []
    valid_losses = []
    learning_rates = []
    best_valid_loss = float('inf')
    best_epoch = 0
    start_time = time.time()
    patience = 15  # Early stopping patience (增加patience到15)
    no_improve_count = 0
    
    print("\n" + "="*80)
    print("开始训练Transformer模型")
    print("="*80)
    print(f"模型配置:")
    print(f"  - 模型维度 (d_model): {args.d_model}")
    print(f"  - 注意力头数: {args.num_heads}")
    print(f"  - 编码器层数: {args.num_encoder_layers}")
    print(f"  - 解码器层数: {args.num_decoder_layers}")
    print(f"  - 前馈网络维度: {args.dim_ff}")
    print(f"  - 位置编码: {args.pos_encoding}")
    print(f"  - 归一化类型: {args.norm_type}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - 最大长度: {args.max_len}")
    print(f"  - 初始学习率: {args.lr}")
    print(f"  - 总训练轮数: {args.epochs}")
    print("="*80 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # 训练阶段（静默执行）
        train_loss, current_lr = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        train_losses.append(train_loss)
        
        # 验证阶段（静默执行）
        valid_loss = eval_epoch(model, valid_loader, criterion, device)
        valid_losses.append(valid_loss)
        
        # 更新学习率
        learning_rates.append(current_lr)
        
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start_time
        
        # 损失变化
        train_loss_change = ""
        valid_loss_change = ""
        if epoch > 1:
            train_diff = train_loss - train_losses[-2]
            valid_diff = valid_loss - valid_losses[-2]
            train_loss_change = f" ({train_diff:+.4f})" if abs(train_diff) > 0.0001 else ""
            valid_loss_change = f" ({valid_diff:+.4f})" if abs(valid_diff) > 0.0001 else ""
        
        # 检查是否是最佳模型
        is_best = valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_loss
            best_epoch = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # 精简输出：只显示关键结果
        best_mark = " [BEST]" if is_best else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f}{train_loss_change} | "
              f"Valid Loss: {valid_loss:.4f}{valid_loss_change} | LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s{best_mark}")
        
        # 保存checkpoint（静默）
        ckpt_path = os.path.join(args.save_dir, f"transformer_epoch{epoch}.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "args": vars(args),
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            },
            ckpt_path,
        )
        
        # 每10个epoch显示训练趋势
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"\n训练趋势 (最近10个epoch):")
            print(f"{'Epoch':<8} {'Train Loss':<12} {'Valid Loss':<12} {'LR':<12}")
            print("-" * 50)
            for i in range(max(0, epoch-10), epoch):
                print(f"{i+1:<8} {train_losses[i]:<12.4f} {valid_losses[i]:<12.4f} {learning_rates[i]:<12.6f}")
            print()
        
        # Early stopping: 如果验证损失连续patience个epoch没有改善，提前停止
        if no_improve_count >= patience:
            print(f"\nEarly Stopping触发! 验证损失连续{patience}个epoch没有改善")
            print(f"最佳模型: Epoch {best_epoch} (验证损失: {best_valid_loss:.4f})\n")
            break
    
    # 训练完成总结
    total_time = time.time() - start_time
    print(f"\n训练完成!")
    print(f"总训练时间: {timedelta(seconds=int(total_time))}")
    print(f"最佳模型: Epoch {best_epoch} (验证损失: {best_valid_loss:.4f})")
    print(f"最终训练损失: {train_losses[-1]:.4f} | 最终验证损失: {valid_losses[-1]:.4f}\n")


if __name__ == "__main__":
    main()


