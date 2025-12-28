import argparse
import os
import time
from datetime import timedelta
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_jsonl, TranslationDataset, collate_fn, Vocab
from models.models_rnn import EncoderRNN, DecoderRNN, Seq2Seq


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    max_len: int = 128,
    use_small: bool = True,
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


def _scheduled_sampling_forward(
    model: Seq2Seq,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    tgt: torch.Tensor,
    tf_ratio: float,
) -> torch.Tensor:
    """
    按时间步的 scheduled sampling：
    - 每步以 tf_ratio 概率用 gold token，否则用模型预测 token 作为下一步输入
    返回 logits: (B, T-1, V) 对齐 tgt[:, 1:]
    """
    enc_out, hidden = model.encoder(src, src_mask)
    B, T = tgt.shape
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
    for src, tgt, src_mask, tgt_mask in loader:
        src, tgt, src_mask = src.to(device), tgt.to(device), src_mask.to(device)
        optimizer.zero_grad()
        logits = _scheduled_sampling_forward(model, src, src_mask, tgt, tf_ratio=tf_ratio)
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
    for src, tgt, src_mask, tgt_mask in loader:
        src, tgt, src_mask = src.to(device), tgt.to(device), src_mask.to(device)
        # 验证时固定 teacher forcing = 1.0，保证loss稳定可比
        logits = _scheduled_sampling_forward(model, src, src_mask, tgt, tf_ratio=1.0)
        tgt_y = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_raw/AP0004_Midterm&Final_translation_dataset_zh_en")
    parser.add_argument("--use_small", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=100)
    # 更稳的默认值：RNN不要太大，否则更容易不稳定/重复
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attn_type", type=str, default="general", choices=["general", "additive", "dot", "multiplicative"])
    parser.add_argument("--use_spm", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints/checkpoints_rnn")
    # scheduled sampling
    parser.add_argument("--tf_start", type=float, default=1.0)
    parser.add_argument("--tf_end", type=float, default=0.6)
    parser.add_argument("--tf_decay_epochs", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, src_vocab, tgt_vocab = build_dataloaders(
        args.data_dir, args.batch_size, args.max_len, args.use_small, use_spm=args.use_spm
    )

    encoder = EncoderRNN(len(src_vocab.itos), args.embed_dim, args.hidden_size, args.num_layers, args.dropout)
    decoder = DecoderRNN(len(tgt_vocab.itos), args.embed_dim, args.hidden_size, args.num_layers, args.dropout, args.attn_type)
    model = Seq2Seq(encoder, decoder).to(device)

    # label_smoothing 对 NMT 有用，但RNN太弱时 smoothing 太大可能"说话更飘"
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_valid = float("inf")
    best_epoch = 0
    no_improve = 0
    patience = 10
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
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

        print(f"Epoch {epoch:>3}/{args.epochs} | Train {train_loss:.4f} | Valid {valid_loss:.4f} | LR {lr:.6f} | TF {tf_ratio:.2f} | Elapsed {elapsed}")

        is_best = valid_loss < best_valid
        if is_best:
            best_valid = valid_loss
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        ckpt_path = os.path.join(args.save_dir, f"rnn_epoch{epoch}.pt")
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

        if no_improve >= patience:
            print(f"Early stop. best_epoch={best_epoch}, best_valid={best_valid:.4f}")
            break


if __name__ == "__main__":
    main()
