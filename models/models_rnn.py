from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    """单向GRU编码器，使用packed sequence高效处理变长序列"""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 单向GRU
        self.rnn = nn.GRU(
            embed_dim, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, lengths):
        """
        Args:
            src: (B, S) 源序列
            lengths: (B,) 每个序列的实际长度
        Returns:
            outputs: (B, S, H) 编码器输出
            hidden: (num_layers, B, H) 最终隐藏状态
        """
        emb = self.dropout(self.embedding(src))  # (B, S, E)
        
        # 使用packed sequence高效处理变长序列
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        
        return outputs, hidden


class DotAttention(nn.Module):
    """点积注意力 - 适用于单向编码器"""
    def forward(self, query, keys, values, mask=None):
        # query: (B, 1, H)
        # keys: (B, S, H)
        # values: (B, S, H)
        d_k = keys.size(-1)
        scores = torch.bmm(query, keys.transpose(1, 2)) / (d_k ** 0.5)  # (B, 1, S)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, values)  # (B, 1, H)
        return ctx, attn


class GeneralAttention(nn.Module):
    """乘性注意力(General/Multiplicative)"""
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, query, keys, values, mask=None):
        # query: (B, 1, H)
        # keys: (B, S, H)
        d_k = keys.size(-1)
        q = self.Wa(query)  # (B, 1, H)
        
        scores = torch.bmm(q, keys.transpose(1, 2)) / (d_k ** 0.5)  # (B, 1, S)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, values)  # (B, 1, H)
        return ctx, attn


class AdditiveAttention(nn.Module):
    """加性注意力(Bahdanau)"""
    def __init__(self, hidden_size, attn_hidden_size=None):
        super().__init__()
        if attn_hidden_size is None:
            attn_hidden_size = hidden_size
            
        self.Wq = nn.Linear(hidden_size, attn_hidden_size, bias=False)
        self.Wk = nn.Linear(hidden_size, attn_hidden_size, bias=False)
        self.v = nn.Linear(attn_hidden_size, 1, bias=False)

    def forward(self, query, keys, values, mask=None):
        # query: (B, 1, H)
        # keys: (B, S, H)
        q = self.Wq(query)  # (B, 1, attn_H)
        k = self.Wk(keys)   # (B, S, attn_H)
        
        # 广播加法: (B, 1, attn_H) + (B, S, attn_H) -> (B, S, attn_H)
        scores = self.v(torch.tanh(q + k))  # (B, S, 1)
        scores = scores.transpose(1, 2)     # (B, 1, S)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        
        attn = F.softmax(scores, dim=-1)  # (B, 1, S)
        ctx = torch.bmm(attn, values)     # (B, 1, H)
        return ctx, attn


def build_attention(hidden_size: int, attn_type: str):
    attn_type = attn_type.lower()
    if attn_type == "dot":
        return DotAttention()
    if attn_type in ("general", "multiplicative"):
        return GeneralAttention(hidden_size)
    if attn_type in ("additive", "bahdanau"):
        return AdditiveAttention(hidden_size)
    raise ValueError(f"Unknown attention type: {attn_type}")


class DecoderRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        attn_type: str = "general",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 单向编码器，context维度 = hidden_size
        self.enc_output_size = hidden_size
        
        # GRU 输入: Embedding + Context
        self.rnn = nn.GRU(
            embed_dim + self.enc_output_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.attn = build_attention(hidden_size, attn_type)
        
        # 输出层: Hidden + Context -> Vocab
        self.fc_out = nn.Linear(hidden_size + self.enc_output_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, hidden, encoder_outputs, src_mask):
        """单步解码"""
        # input_tok: (B,)
        emb = self.dropout(self.embedding(input_tok).unsqueeze(1))  # (B, 1, E)
        
        # attention
        query = hidden[-1].unsqueeze(1)  # (B, 1, H) - 取最后一层的hidden
        ctx, attn = self.attn(query, encoder_outputs, encoder_outputs, src_mask)
        
        # ctx is (B, 1, H)
        rnn_input = torch.cat([emb, ctx], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)
        
        output = output.squeeze(1)  # (B, H)
        ctx = ctx.squeeze(1)        # (B, H)
        
        logits = self.fc_out(torch.cat([output, ctx], dim=-1))  # (B, V)
        return logits, hidden, attn

    def forward(self, tgt_inputs, hidden, encoder_outputs, src_mask):
        """完整解码（Teacher Forcing）"""
        B, T = tgt_inputs.shape
        logits_list = []
        input_tok = tgt_inputs[:, 0]
        
        for t in range(1, T):
            logits, hidden, _ = self.forward_step(input_tok, hidden, encoder_outputs, src_mask)
            logits_list.append(logits.unsqueeze(1))
            input_tok = tgt_inputs[:, t]  # Teacher Forcing
            
        return torch.cat(logits_list, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN, pad_idx: int = 0, sos_idx: int = 2, eos_idx: int = 3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, src, src_lengths, tgt, src_mask=None):
        """
        Args:
            src: (B, S) 源序列
            src_lengths: (B,) 源序列长度
            tgt: (B, T) 目标序列
            src_mask: (B, S) 源序列mask（可选，用于attention）
        """
        enc_outputs, hidden = self.encoder(src, src_lengths)
        
        # 根据encoder outputs的实际长度生成mask（pack后长度可能变短）
        enc_seq_len = enc_outputs.size(1)
        if src_mask is None or src_mask.size(1) != enc_seq_len:
            src_mask = torch.arange(enc_seq_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)
        
        logits = self.decoder(tgt, hidden, enc_outputs, src_mask)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src, src_lengths, max_len: int = 64) -> torch.Tensor:
        self.eval()
        enc_outputs, hidden = self.encoder(src, src_lengths)
        B = src.size(0)
        # 注意：pad_packed_sequence后enc_outputs长度可能小于原始src长度
        enc_seq_len = enc_outputs.size(1)
        src_mask = torch.arange(enc_seq_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)
        
        preds = torch.full((B,), self.sos_idx, dtype=torch.long, device=src.device)
        outputs = []
        for _ in range(max_len):
            logits, hidden, _ = self.decoder.forward_step(preds, hidden, enc_outputs, src_mask)
            preds = torch.argmax(logits, dim=-1)
            outputs.append(preds.unsqueeze(1))
            if (preds == self.eos_idx).all():
                break
        return torch.cat(outputs, dim=1) if outputs else torch.empty(B, 0, dtype=torch.long, device=src.device)

    @torch.no_grad()
    def beam_search(
        self,
        src,
        src_lengths,
        beam_size: int = 5,
        max_len: int = 80,
        alpha: float = 0.6,
        repetition_penalty: float = 1.15,
        block_trigram: bool = True,
    ) -> torch.Tensor:
        """
        改进版 beam search:
        - repetition_penalty: 对已经生成过的token降低概率
        - block_trigram: trigram blocking
        """
        self.eval()
        assert src.size(0) == 1, "beam_search currently only supports batch_size=1"
        enc_outputs, hidden = self.encoder(src, src_lengths)
        # 注意：pad_packed_sequence后enc_outputs长度可能小于原始src长度
        enc_seq_len = enc_outputs.size(1)
        src_mask = torch.arange(enc_seq_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)

        def length_penalty(seq_len: int) -> float:
            return ((5.0 + seq_len) / 6.0) ** alpha

        def has_repeat_trigram(seq, new_tok: int) -> bool:
            if not block_trigram or len(seq) < 3:
                return False
            tri = (seq[-2], seq[-1], new_tok)
            for i in range(len(seq) - 2):
                if (seq[i], seq[i + 1], seq[i + 2]) == tri:
                    return True
            return False

        beams = [(0.0, [self.sos_idx], hidden)]
        finished = []
        for _ in range(max_len):
            new_beams = []
            for logprob, seq, h in beams:
                if seq[-1] == self.eos_idx:
                    finished.append((logprob, seq, h))
                    continue
                last_tok = torch.tensor([seq[-1]], device=src.device)
                logits, new_h, _ = self.decoder.forward_step(last_tok, h, enc_outputs, src_mask)
                # repetition penalty
                if repetition_penalty and repetition_penalty > 1.0:
                    for tok in set(seq):
                        logits[0, tok] = logits[0, tok] / repetition_penalty
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                topk_logprob, topk_idx = torch.topk(log_probs, beam_size)
                for lp, idx in zip(topk_logprob.tolist(), topk_idx.tolist()):
                    if has_repeat_trigram(seq, int(idx)):
                        continue
                    new_seq = seq + [int(idx)]
                    new_beams.append((logprob + float(lp), new_seq, new_h))
            if not new_beams:
                break
            all_beams = finished + new_beams
            all_beams.sort(key=lambda x: x[0] / length_penalty(len(x[1]) - 1), reverse=True)
            beams = all_beams[:beam_size]
            if all(seq[-1] == self.eos_idx for _, seq, _ in beams):
                break
        best = max(beams, key=lambda x: (x[1][-1] == self.eos_idx, x[0] / length_penalty(len(x[1]) - 1)))[1][1:]
        return torch.tensor(best, dtype=torch.long, device=src.device).unsqueeze(0)

