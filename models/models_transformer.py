import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / norm


def make_norm(norm_type: str, dim: int):
    norm_type = norm_type.lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm type: {norm_type}")


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding with auto-extend for long sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.register_buffer("pe", self._build_pe(max_len), persistent=False)

    def _build_pe(self, max_len: int, device=None) -> torch.Tensor:
        position = torch.arange(0, max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(max_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        if x.size(1) > self.pe.size(1):
            # 动态扩展以覆盖更长序列（例如包含SOS/EOS后长度超出max_len）
            self.pe = self._build_pe(x.size(1), device=x.device)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        return self.dropout(x + self.pe(positions))


class RelativePositionBias(nn.Module):
    """T5-style relative position bias for simplicity."""

    def __init__(self, num_heads: int, max_distance: int):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.embedding = nn.Embedding(2 * max_distance - 1, num_heads)

    def forward(self, q_len: int, k_len: int) -> torch.Tensor:
        # 确保索引张量与 embedding 权重在同一设备，避免 CPU/CUDA 混用错误
        device = self.embedding.weight.device
        context_position = torch.arange(q_len, device=device)[:, None]
        memory_position = torch.arange(k_len, device=device)[None, :]
        relative_position = memory_position - context_position  # (q, k)
        relative_position = relative_position.clamp(-self.max_distance + 1, self.max_distance - 1)
        relative_position = relative_position + self.max_distance - 1
        values = self.embedding(relative_position)  # (q, k, heads)
        return values.permute(2, 0, 1).unsqueeze(0)  # (1, H, q, k)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        relative_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Q, _ = query.shape
        _, K, _ = key.shape

        q = self.q_proj(query).view(B, Q, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, Q, Dh)
        k = self.k_proj(key).view(B, K, self.num_heads, self.d_head).transpose(1, 2)   # (B, H, K, Dh)
        v = self.v_proj(value).view(B, K, self.num_heads, self.d_head).transpose(1, 2) # (B, H, K, Dh)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, Q, K)

        if relative_bias is not None:
            scores = scores + relative_bias  # broadcast over batch

        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, Q, Dh)
        out = out.transpose(1, 2).contiguous().view(B, Q, self.d_model)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        # 使用GELU激活函数，提升模型性能
        self.w1 = nn.Linear(d_model, dim_ff)
        self.w2 = nn.Linear(dim_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU激活函数
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float, norm_type: str, use_relative_bias: bool):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.use_relative_bias = use_relative_bias
        self.norm1 = make_norm(norm_type, d_model)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.norm2 = make_norm(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask, rel_bias):
        # 优化：修改为真正的 Pre-Norm 架构 (x + Sublayer(Norm(x)))
        # 1. Self Attention
        residual = x
        x = self.norm1(x) # 先 Norm
        attn_out = self.self_attn(x, x, x, key_padding_mask=padding_mask, relative_bias=rel_bias)
        x = residual + self.dropout(attn_out) # 后加 Residual
        
        # 2. Feed Forward
        residual = x
        x = self.norm2(x) # 先 Norm
        ff_out = self.ff(x)
        x = residual + ff_out # FF 内部已有 dropout
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float, norm_type: str, use_relative_bias: bool):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.use_relative_bias = use_relative_bias
        self.norm1 = make_norm(norm_type, d_model)
        self.norm2 = make_norm(norm_type, d_model)
        self.norm3 = make_norm(norm_type, d_model)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_padding_mask, tgt_padding_mask, causal_mask, rel_bias_self):
        # 优化：修改为真正的 Pre-Norm 架构
        
        # 1. Self Attention
        residual = x
        x = self.norm1(x)
        attn_out = self.self_attn(
            x, x, x, key_padding_mask=tgt_padding_mask, attn_mask=causal_mask, relative_bias=rel_bias_self
        )
        x = residual + self.dropout(attn_out)
        
        # 2. Cross Attention
        residual = x
        x = self.norm2(x)
        # 注意：Cross Attention 的 query 来自 decoder (x)，但 key/value 来自 encoder (enc_out)
        # enc_out 不需要在这里 norm，因为它在 encoder 出来时已经 norm 过了 (或者在 decoder 开始前统一 norm)
        cross_out = self.cross_attn(x, enc_out, enc_out, key_padding_mask=src_padding_mask)
        x = residual + self.dropout(cross_out)
        
        # 3. Feed Forward
        residual = x
        x = self.norm3(x)
        ff_out = self.ff(x)
        x = residual + ff_out
        
        return x


class TransformerNMT(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 256,
        pos_encoding: str = "sinusoidal",  # sinusoidal | learned | relative
        norm_type: str = "layernorm",  # layernorm | rmsnorm
        pad_idx: int = 0,
        sos_idx: int = 2,
        eos_idx: int = 3,
        share_decoder_embeddings: bool = True,  # 新增参数：默认共享输出权重
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.d_model = d_model  # 保存 d_model 以便缩放
        self.max_len = max_len
        pos_len = max_len + 50
        self.pos_encoding_type = pos_encoding

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        
        # 优化：使用 Xavier 初始化，更有利于 Transformer 收敛
        nn.init.xavier_uniform_(self.src_embed.weight)
        nn.init.xavier_uniform_(self.tgt_embed.weight)
        self.src_embed.weight.data[pad_idx].zero_()
        self.tgt_embed.weight.data[pad_idx].zero_()
        self.dropout = nn.Dropout(dropout)

        self.positional = None
        if pos_encoding == "sinusoidal":
            self.positional = PositionalEncoding(d_model, pos_len, dropout)
        elif pos_encoding == "learned":
            self.positional = LearnedPositionalEmbedding(pos_len, d_model, dropout)
        elif pos_encoding == "relative":
            self.relative_bias = RelativePositionBias(num_heads, pos_len)
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, dim_ff, dropout, norm_type, pos_encoding == "relative")
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, dim_ff, dropout, norm_type, pos_encoding == "relative")
                for _ in range(num_decoder_layers)
            ]
        )
        
        self.norm = make_norm(norm_type, d_model)
        
        # Output Projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size, bias=False)
        
        # 保存权重共享标志
        self.share_decoder_embeddings = share_decoder_embeddings
        
        # 先初始化参数
        self._reset_parameters()
        
        # 修改核心：权重共享 (Weight Tying)
        # 将 Decoder Embedding 的权重共享给 fc_out
        # 这对于小数据量训练至关重要
        # 注意：必须在_reset_parameters()之后设置，否则会被重新初始化覆盖
        if share_decoder_embeddings:
            self.fc_out.weight = self.tgt_embed.weight
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _add_position(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_encoding_type in ("sinusoidal", "learned"):
            return self.positional(x)
        return self.dropout(x)  # relative uses bias only

    def _causal_mask(self, size: int, device) -> torch.Tensor:
        # 1 for allowed, 0 for masked
        mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        src_padding = src_mask
        tgt_padding = tgt_inp.ne(self.pad_idx)

        # ================== 核心修复：增加 math.sqrt(self.d_model) 缩放 ==================
        # 如果不乘这个系数，位置编码（范围-1到1）会掩盖 Embedding 的语义信息
        src_emb_val = self.src_embed(src) * math.sqrt(self.d_model)
        tgt_emb_val = self.tgt_embed(tgt_inp) * math.sqrt(self.d_model)
        
        src_emb = self._add_position(src_emb_val)
        tgt_emb = self._add_position(tgt_emb_val)
        # ==============================================================================

        rel_bias_src = None
        rel_bias_tgt = None
        if self.pos_encoding_type == "relative":
            rel_bias_src = self.relative_bias(src_emb.size(1), src_emb.size(1))  # (1,H,S,S)
            rel_bias_tgt = self.relative_bias(tgt_emb.size(1), tgt_emb.size(1))

        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_padding, rel_bias_src)

        causal_mask = self._causal_mask(tgt_emb.size(1), tgt_emb.device)

        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_padding, tgt_padding, causal_mask, rel_bias_tgt)
        
        dec_out = self.norm(dec_out)
        logits = self.fc_out(dec_out)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, src_mask: torch.Tensor, max_len: int = 64) -> torch.Tensor:
        self.eval()
        src_padding = src_mask
        
        # 修复缩放
        src_emb_val = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self._add_position(src_emb_val)
        rel_bias_src = None
        if self.pos_encoding_type == "relative":
            rel_bias_src = self.relative_bias(src_emb.size(1), src_emb.size(1))

        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_padding, rel_bias_src)

        ys = torch.full((src.size(0), 1), self.sos_idx, device=src.device, dtype=torch.long)
        for _ in range(max_len):
            # 修复缩放
            tgt_emb_val = self.tgt_embed(ys) * math.sqrt(self.d_model)
            tgt_emb = self._add_position(tgt_emb_val)
            rel_bias_tgt = None
            if self.pos_encoding_type == "relative":
                rel_bias_tgt = self.relative_bias(tgt_emb.size(1), tgt_emb.size(1))
            tgt_padding = ys.ne(self.pad_idx)
            causal_mask = self._causal_mask(ys.size(1), ys.device)
            dec_out = tgt_emb
            for layer in self.decoder_layers:
                dec_out = layer(dec_out, enc_out, src_padding, tgt_padding, causal_mask, rel_bias_tgt)
            
            dec_out = self.norm(dec_out)
            prob = self.fc_out(dec_out[:, -1])
            next_tok = torch.argmax(prob, dim=-1)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            # 检查是否所有序列都生成了EOS
            if (next_tok == self.eos_idx).all():
                break
        return ys[:, 1:]  # drop SOS

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        beam_size: int = 5,
        max_len: int = 80,
        alpha: float = 0.6,
        repetition_penalty: float = 1.10,
        block_trigram: bool = True,
    ) -> torch.Tensor:
        self.eval()
        assert src.size(0) == 1, "beam_search currently supports batch_size=1"

        src_padding = src_mask
        src_emb_val = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self._add_position(src_emb_val)
        rel_bias_src = None
        if self.pos_encoding_type == "relative":
            rel_bias_src = self.relative_bias(src_emb.size(1), src_emb.size(1))

        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_padding, rel_bias_src)

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

        beams = [(0.0, [self.sos_idx])]
        finished = []
        for _ in range(max_len):
            new_beams = []
            for logprob, seq in beams:
                if seq[-1] == self.eos_idx:
                    finished.append((logprob, seq))
                    continue
                ys = torch.tensor(seq, device=src.device).unsqueeze(0)
                tgt_emb_val = self.tgt_embed(ys) * math.sqrt(self.d_model)
                tgt_emb = self._add_position(tgt_emb_val)
                rel_bias_tgt = None
                if self.pos_encoding_type == "relative":
                    rel_bias_tgt = self.relative_bias(tgt_emb.size(1), tgt_emb.size(1))
                tgt_padding = ys.ne(self.pad_idx)
                causal_mask = self._causal_mask(ys.size(1), ys.device)
                dec_out = tgt_emb
                for layer in self.decoder_layers:
                    dec_out = layer(dec_out, enc_out, src_padding, tgt_padding, causal_mask, rel_bias_tgt)
                dec_out = self.norm(dec_out)
                logits = self.fc_out(dec_out[:, -1])  # (1, V)
                # repetition penalty
                if repetition_penalty and repetition_penalty > 1.0:
                    for tok in set(seq):
                        logits[0, tok] = logits[0, tok] / repetition_penalty
                log_probs = F.log_softmax(logits, dim=-1)  # (1, V)
                topk_logprob, topk_idx = torch.topk(log_probs, beam_size, dim=-1)
                topk_logprob = topk_logprob.squeeze(0)
                topk_idx = topk_idx.squeeze(0)
                for lp, idx in zip(topk_logprob.tolist(), topk_idx.tolist()):
                    if has_repeat_trigram(seq, int(idx)):
                        continue
                    new_seq = seq + [int(idx)]
                    new_beams.append((logprob + float(lp), new_seq))
            if not new_beams:
                break
            all_beams = finished + new_beams
            all_beams.sort(key=lambda x: x[0] / length_penalty(len(x[1]) - 1), reverse=True)
            beams = all_beams[:beam_size]
            if all(seq[-1] == self.eos_idx for _, seq in beams):
                break
        best = max(beams, key=lambda x: (x[1][-1] == self.eos_idx, x[0] / length_penalty(len(x[1]) - 1)))[1][1:]
        return torch.tensor(best, device=src.device).unsqueeze(0)


