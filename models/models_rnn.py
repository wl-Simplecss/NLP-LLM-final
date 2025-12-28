from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 修改1: 启用双向 bidirectional=True
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers=num_layers, 
                          batch_first=True, dropout=dropout, bidirectional=True)
        
        # 修改2: 添加投影层，将双向hidden合并为单向hidden (2*H -> H)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src: (B, S)
        emb = self.dropout(self.embedding(src))
        
        # outputs: (B, S, hidden_size * 2)
        # hidden: (num_layers * 2, B, hidden_size)
        outputs, hidden = self.rnn(emb)
        
        # 处理 hidden state: 将双向层拼接并映射
        # 变换形状: (num_layers, 2, B, H) -> (num_layers, B, 2*H)
        hidden = hidden.view(self.num_layers, 2, hidden.size(1), self.hidden_size)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        
        # 投影回 decoder 需要的 hidden_size
        hidden = torch.tanh(self.fc_hidden(hidden))  # (num_layers, B, H)
        
        return outputs, hidden


class DotAttention(nn.Module):
    def forward(self, query, keys, values, mask=None):
        # 仅当 encoder也是单向时可用，否则维度不匹配会报错
        # 为了兼容，这里抛出一个错误（建议用General）
        raise ValueError("Dot Attention not supported for Bidirectional Encoder. Use 'general'.")


class GeneralAttention(nn.Module):
    def __init__(self, dec_hidden_size, enc_hidden_size):
        super().__init__()
        # 修改3: Wa 负责将 decoder query (H) 映射到 encoder key (2*H) 的空间
        self.Wa = nn.Linear(dec_hidden_size, enc_hidden_size, bias=False)

    def forward(self, query, keys, values, mask=None):
        # query: (B, 1, H_dec)
        # keys: (B, S, H_enc) where H_enc = 2 * H_dec
        
        d_k = keys.size(-1)
        q = self.Wa(query)  # (B, 1, H_enc)
        
        # (B, 1, H_enc) @ (B, H_enc, S) -> (B, 1, S)
        scores = torch.bmm(q, keys.transpose(1, 2)) / (d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        
        # (B, 1, S) @ (B, S, H_enc) -> (B, 1, H_enc)
        ctx = torch.bmm(attn, values)
        return ctx, attn


class AdditiveAttention(nn.Module):
    def __init__(self, dec_hidden_size, enc_hidden_size, attn_hidden_size=None):
        super().__init__()
        if attn_hidden_size is None:
            attn_hidden_size = dec_hidden_size
            
        # Wq 将 Decoder 的 hidden 映射到 attn空间
        self.Wq = nn.Linear(dec_hidden_size, attn_hidden_size, bias=False)
        # Wk 将 Encoder 的 outputs (2*H) 映射到 attn空间
        self.Wk = nn.Linear(enc_hidden_size, attn_hidden_size, bias=False)
        # v 计算评分
        self.v = nn.Linear(attn_hidden_size, 1, bias=False)

    def forward(self, query, keys, values, mask=None):
        # query: (B, 1, dec_H)
        # keys: (B, S, enc_H)
        # values: (B, S, enc_H)
        
        q = self.Wq(query)  # (B, 1, attn_H)
        k = self.Wk(keys)   # (B, S, attn_H)
        
        # 广播加法: (B, 1, attn_H) + (B, S, attn_H) -> (B, S, attn_H)
        scores = self.v(torch.tanh(q + k))  # (B, S, 1)
        scores = scores.transpose(1, 2)     # (B, 1, S)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        
        attn = F.softmax(scores, dim=-1) # (B, 1, S)
        
        ctx = torch.bmm(attn, values) # (B, 1, enc_H)
        return ctx, attn


def build_attention(dec_hidden_size: int, enc_hidden_size: int, attn_type: str):
    attn_type = attn_type.lower()
    if attn_type in ("general", "multiplicative", "dot"):
        return GeneralAttention(dec_hidden_size, enc_hidden_size)
    if attn_type in ("additive", "bahdanau"):
        # 修复：现在使用真正的 Additive Attention，并传入正确的维度
        return AdditiveAttention(dec_hidden_size, enc_hidden_size)
    raise ValueError(f"Unknown attention type: {attn_type}")


class DecoderRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        attn_type: str = "general",
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 编码器是双向的，所以 Context Vector 的维度是 2 * hidden_size
        self.enc_output_size = hidden_size * 2
        
        # GRU 输入: Embedding + Context
        self.rnn = nn.GRU(embed_dim + self.enc_output_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        self.attn = build_attention(hidden_size, self.enc_output_size, attn_type)
        
        # 输出层: Hidden + Context -> Vocab
        self.fc_out = nn.Linear(hidden_size + self.enc_output_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, hidden, encoder_outputs, src_mask):
        # input_tok: (B,)
        emb = self.dropout(self.embedding(input_tok).unsqueeze(1))  # (B, 1, E)
        
        # attention
        query = hidden[-1].unsqueeze(1)  # (B, 1, H) - 取最后一层的hidden
        ctx, attn = self.attn(query, encoder_outputs, encoder_outputs, src_mask)
        
        # ctx is (B, 1, 2*H)
        rnn_input = torch.cat([emb, ctx], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)
        
        output = output.squeeze(1)  # (B, H)
        ctx = ctx.squeeze(1)         # (B, 2*H)
        
        logits = self.fc_out(torch.cat([output, ctx], dim=-1))  # (B, V)
        return logits, hidden, attn

    def forward(self, tgt_inputs, hidden, encoder_outputs, src_mask):
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

    def forward(self, src, src_mask, tgt):
        enc_outputs, hidden = self.encoder(src, src_mask)
        logits = self.decoder(tgt, hidden, enc_outputs, src_mask)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src, src_mask, max_len: int = 64) -> torch.Tensor:
        self.eval()
        enc_outputs, hidden = self.encoder(src, src_mask)
        B = src.size(0)
        preds = torch.full((B,), self.sos_idx, dtype=torch.long, device=src.device)
        outputs = []
        for _ in range(max_len):
            logits, hidden, _ = self.decoder.forward_step(preds, hidden, enc_outputs, src_mask)
            preds = torch.argmax(logits, dim=-1)
            outputs.append(preds.unsqueeze(1))
            # 检查是否所有序列都生成了EOS
            if (preds == self.eos_idx).all():
                break
        return torch.cat(outputs, dim=1) if outputs else torch.empty(B, 0, dtype=torch.long, device=src.device)

    @torch.no_grad()
    def beam_search(
        self,
        src,
        src_mask,
        beam_size: int = 5,
        max_len: int = 80,
        alpha: float = 0.6,
        repetition_penalty: float = 1.15,
        block_trigram: bool = True,
    ) -> torch.Tensor:
        """
        改进版 beam search:
        - repetition_penalty: 对已经生成过的token降低概率，减少循环
        - block_trigram: trigram blocking，显著减少"重复短语"
        """
        self.eval()
        assert src.size(0) == 1, "beam_search currently only supports batch_size=1"
        enc_outputs, hidden = self.encoder(src, src_mask)

        def length_penalty(seq_len: int) -> float:
            return ((5.0 + seq_len) / 6.0) ** alpha

        def has_repeat_trigram(seq, new_tok: int) -> bool:
            if not block_trigram or len(seq) < 3:
                return False
            tri = (seq[-2], seq[-1], new_tok)
            # 是否在历史中出现过同样 trigram
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
                # repetition penalty（在logits上做）
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



