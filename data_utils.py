import json
import warnings
import re  # 增加正则表达式
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable
import os

# 1. 优先过滤警告（必须在导入 jieba 之前）
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")

# 2. 然后再导入 jieba 和 sentencepiece
import jieba
from torch.utils.data import Dataset
import sentencepiece as spm

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: data.append(json.loads(line))
    return data

def clean_text(text: str, remove_illegal: bool = True) -> str:
    """
    数据清洗：去除非法字符，处理异常字符
    
    Args:
        text: 输入文本
        remove_illegal: 是否去除非法字符（控制字符、零宽字符等）
    
    Returns:
        清洗后的文本
    """
    if not text:
        return ""
    
    # 去除首尾空白
    text = text.strip()
    
    if remove_illegal:
        # 去除控制字符（保留换行符、制表符等常用空白字符）
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 去除零宽字符
        text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
        
        # 去除其他异常Unicode字符（保留常用中英文、标点、数字）
        # 允许：中文字符、英文字母、数字、常用标点、空格
        text = re.sub(r'[^\u4e00-\u9fff\w\s.,!?;:()\[\]{}\'"\-—–…。，、；：？！""''（）【】《》]', '', text)
    
    # 规范化空白字符（多个空格合并为一个）
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def tokenize_zh(text: str, clean: bool = True) -> List[str]:
    """中文分词（使用Jieba）"""
    if clean:
        text = clean_text(text, remove_illegal=True)
    return list(jieba.cut(text.strip()))


def tokenize_en(text: str, clean: bool = True) -> List[str]:
    """英文分词（使用正则分离标点符号）"""
    if clean:
        text = clean_text(text, remove_illegal=True)
    # 改进：使用正则分离标点符号
    text = text.strip().lower()
    text = re.sub(r"([.,!?\"':;])", r" \1 ", text) # 在标点前后加空格
    text = re.sub(r"\s+", " ", text) # 合并多余空格
    return text.strip().split()

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, token_seqs: Iterable[List[str]], min_freq: int = 3, max_size: int = 30000):
        from collections import Counter
        counter = Counter()
        for seq in token_seqs:
            counter.update(seq)

        specials = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
        itos = list(specials)
        # 改进：提高min_freq到3，限制max_size到30000，解决词表稀疏问题
        for token, freq in counter.most_common():
            if freq < min_freq: continue
            if token in specials: continue
            if len(itos) >= max_size: break
            itos.append(token)
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, tokens: List[str], add_sos_eos: bool = True) -> List[int]:
        ids = []
        if add_sos_eos: ids.append(self.stoi[SOS_TOKEN])
        for t in tokens:
            ids.append(self.stoi.get(t, self.stoi[UNK_TOKEN]))
        if add_sos_eos: ids.append(self.stoi[EOS_TOKEN])
        return ids

    def decode(self, ids: List[int], remove_special: bool = True) -> List[str]:
        tokens = []
        for i in ids:
            if 0 <= i < len(self.itos):
                tok = self.itos[i]
                if remove_special and tok in {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}: continue
                tokens.append(tok)
        return tokens


# 新增：SentencePiece 包装类，使其接口与 Vocab 兼容
class SPMVocabWrapper:
    def __init__(self, sp_model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        # SentencePiece 直接处理 ID，itos/stoi 仅用于兼容性检查
        self.itos = [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]
        # 为了兼容性，创建 stoi
        self.stoi = {piece: i for i, piece in enumerate(self.itos)}
    
    def encode(self, text: str, add_sos_eos: bool = True) -> List[int]:
        # SPM 直接输入字符串，不需要预分词列表
        # 如果传入的是列表（兼容旧接口），合并成字符串
        if isinstance(text, list): 
            text = "".join(text) if isinstance(text[0], str) else " ".join(text)
        
        ids = self.sp.encode_as_ids(text)
        if add_sos_eos:
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        return ids
    
    def decode(self, ids: List[int], remove_special: bool = True) -> List[str]:
        # decode_ids 返回字符串，但为了兼容 evaluator 的 split 逻辑，返回 token piece list
        tokens = [self.sp.id_to_piece(i) for i in ids]
        if remove_special:
            tokens = [t for t in tokens if t not in ['<pad>', '<sos>', '<eos>', '<unk>']]
        return tokens
    
    # 额外方法：直接解码成句子
    def decode_to_sentence(self, ids: List[int]) -> str:
        # 过滤特殊token
        filtered_ids = [i for i in ids if i not in [0, 1, 2, 3]]
        return self.sp.decode_ids(filtered_ids)

class TranslationDataset(Dataset):
    def __init__(self, data: List[Dict], src_lang: str = "zh", tgt_lang: str = "en", 
                 src_vocab = None, tgt_vocab = None, max_len: int = 128,
                 use_spm: bool = False):
        self.src_tokens = []  # 如果用 SPM，这里存原始字符串
        self.tgt_tokens = []
        self.use_spm = use_spm
        
        # 尝试加载 SPM 模型
        if use_spm and src_vocab is None:
            # 检查 spm 目录中的模型文件
            spm_dir = "spm"
            spm_zh_path = os.path.join(spm_dir, "spm_zh.model")
            spm_en_path = os.path.join(spm_dir, "spm_en.model")
            
            if os.path.exists(spm_zh_path) and os.path.exists(spm_en_path):
                print("Loading SentencePiece models...")
                self.src_vocab = SPMVocabWrapper(spm_zh_path if src_lang == "zh" else spm_en_path)
                self.tgt_vocab = SPMVocabWrapper(spm_en_path if tgt_lang == "en" else spm_zh_path)
                src_vocab = self.src_vocab
                tgt_vocab = self.tgt_vocab
            else:
                print(f"Warning: {spm_zh_path} or {spm_en_path} not found. Falling back to basic tokenization.")
                self.use_spm = False
        
        if not self.use_spm:
            # 传统分词模式（使用数据清洗）
            for item in data:
                # 数据清洗：去除非法字符
                src_text = clean_text(item[src_lang], remove_illegal=True)
                tgt_text = clean_text(item[tgt_lang], remove_illegal=True)
                
                s_tok = tokenize_zh(src_text, clean=False) if src_lang == "zh" else tokenize_en(src_text, clean=False)
                t_tok = tokenize_zh(tgt_text, clean=False) if tgt_lang == "zh" else tokenize_en(tgt_text, clean=False)
                
                if 0 < len(s_tok) <= max_len and 0 < len(t_tok) <= max_len:
                    self.src_tokens.append(s_tok)
                    self.tgt_tokens.append(t_tok)
            # 改进：min_freq 设为 2，限制max_size到15000，避免过于稀疏
            self.src_vocab = src_vocab or Vocab.build(self.src_tokens, min_freq=2, max_size=15000)
            self.tgt_vocab = tgt_vocab or Vocab.build(self.tgt_tokens, min_freq=2, max_size=15000)
        else:
            # SPM 模式下，直接存原始文本，编码时处理（使用数据清洗）
            for item in data:
                # 数据清洗：去除非法字符
                src_text = clean_text(item[src_lang], remove_illegal=True)
                tgt_text = clean_text(item[tgt_lang], remove_illegal=True)
                
                # 简单过滤超长句子 (粗略估计)
                if len(src_text) > max_len * 2 or len(tgt_text) > max_len * 2:
                    continue
                self.src_tokens.append(src_text)
                self.tgt_tokens.append(tgt_text)
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

    def __len__(self): return len(self.src_tokens)
    def __getitem__(self, idx):
        if self.use_spm:
            # SPM模式下，直接传入原始字符串
            return {
                "src_ids": self.src_vocab.encode(self.src_tokens[idx]),
                "tgt_ids": self.tgt_vocab.encode(self.tgt_tokens[idx])
            }
        else:
            # 传统模式，传入token列表
            return {
                "src_ids": self.src_vocab.encode(self.src_tokens[idx]),
                "tgt_ids": self.tgt_vocab.encode(self.tgt_tokens[idx])
            }

def collate_fn(batch: List[Dict]) -> Tuple:
    import torch
    src_seqs = [b["src_ids"] for b in batch]
    tgt_seqs = [b["tgt_ids"] for b in batch]
    max_src, max_tgt = max(len(s) for s in src_seqs), max(len(t) for t in tgt_seqs)
    
    src_b, tgt_b, src_m, tgt_m = [], [], [], []
    for s, t in zip(src_seqs, tgt_seqs):
        src_b.append(s + [0] * (max_src - len(s)))
        tgt_b.append(t + [0] * (max_tgt - len(t)))
        src_m.append([1] * len(s) + [0] * (max_src - len(s)))
        tgt_m.append([1] * len(t) + [0] * (max_tgt - len(t)))
    return (torch.tensor(src_b), torch.tensor(tgt_b), torch.tensor(src_m, dtype=torch.bool), torch.tensor(tgt_m, dtype=torch.bool))