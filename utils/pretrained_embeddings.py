"""
预训练词向量加载工具
支持 GloVe、Word2Vec 等格式
"""
import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np


def load_glove_embeddings(glove_path: str, vocab: 'Vocab', embed_dim: int) -> Optional[torch.Tensor]:
    """
    加载 GloVe 预训练词向量
    
    Args:
        glove_path: GloVe 词向量文件路径（如 glove.6B.100d.txt）
        vocab: 词汇表对象（包含 stoi 和 itos）
        embed_dim: 嵌入维度（必须与 GloVe 文件中的维度匹配）
    
    Returns:
        预训练词向量矩阵 (vocab_size, embed_dim)，如果加载失败返回 None
    """
    if not os.path.exists(glove_path):
        print(f"Warning: GloVe file not found: {glove_path}")
        return None
    
    print(f"Loading GloVe embeddings from {glove_path}...")
    embeddings_dict = {}
    
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                if len(values) < embed_dim + 1:
                    continue
                word = values[0]
                vector = np.array([float(x) for x in values[1:embed_dim+1]])
                if len(vector) == embed_dim:
                    embeddings_dict[word] = vector
        
        print(f"Loaded {len(embeddings_dict)} word vectors from GloVe")
    except Exception as e:
        print(f"Error loading GloVe embeddings: {e}")
        return None
    
    # 构建词向量矩阵
    vocab_size = len(vocab.itos)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))
    embedding_matrix = embedding_matrix.astype('float32')
    
    # 匹配词汇表中的词
    matched = 0
    for i, word in enumerate(vocab.itos):
        if word in embeddings_dict:
            embedding_matrix[i] = embeddings_dict[word]
            matched += 1
        elif word.lower() in embeddings_dict:
            # 尝试小写形式
            embedding_matrix[i] = embeddings_dict[word.lower()]
            matched += 1
    
    print(f"Matched {matched}/{vocab_size} words from vocabulary")
    
    # 特殊token处理：PAD设为0，其他保持随机初始化
    pad_idx = vocab.stoi.get('<pad>', 0)
    if pad_idx < vocab_size:
        embedding_matrix[pad_idx] = 0.0
    
    return torch.from_numpy(embedding_matrix)


def init_embedding_with_pretrained(
    embedding_layer: nn.Embedding,
    pretrained_embeddings: torch.Tensor,
    freeze: bool = False
) -> nn.Embedding:
    """
    使用预训练词向量初始化嵌入层
    
    Args:
        embedding_layer: 要初始化的嵌入层
        pretrained_embeddings: 预训练词向量矩阵 (vocab_size, embed_dim)
        freeze: 是否冻结嵌入层参数（不进行微调）
    
    Returns:
        初始化后的嵌入层
    """
    vocab_size, embed_dim = embedding_layer.weight.shape
    pretrained_vocab_size, pretrained_embed_dim = pretrained_embeddings.shape
    
    if embed_dim != pretrained_embed_dim:
        print(f"Warning: Embedding dimension mismatch: {embed_dim} vs {pretrained_embed_dim}")
        print("Using random initialization instead")
        return embedding_layer
    
    if vocab_size != pretrained_vocab_size:
        print(f"Warning: Vocabulary size mismatch: {vocab_size} vs {pretrained_vocab_size}")
        print("Copying available embeddings...")
        min_size = min(vocab_size, pretrained_vocab_size)
        embedding_layer.weight.data[:min_size] = pretrained_embeddings[:min_size]
    else:
        embedding_layer.weight.data = pretrained_embeddings
    
    if freeze:
        embedding_layer.weight.requires_grad = False
        print("Embedding layer frozen (not trainable)")
    else:
        print("Embedding layer initialized with pre-trained vectors (trainable)")
    
    return embedding_layer


def load_word2vec_embeddings(word2vec_path: str, vocab: 'Vocab', embed_dim: int) -> Optional[torch.Tensor]:
    """
    加载 Word2Vec 格式的词向量（使用 gensim，如果可用）
    
    Args:
        word2vec_path: Word2Vec 模型文件路径（.bin 或 .txt）
        vocab: 词汇表对象
        embed_dim: 嵌入维度
    
    Returns:
        预训练词向量矩阵，如果加载失败返回 None
    """
    try:
        from gensim.models import KeyedVectors
    except ImportError:
        print("Warning: gensim not installed. Cannot load Word2Vec embeddings.")
        print("Install with: pip install gensim")
        return None
    
    if not os.path.exists(word2vec_path):
        print(f"Warning: Word2Vec file not found: {word2vec_path}")
        return None
    
    print(f"Loading Word2Vec embeddings from {word2vec_path}...")
    
    try:
        # 尝试加载二进制格式
        try:
            model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        except:
            # 尝试加载文本格式
            model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
        
        if model.vector_size != embed_dim:
            print(f"Warning: Dimension mismatch: {embed_dim} vs {model.vector_size}")
            return None
        
        vocab_size = len(vocab.itos)
        embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))
        embedding_matrix = embedding_matrix.astype('float32')
        
        matched = 0
        for i, word in enumerate(vocab.itos):
            if word in model:
                embedding_matrix[i] = model[word]
                matched += 1
            elif word.lower() in model:
                embedding_matrix[i] = model[word.lower()]
                matched += 1
        
        print(f"Matched {matched}/{vocab_size} words from vocabulary")
        
        # 特殊token处理
        pad_idx = vocab.stoi.get('<pad>', 0)
        if pad_idx < vocab_size:
            embedding_matrix[pad_idx] = 0.0
        
        return torch.from_numpy(embedding_matrix)
    
    except Exception as e:
        print(f"Error loading Word2Vec embeddings: {e}")
        return None

