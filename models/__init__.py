"""
模型结构定义模块
"""
from .models_rnn import EncoderRNN, DecoderRNN, Seq2Seq
from .models_transformer import TransformerNMT

__all__ = ['EncoderRNN', 'DecoderRNN', 'Seq2Seq', 'TransformerNMT']

