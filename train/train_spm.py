"""
训练 SentencePiece (BPE) 分词模型
用于解决词级别分词导致的词表稀疏和 <unk> 问题
"""
import os
import sys
import argparse
from io import StringIO
import sentencepiece as spm

# 添加父目录到路径，以便导入 data_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_jsonl


def train_spm(data_dir, vocab_size, model_prefix, is_zh=True):
    """
    训练 SentencePiece 模型
    
    Args:
        data_dir: 数据目录
        vocab_size: 词表大小（推荐8k-16k）
        model_prefix: 模型文件前缀（完整路径，如 "spm/spm_zh"）
        is_zh: 是否为中文
    """
    # 1. 提取文本
    train_file = os.path.join(data_dir, "train_100k.jsonl")
    if not os.path.exists(train_file):
        train_file = os.path.join(data_dir, "train_10k.jsonl")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练文件不存在: {train_file}")
    
    data = load_jsonl(train_file)
    # 临时文件放在与模型相同的目录
    text_file = f"{model_prefix}_corpus.txt"
    
    print(f"正在提取文本到 {text_file}...")
    with open(text_file, "w", encoding="utf-8") as f:
        for item in data:
            # 根据语言选择字段
            text = item['zh'] if is_zh else item['en']
            f.write(text + "\n")
    
    print(f"已提取 {len(data)} 条文本")
    
    # 2. 训练SentencePiece模型
    # model_type=bpe, character_coverage=0.9995(en) / 0.995(zh)
    coverage = 0.995 if is_zh else 1.0
    lang_name = "中文" if is_zh else "英文"
    print(f"训练 {lang_name} SentencePiece 模型 (vocab_size={vocab_size})...", end=" ", flush=True)
    
    # 重定向stderr以隐藏详细配置信息
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=coverage,
        model_type='bpe',
        # <unk>, <sos>, <eos> 是 SentencePiece 的默认控制符号，不能放在 user_defined_symbols 中
        # 只需要添加 <pad> 作为用户定义符号
        user_defined_symbols=['<pad>'],
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece='<pad>', unk_piece='<unk>', bos_piece='<sos>', eos_piece='<eos>'
    )
    
    # 恢复stderr
    sys.stderr = old_stderr
    
    # 清理临时文件
    os.remove(text_file)
    print(f"完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 SentencePiece 分词模型")
    parser.add_argument("--data_dir", type=str, 
                       default="data_raw/AP0004_Midterm&Final_translation_dataset_zh_en",
                       help="数据目录")
    parser.add_argument("--vocab_size", type=int, default=8000,
                       help="词表大小（推荐8k-16k，对100k数据比较合适）")
    parser.add_argument("--output_dir", type=str, default="spm",
                       help="模型输出目录")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("训练 SentencePiece 分词模型")
    print("="*80)
    
    # 构建模型文件路径（包含目录）
    spm_dir = args.output_dir
    spm_zh_path = os.path.join(spm_dir, "spm_zh")
    spm_en_path = os.path.join(spm_dir, "spm_en")
    
    # 训练中文和英文的 SentencePiece 模型
    train_spm(args.data_dir, args.vocab_size, spm_zh_path, is_zh=True)
    train_spm(args.data_dir, args.vocab_size, spm_en_path, is_zh=False)
    
    print("\n" + "="*80)
    print("所有 SentencePiece 模型训练完成！")
    print("="*80)
    print(f"生成的文件（保存在 {spm_dir}/ 目录）:")
    print(f"  - {spm_dir}/spm_zh.model")
    print(f"  - {spm_dir}/spm_zh.vocab")
    print(f"  - {spm_dir}/spm_en.model")
    print(f"  - {spm_dir}/spm_en.vocab")
    print("="*80)

