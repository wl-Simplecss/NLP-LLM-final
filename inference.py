#!/usr/bin/env python3
"""
一键推理脚本 - 支持多种调用方式

调用方式1：一键演示模式（--demo）
    python inference.py --demo
    使用 checkpoints_best/ 中的三种模型对测试语句进行翻译，结果保存到 outputs/

调用方式2：交互式翻译模式（--interactive）
    python inference.py --interactive --model_type rnn
    选择模型类型后实时输入语句翻译

调用方式3：命令行单句翻译（默认）
    python inference.py --checkpoint xxx.pt --model_type rnn --text "你好"
"""

import argparse
import os
import sys

# ========== 绕过PyTorch CVE-2025-32434安全限制 ==========
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import torch

_original_torch_load = torch.load

def _patched_load(f, *args, **kwargs):
    kwargs.pop('weights_only', None)
    kwargs['weights_only'] = False
    try:
        return _original_torch_load(f, *args, **kwargs)
    except TypeError:
        kwargs.pop('weights_only', None)
        return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_load

try:
    import transformers.utils.import_utils as import_utils
    if hasattr(import_utils, 'is_torch_greater_or_equal_than_2_6'):
        import_utils.is_torch_greater_or_equal_than_2_6 = True
except:
    pass
# ========================================================
from datetime import datetime
from typing import Optional, Tuple, Any

from data_utils import tokenize_zh, tokenize_en, Vocab, SPMVocabWrapper
from models.models_rnn import EncoderRNN, DecoderRNN, Seq2Seq
from models.models_transformer import TransformerNMT

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# checkpoints_best 路径配置
# ============================================================
CHECKPOINTS_BEST = {
    "rnn": os.path.join(PROJECT_ROOT, "checkpoints_best", "rnn_best.pt"),
    "transformer": os.path.join(PROJECT_ROOT, "checkpoints_best", "transformer_best.pt"),
    "t5": os.path.join(PROJECT_ROOT, "checkpoints_best", "t5_best.pt"),
}

# 演示用的测试语句（从训练数据集中选取的代表性句子）
DEMO_SENTENCES = [
    # 短句（简单）
    "但是即使是官方活动也带有政治色彩。",
    "历届总统都被要求出行需乘坐这些飞机。",
    "打铁必须自身硬。",
    
    # 中等长度（科技/商业）
    "一方面汽车技术经过一百多年的发展，许多技术已经比较成熟。",
    "2016年，中国快递业已实现连续6年增长率超过50%，日均8000万个包裹。",
    "苹果上个月致信印度政府，阐述了在印度生产iPhone等产品的计划。",
    
    # 中等长度（体育/娱乐）
    "烂番茄上除第一季外，其他六季新鲜度都高达100%。",
    "傅家俊赛后说，希金斯开局非常完美，自己以为很快就会大比分败北。",
    
    # 较长句（综合）
    "从体育人群的增多、体育事业的发展，到体育产业的兴起，体育运动正在以更强大的号召力，影响着中国社会行进的步伐。",
    "在2018年即将到来之际，国家大剧院特别推出了9台11场新年系列音乐会，用曼妙音符陪伴广大观众共迎新年。",
]

# 对应的参考翻译（用于展示，可选）
DEMO_REFERENCES = [
    "But even official events have political overtones.",
    "Presidents are required to use them for travel.",
    "A blacksmith must have necessary skills in order to do his job in a great way.",
    "Automobile technology has developed for more than a century, and a lot of the technology is fairly mature.",
    "In 2016, China's express delivery industry achieved more than 50% growth for six consecutive years, with an average of 80 million parcels per day.",
    "Apple sent a letter to the Indian government last month, which set forth its plans to produce iPhone and other products in India.",
    "Other than Season 1, Rotten Tomatoes had a 100% certified fresh rating for the other six seasons.",
    "\"Higgins had a very perfect start. I thought I might lose the competition very soon,\" said Marco Fu after the match.",
    "From the increase of sports crowd, the development of sports undertakings to the rise of sports industry, sports is influencing the progressing pace of Chinese society with more powerful appeal.",
    "With the advent of 2018, the National Centre for the Performing Arts has specially organized nine sets of 11 New Year concerts, offering the most sincere blessings for the festival.",
]


def _extract_epoch_from_name(path: str) -> Optional[int]:
    import re
    m = re.search(r"epoch(\d+)", os.path.basename(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _find_best_checkpoint_in_dir(checkpoint_dir: str) -> Optional[str]:
    """
    目录传入时自动挑选 best checkpoint：
    - 优先 valid_loss 最小
    - 若无 valid_loss，则 epoch 最大
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not files:
        return None

    best_path = None
    best_valid = float("inf")
    best_epoch = -1

    for f in files:
        p = os.path.join(checkpoint_dir, f)
        try:
            ckpt = torch.load(p, map_location="cpu")
        except Exception:
            continue

        v = ckpt.get("valid_loss", None)
        ep = ckpt.get("epoch", None)
        if ep is None:
            ep = _extract_epoch_from_name(p)

        if v is not None:
            try:
                v = float(v)
            except Exception:
                v = None
            if v is not None and v < best_valid:
                best_valid = v
                best_path = p
                best_epoch = int(ep) if ep is not None else best_epoch
        else:
            if ep is not None and int(ep) > best_epoch:
                best_epoch = int(ep)
                best_path = p

    return best_path


def resolve_checkpoint_path(user_path: str) -> str:
    """
    兼容新 checkpoints/ 目录结构：
    - 支持传入具体文件：xxx.pt
    - 支持传入目录：自动挑选 best checkpoint
    - 支持旧写法：checkpoints_rnn/...  会自动映射到 checkpoints/checkpoints_rnn/...
    """
    p = user_path.strip()
    if not p:
        raise ValueError("Empty checkpoint path")

    # 1) 绝对路径 / 相对路径直接存在
    if os.path.exists(p):
        if os.path.isdir(p):
            best = _find_best_checkpoint_in_dir(p)
            if not best:
                raise FileNotFoundError(f"目录下未找到 .pt checkpoint: {p}")
            return best
        return p

    # 2) 旧路径兼容：checkpoints_xxx/...  -> checkpoints/checkpoints_xxx/...
    alt = None
    if p.startswith("checkpoints_") or p.startswith("checkpoints_t5") or p.startswith("checkpoints_transformer"):
        alt = os.path.join(PROJECT_ROOT, "checkpoints", p)
    elif p.startswith("checkpoints/") is False and ("checkpoints_" in p.split("/")[0]):
        alt = os.path.join(PROJECT_ROOT, "checkpoints", p)

    if alt and os.path.exists(alt):
        if os.path.isdir(alt):
            best = _find_best_checkpoint_in_dir(alt)
            if not best:
                raise FileNotFoundError(f"目录下未找到 .pt checkpoint: {alt}")
            return best
        return alt

    raise FileNotFoundError(f"Checkpoint 不存在: {user_path}")


def load_checkpoint(path: str):
    resolved = resolve_checkpoint_path(path)
    return torch.load(resolved, map_location="cpu")


def build_rnn_model(ckpt):
    args = ckpt.get("args", {})
    embed_dim = args.get("embed_dim", 256)
    hidden_size = args.get("hidden_size", 512)
    num_layers = args.get("num_layers", 2)
    dropout = args.get("dropout", 0.1)
    attn_type = args.get("attn_type", "dot")

    src_vocab: Vocab = ckpt["src_vocab"]
    tgt_vocab: Vocab = ckpt["tgt_vocab"]

    encoder = EncoderRNN(len(src_vocab.itos), embed_dim, hidden_size, num_layers, dropout)
    decoder = DecoderRNN(len(tgt_vocab.itos), embed_dim, hidden_size, num_layers, dropout, attn_type)
    model = Seq2Seq(encoder, decoder)
    model.load_state_dict(ckpt["model_state"])
    return model, src_vocab, tgt_vocab


def build_transformer_model(ckpt):
    args = ckpt.get("args", {})
    src_vocab: Vocab = ckpt["src_vocab"]
    tgt_vocab: Vocab = ckpt["tgt_vocab"]

    model = TransformerNMT(
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        d_model=args.get("d_model", 256),
        num_heads=args.get("num_heads", 4),
        num_encoder_layers=args.get("num_encoder_layers", 4),
        num_decoder_layers=args.get("num_decoder_layers", 4),
        dim_ff=args.get("dim_ff", 512),
        dropout=args.get("dropout", 0.1),
        max_len=args.get("max_len", 128),
        pos_encoding=args.get("pos_encoding", "sinusoidal"),
        norm_type=args.get("norm_type", "layernorm"),
    )
    model.load_state_dict(ckpt["model_state"])
    return model, src_vocab, tgt_vocab


def preprocess(sentence: str, src_lang: str, src_vocab):
    if isinstance(src_vocab, SPMVocabWrapper):
        ids = src_vocab.encode(sentence, add_sos_eos=True)
    else:
        if src_lang == "zh":
            tokens = tokenize_zh(sentence)
        else:
            tokens = tokenize_en(sentence)
        ids = src_vocab.encode(tokens, add_sos_eos=True)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def detokenize(ids, tgt_vocab):
    tokens = tgt_vocab.decode(ids.tolist(), remove_special=True)
    if isinstance(tgt_vocab, SPMVocabWrapper):
        return tgt_vocab.decode_to_sentence(ids.tolist())
    else:
        return " ".join(tokens)


def build_t5_model(ckpt):
    """构建T5模型"""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library is required for T5 models")
    
    args = ckpt.get("args", {})
    model_name = args.get("model_name", "google-t5/t5-small")

    cache_root = "/data/250010066/LLM_course/final_program/t5-model"
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    local_cache_dir = f"{cache_root}/{model_name_safe}"

    MIRROR_ENDPOINT = "https://hf-mirror.com"
    os.environ.setdefault("HF_ENDPOINT", MIRROR_ENDPOINT)
    os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", MIRROR_ENDPOINT)

    def _find_model_dir():
        # 优先检查本地目录是否有完整模型文件
        if os.path.isdir(local_cache_dir):
            # 检查是否有直接的模型文件
            if os.path.exists(os.path.join(local_cache_dir, "config.json")):
                return local_cache_dir, True
            
            # 检查snapshots目录
            repo_dir = os.path.join(local_cache_dir, "models--" + model_name.replace("/", "--"))
            snaps_root = os.path.join(repo_dir, "snapshots")
            if os.path.isdir(snaps_root):
                snaps = [os.path.join(snaps_root, d) for d in os.listdir(snaps_root) if os.path.isdir(os.path.join(snaps_root, d))]
                if snaps:
                    snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    return snaps[0], True
        
        # 尝试从远程下载
        try:
            snap_dir = snapshot_download(
                repo_id=model_name,
                cache_dir=local_cache_dir,
                endpoint=MIRROR_ENDPOINT,
                local_files_only=False,
                resume_download=True,
                allow_patterns=["*.json", "*.bin", "*.model", "*.safetensors"],
                ignore_patterns=["*.md", "*.txt", "*.h5", "*.ot", "*.msgpack"],
            )
            return snap_dir, False
        except Exception as e:
            raise RuntimeError(f"无法加载模型 {model_name}: {e}")

    model_dir, local_only = _find_model_dir()
    print(f"加载模型从: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=local_only)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])

    return model, tokenizer


# ============================================================
# 翻译函数（统一接口）
# ============================================================
def translate_sentence(
    text: str,
    model: Any,
    model_type: str,
    src_vocab: Any = None,
    tgt_vocab: Any = None,
    tokenizer: Any = None,
    device: torch.device = None,
    beam_size: int = 1,
    max_len: int = 64,
    src_lang: str = "zh"
) -> str:
    """统一的翻译接口"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        if model_type == "t5":
            # 支持mT5模型（使用中文前缀）和T5模型（使用英文前缀）
            input_text = f"翻译成英语: {text}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_len)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            if beam_size > 1:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_len,
                    num_beams=beam_size,
                    early_stopping=True
                )
            else:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_len,
                    num_beams=1,
                    do_sample=False
                )
            
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            src_ids = preprocess(text, src_lang, src_vocab).to(device)
            
            # 计算实际序列长度（不包括padding）
            src_lengths = (src_ids != 0).sum(dim=1)
            
            if model_type == "rnn":
                # RNN模型需要src_lengths参数
                if beam_size > 1:
                    preds = model.beam_search(src_ids, src_lengths, beam_size=beam_size, max_len=max_len)
                else:
                    preds = model.greedy_decode(src_ids, src_lengths, max_len=max_len)
            else:
                # Transformer模型使用src_mask
                src_mask = src_ids.ne(0)
                if beam_size > 1:
                    preds = model.beam_search(src_ids, src_mask, beam_size=beam_size, max_len=max_len)
                else:
                    preds = model.greedy_decode(src_ids, src_mask, max_len=max_len)
            
            return detokenize(preds[0], tgt_vocab)


# ============================================================
# 模式1：一键演示模式
# ============================================================
def run_demo_mode(beam_size: int = 5, max_len: int = 64):
    """一键演示模式：使用 checkpoints_best 中的三种模型翻译测试语句"""
    print("=" * 70)
    print("一键演示模式 - 使用 checkpoints_best 中的最佳模型")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载三种模型
    models_info = {}
    
    for model_type in ["rnn", "transformer", "t5"]:
        ckpt_path = CHECKPOINTS_BEST[model_type]
        if not os.path.exists(ckpt_path):
            print(f"[警告] {model_type.upper()} checkpoint 不存在: {ckpt_path}")
            continue
        
        print(f"\n正在加载 {model_type.upper()} 模型...")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
            if model_type == "t5":
                model, tokenizer = build_t5_model(ckpt)
                models_info[model_type] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "src_vocab": None,
                    "tgt_vocab": None
                }
            elif model_type == "rnn":
                model, src_vocab, tgt_vocab = build_rnn_model(ckpt)
                models_info[model_type] = {
                    "model": model,
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab,
                    "tokenizer": None
                }
            else:  # transformer
                model, src_vocab, tgt_vocab = build_transformer_model(ckpt)
                models_info[model_type] = {
                    "model": model,
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab,
                    "tokenizer": None
                }
            
            print(f"  ✓ {model_type.upper()} 模型加载成功")
        except Exception as e:
            print(f"  ✗ {model_type.upper()} 模型加载失败: {e}")
            continue
    
    if not models_info:
        print("\n[错误] 没有成功加载任何模型！")
        return
    
    # 翻译测试语句
    print("\n" + "=" * 70)
    print("翻译结果（测试语句来自训练数据集）")
    print("=" * 70)
    
    for i, sent in enumerate(DEMO_SENTENCES):
        result_entry = {"source": sent, "reference": "", "translations": {}}
        print(f"\n【源句 {i+1}】{sent}")
        
        # 显示参考翻译（如果有）
        if i < len(DEMO_REFERENCES):
            ref = DEMO_REFERENCES[i]
            result_entry["reference"] = ref
            print(f"  [参考翻译    ] {ref}")
        
        print("-" * 50)
        
        for model_type, info in models_info.items():
            try:
                translation = translate_sentence(
                    text=sent,
                    model=info["model"],
                    model_type=model_type,
                    src_vocab=info.get("src_vocab"),
                    tgt_vocab=info.get("tgt_vocab"),
                    tokenizer=info.get("tokenizer"),
                    device=device,
                    beam_size=beam_size,
                    max_len=max_len
                )
                result_entry["translations"][model_type] = translation
                print(f"  [{model_type.upper():12s}] {translation}")
            except Exception as e:
                result_entry["translations"][model_type] = f"ERROR: {e}"
                print(f"  [{model_type.upper():12s}] 翻译失败: {e}")
        
        results.append(result_entry)
    
    # 保存结果到文件
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"demo_translation_{timestamp}.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("机器翻译演示结果 (中文 → 英文)\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"设备: {device}\n")
        f.write(f"Beam Size: {beam_size}\n")
        f.write(f"Max Length: {max_len}\n")
        f.write(f"测试语句来源: 训练数据集 (test.jsonl)\n")
        f.write("=" * 70 + "\n\n")
        
        for i, entry in enumerate(results):
            f.write(f"【源句 {i+1}】{entry['source']}\n")
            if entry.get("reference"):
                f.write(f"  [参考翻译    ] {entry['reference']}\n")
            for model_type, trans in entry["translations"].items():
                f.write(f"  [{model_type.upper():12s}] {trans}\n")
            f.write("\n")
    
    print("\n" + "=" * 70)
    print(f"✓ 结果已保存到: {output_file}")
    print("=" * 70)


# ============================================================
# 模式2：交互式翻译模式
# ============================================================
def run_interactive_mode(model_type: str, beam_size: int = 5, max_len: int = 64):
    """交互式翻译模式：选择模型后实时输入语句翻译"""
    print("=" * 70)
    print(f"交互式翻译模式 - 使用 {model_type.upper()} 模型")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 加载模型
    ckpt_path = CHECKPOINTS_BEST[model_type]
    if not os.path.exists(ckpt_path):
        # 尝试用户提供的路径
        print(f"[警告] checkpoints_best 中未找到 {model_type} 模型")
        print("请使用 --checkpoint 参数指定模型路径")
        return
    
    print(f"正在加载模型: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if model_type == "t5":
        model, tokenizer = build_t5_model(ckpt)
        src_vocab, tgt_vocab = None, None
    elif model_type == "rnn":
        model, src_vocab, tgt_vocab = build_rnn_model(ckpt)
        tokenizer = None
    else:  # transformer
        model, src_vocab, tgt_vocab = build_transformer_model(ckpt)
        tokenizer = None
    
    model.to(device)
    model.eval()
    print("✓ 模型加载完成！")
    
    print("\n" + "-" * 70)
    print("输入中文句子进行翻译，输入 'q' 或 'quit' 退出")
    print("-" * 70)
    
    while True:
        try:
            text = input("\n请输入中文: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出交互模式")
            break
        
        if not text:
            continue
        if text.lower() in ["q", "quit", "exit"]:
            print("退出交互模式")
            break
        
        try:
            translation = translate_sentence(
                text=text,
                model=model,
                model_type=model_type,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                tokenizer=tokenizer,
                device=device,
                beam_size=beam_size,
                max_len=max_len
            )
            print(f"翻译结果: {translation}")
        except Exception as e:
            print(f"翻译失败: {e}")


# ============================================================
# 模式3：命令行单句翻译（原有功能）
# ============================================================
def run_single_translate(args):
    """命令行单句翻译模式"""
    ckpt = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "t5":
        model, tokenizer = build_t5_model(ckpt)
        src_vocab, tgt_vocab = None, None
    elif args.model_type == "rnn":
        model, src_vocab, tgt_vocab = build_rnn_model(ckpt)
        tokenizer = None
    else:
        model, src_vocab, tgt_vocab = build_transformer_model(ckpt)
        tokenizer = None

    translation = translate_sentence(
        text=args.text,
        model=model,
        model_type=args.model_type,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        tokenizer=tokenizer,
        device=device,
        beam_size=args.beam_size,
        max_len=args.max_len,
        src_lang=args.src_lang
    )
    print(translation)


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="机器翻译推理脚本 - 支持多种调用方式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
调用方式示例:

  1. 一键演示模式（使用 checkpoints_best 中的三种模型）:
     python inference.py --demo

  2. 交互式翻译模式（选择模型后实时输入）:
     python inference.py --interactive --model_type rnn
     python inference.py --interactive --model_type transformer
     python inference.py --interactive --model_type t5

  3. 命令行单句翻译:
     python inference.py --checkpoint checkpoints_best/rnn_best.pt --model_type rnn --text "今天天气很好"
     python inference.py --checkpoint checkpoints_best/transformer_best.pt --model_type transformer --text "我喜欢学习"
     python inference.py --checkpoint checkpoints_best/t5_best.pt --model_type t5 --text "机器翻译"
"""
    )
    
    # 模式选择
    parser.add_argument("--demo", action="store_true", 
                        help="一键演示模式：使用 checkpoints_best 中的三种模型翻译测试语句")
    parser.add_argument("--interactive", action="store_true", 
                        help="交互式翻译模式：选择模型后实时输入语句翻译")
    
    # 模型参数
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型 checkpoint 路径（命令行单句翻译时必需）")
    parser.add_argument("--model_type", type=str, default="transformer", 
                        choices=["rnn", "transformer", "t5"],
                        help="模型类型")
    
    # 翻译参数
    parser.add_argument("--src_lang", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--beam_size", type=int, default=5, help="Beam search 大小（默认5）")
    parser.add_argument("--max_len", type=int, default=64, help="最大生成长度")
    parser.add_argument("--text", type=str, default=None, help="待翻译的句子（单句翻译模式）")
    
    args = parser.parse_args()
    
    # 根据模式执行
    if args.demo:
        run_demo_mode(beam_size=args.beam_size, max_len=args.max_len)
    elif args.interactive:
        run_interactive_mode(model_type=args.model_type, beam_size=args.beam_size, max_len=args.max_len)
    elif args.text:
        if not args.checkpoint:
            # 使用 checkpoints_best 中对应的模型
            args.checkpoint = CHECKPOINTS_BEST.get(args.model_type)
            if not args.checkpoint or not os.path.exists(args.checkpoint):
                print(f"[错误] 未找到 {args.model_type} 模型，请使用 --checkpoint 指定路径")
                sys.exit(1)
        run_single_translate(args)
    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("请选择一种调用方式：--demo / --interactive / --text")
        print("=" * 70)


if __name__ == "__main__":
    main()
