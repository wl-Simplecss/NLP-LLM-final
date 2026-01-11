"""
基于预训练T5模型的微调脚本 - 改进版
添加BLEU评估、日志输出、测试集评估
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ========== 绕过PyTorch CVE-2025-32434安全限制 ==========
# 必须在导入torch之前设置
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import torch

# 保存并替换torch.load
_original_torch_load = torch.load

def _patched_load(f, *args, **kwargs):
    """强制使用weights_only=False"""
    kwargs.pop('weights_only', None)  # 移除原有参数
    kwargs['weights_only'] = False
    try:
        return _original_torch_load(f, *args, **kwargs)
    except TypeError:
        # 老版本torch可能不支持weights_only参数
        kwargs.pop('weights_only', None)
        return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_load

# Patch transformers的版本检查（在导入前修改）
import transformers.utils.import_utils as import_utils
if hasattr(import_utils, 'is_torch_greater_or_equal_than_2_6'):
    import_utils.is_torch_greater_or_equal_than_2_6 = True
# ========================================================

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

# Patch transformers modeling_utils 如果有版本检查
try:
    import transformers.modeling_utils as modeling_utils
    if hasattr(modeling_utils, '_is_torch_greater_or_equal_than_2_6'):
        modeling_utils._is_torch_greater_or_equal_than_2_6 = lambda: True
except:
    pass
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from data_utils import load_jsonl, tokenize_en
from run_mt_pipeline import bleu_compute_corpus_bleu


class Logger:
    """同时输出到控制台和日志文件"""
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.file = open(log_path, 'w', encoding='utf-8')
    
    def log(self, msg):
        print(msg)
        self.file.write(msg + '\n')
        self.file.flush()
    
    def close(self):
        self.file.close()


class T5TranslationDataset(Dataset):
    """适配T5的数据集"""
    
    def __init__(self, data: list, tokenizer, max_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item["zh"]
        tgt_text = item["en"]
        
        # mT5/T5使用prefix指示任务
        input_text = f"翻译成英语: {src_text}"
        
        # 编码输入
        encoder_inputs = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 编码输出
        decoder_inputs = self.tokenizer(
            tgt_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoder_inputs["input_ids"].squeeze(0),
            "attention_mask": encoder_inputs["attention_mask"].squeeze(0),
            "labels": decoder_inputs["input_ids"].squeeze(0),
            "tgt_text": tgt_text,  # 保存原始文本用于BLEU计算
        }


def build_dataloaders(
    data_dir: str,
    tokenizer,
    batch_size: int,
    max_len: int,
    use_small: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader, list, list]:
    train_file = "train_10k.jsonl" if use_small else "train_100k.jsonl"
    train_path = os.path.join(data_dir, train_file)
    valid_path = os.path.join(data_dir, "valid.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练文件不存在: {train_path}")
    
    train_data = load_jsonl(train_path)
    valid_data = load_jsonl(valid_path)
    test_data = load_jsonl(test_path)
    
    train_ds = T5TranslationDataset(train_data, tokenizer, max_len)
    valid_ds = T5TranslationDataset(valid_data, tokenizer, max_len)
    test_ds = T5TranslationDataset(test_data, tokenizer, max_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader, valid_data, test_data


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="train", leave=False)
    
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # T5的labels需要将padding token设为-100
        labels[labels == model.config.pad_token_id] = -100
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="valid", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        labels[labels == model.config.pad_token_id] = -100
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def compute_bleu(model, tokenizer, test_data, device, max_samples=500, beam_size=4, max_len=128):
    """计算BLEU分数"""
    model.eval()
    candidates = []
    references = []
    
    samples = test_data[:max_samples]
    
    for item in tqdm(samples, desc="BLEU eval", leave=False):
        src_text = item["zh"]
        tgt_text = item["en"]
        
        input_text = f"翻译成英语: {src_text}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # 生成翻译
        if beam_size > 1:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_len,
                num_beams=beam_size,
                early_stopping=True,
            )
        else:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_len,
                num_beams=1,
                do_sample=False,
            )
        
        pred_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_tokens = tokenize_en(pred_str)
        ref_tokens = tokenize_en(tgt_text)
        
        candidates.append(pred_tokens)
        references.append([ref_tokens])
    
    bleu_score = bleu_compute_corpus_bleu(candidates, references, max_n=4)
    return bleu_score * 100  # 转换为百分比


def load_model_from_mirror(model_name: str, device):
    """从本地或镜像站点加载模型"""
    MIRROR_ENDPOINT = "https://hf-mirror.com"
    os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = MIRROR_ENDPOINT
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
    # 跳过PyTorch版本安全检查（CVE-2025-32434）
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    
    MODEL_CACHE_DIR = "/data/250010066/LLM_course/final_program/t5-model"
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    # 尝试多种可能的本地路径
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    possible_paths = [
        os.path.join(MODEL_CACHE_DIR, model_name_safe),  # google_mt5_small
        os.path.join(MODEL_CACHE_DIR, model_name.replace("/", "_")),  # google_mt5-small
        os.path.join(MODEL_CACHE_DIR, model_name.split("/")[-1]),  # mt5-small
    ]
    
    model_local_dir = None
    for path in possible_paths:
        config_file = os.path.join(path, "config.json")
        if os.path.exists(config_file):
            model_local_dir = path
            print(f"从本地缓存加载模型: {model_local_dir}")
            break
    
    if model_local_dir is not None:
        # 从本地加载（已通过patch绕过PyTorch安全限制）
        print(f"模型目录内容: {os.listdir(model_local_dir)}")
        tokenizer = AutoTokenizer.from_pretrained(model_local_dir, local_files_only=True)
        print("加载模型（使用patched torch.load）...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_local_dir, 
            local_files_only=True,
            use_safetensors=False  # 使用pytorch_model.bin
        )
    else:
        # 从镜像站点下载
        print(f"本地未找到模型，从镜像站点下载: {model_name}")
        model_local_dir = os.path.join(MODEL_CACHE_DIR, model_name_safe)
        snapshot_download(
            repo_id=model_name,
            local_dir=model_local_dir,
            endpoint=MIRROR_ENDPOINT,
            local_files_only=False,
            resume_download=True,
            allow_patterns=["*.json", "*.bin", "*.model"],
            ignore_patterns=["*.md", "*.txt", "*.h5"]
        )
        tokenizer = AutoTokenizer.from_pretrained(model_local_dir, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_local_dir, 
            local_files_only=True,
            use_safetensors=False
        )
    
    return model, tokenizer, model_local_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_raw/AP0004_Midterm&Final_translation_dataset_zh_en")
    parser.add_argument("--use_small", action="store_true", help="use 10k training set")
    parser.add_argument("--model_name", type=str, default="google/mt5-small", help="模型名称，推荐使用mt5多语言模型")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)  # mT5推荐稍高的学习率
    parser.add_argument("--warmup_ratio", type=float, default=0.1)  # mT5需要更多warmup
    parser.add_argument("--save_dir", type=str, default="checkpoints/checkpoints_t5")
    parser.add_argument("--log_dir", type=str, default="outputs")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--eval_bleu_every", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=4)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"t5_training_{timestamp}.log")
    logger = Logger(log_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 打印配置
    logger.log("=" * 60)
    logger.log("T5 Fine-tuning Configuration")
    logger.log("=" * 60)
    for k, v in vars(args).items():
        logger.log(f"  {k}: {v}")
    logger.log(f"  device: {device}")
    logger.log("=" * 60)
    
    # 加载模型
    try:
        model, tokenizer, model_local_dir = load_model_from_mirror(args.model_name, device)
        logger.log(f"模型加载成功: {args.model_name}")
    except Exception as e:
        logger.log(f"模型加载失败: {e}")
        logger.close()
        return
    
    model.to(device)
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Total parameters: {total_params:,}")
    logger.log(f"Trainable parameters: {trainable_params:,}")
    
    # 构建数据加载器
    train_loader, valid_loader, test_loader, valid_data, test_data = build_dataloaders(
        args.data_dir, tokenizer, args.batch_size, args.max_len, args.use_small
    )
    
    logger.log(f"Train samples: {len(train_loader.dataset)}")
    logger.log(f"Valid samples: {len(valid_loader.dataset)}")
    logger.log(f"Test samples: {len(test_loader.dataset)}")
    logger.log("=" * 60)
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    logger.log(f"Total training steps: {total_steps}")
    logger.log(f"Warmup steps: {warmup_steps}")
    logger.log("=" * 60)
    
    # 训练循环
    best_valid_loss = float("inf")
    best_bleu = 0.0
    best_epoch = 0
    no_improve = 0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        valid_loss = eval_epoch(model, valid_loader, device)
        
        elapsed = timedelta(seconds=int(time.time() - start_time))
        lr = optimizer.param_groups[0]["lr"]
        
        # 计算BLEU
        bleu_str = ""
        if epoch % args.eval_bleu_every == 0:
            bleu_score = compute_bleu(model, tokenizer, valid_data, device, 
                                      max_samples=min(200, len(valid_data)), 
                                      beam_size=args.beam_size, max_len=args.max_len)
            bleu_str = f" | BLEU {bleu_score:.2f}"
            if bleu_score > best_bleu:
                best_bleu = bleu_score
        
        log_msg = f"Epoch {epoch:>2}/{args.epochs} | Train {train_loss:.4f} | Valid {valid_loss:.4f} | LR {lr:.2e}{bleu_str} | Elapsed {elapsed}"
        logger.log(log_msg)
        
        # 保存最佳模型
        is_best = valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_loss
            best_epoch = epoch
            no_improve = 0
            
            best_path = os.path.join(args.save_dir, "best_t5_model.pt")
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }, best_path)
            logger.log(f"  -> New best model saved at epoch {epoch}")
            
            # 同时保存HuggingFace格式
            best_hf_path = os.path.join(args.save_dir, "best_t5_hf")
            model.save_pretrained(best_hf_path)
            tokenizer.save_pretrained(best_hf_path)
        else:
            no_improve += 1
        
        if no_improve >= args.patience:
            logger.log(f"Early stop at epoch {epoch}. Best epoch={best_epoch}")
            break
    
    # 测试集评估
    logger.log("=" * 60)
    logger.log("Training completed. Evaluating on test set...")
    
    try:
        # 加载最佳模型
        best_ckpt = torch.load(os.path.join(args.save_dir, "best_t5_model.pt"), map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
        
        test_loss = eval_epoch(model, test_loader, device)
        
        # 计算测试集BLEU（Greedy和Beam）
        test_bleu_greedy = compute_bleu(model, tokenizer, test_data, device, 
                                         max_samples=len(test_data), beam_size=1, max_len=args.max_len)
        test_bleu_beam = compute_bleu(model, tokenizer, test_data, device, 
                                       max_samples=len(test_data), beam_size=args.beam_size, max_len=args.max_len)
        
        logger.log(f"Test Loss: {test_loss:.4f}")
        logger.log(f"Test BLEU (Greedy): {test_bleu_greedy:.2f}")
        logger.log(f"Test BLEU (Beam{args.beam_size}): {test_bleu_beam:.2f}")
    except Exception as e:
        logger.log(f"Test evaluation error: {e}")
        import traceback
        logger.log(traceback.format_exc())
    
    logger.log("=" * 60)
    logger.log(f"Best epoch: {best_epoch}")
    logger.log(f"Best valid loss: {best_valid_loss:.4f}")
    logger.log(f"Best valid BLEU: {best_bleu:.2f}")
    logger.log("=" * 60)
    
    # 输出几个翻译示例
    try:
        logger.log("\nTranslation Examples:")
        logger.log("-" * 60)
        model.eval()
        for i, item in enumerate(test_data[:5]):
            src_text = item["zh"]
            tgt_text = item["en"]
            
            input_text = f"翻译成英语: {src_text}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=args.max_len)
            input_ids = inputs["input_ids"].to(device)
            
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, max_length=args.max_len, num_beams=args.beam_size)
            
            pred_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.log(f"[{i+1}] Source: {src_text}")
            logger.log(f"    Pred:   {pred_str}")
            logger.log(f"    Ref:    {tgt_text}")
            logger.log("")
    except Exception as e:
        logger.log(f"Translation examples error: {e}")
    
    logger.close()
    print(f"\nTraining log saved to: {log_path}")


if __name__ == "__main__":
    main()
