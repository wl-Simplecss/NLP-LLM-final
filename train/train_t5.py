"""
基于预训练T5模型的微调脚本
"""
import argparse
import os
import sys
from typing import Tuple

# 将父目录加入路径，以便导入 data_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
# 修改导入：引入 AutoTokenizer 和 AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from data_utils import load_jsonl


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
        
        # T5使用prefix "translate Chinese to English: "
        input_text = f"translate Chinese to English: {src_text}"
        
        # 编码输入 (Encoder Inputs)
        encoder_inputs = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 编码输出 (Decoder Inputs / Labels)
        # T5 的 tokenizer 通常不需要特殊的 as_target_tokenizer 上下文（除非涉及特殊语言对配置），
        # 但直接编码通常是安全的。
        with self.tokenizer.as_target_tokenizer():
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
        }


def build_dataloaders(
    data_dir: str,
    tokenizer,
    batch_size: int,
    max_len: int,
    use_small: bool,
) -> Tuple[DataLoader, DataLoader]:
    train_file = "train_10k.jsonl" if use_small else "train_100k.jsonl"
    train_path = os.path.join(data_dir, train_file)
    valid_path = os.path.join(data_dir, "valid.jsonl")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练文件不存在: {train_path}")
    
    train_data = load_jsonl(train_path)
    valid_data = load_jsonl(valid_path)
    
    train_ds = T5TranslationDataset(train_data, tokenizer, max_len)
    valid_ds = T5TranslationDataset(valid_data, tokenizer, max_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="train")
    
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # T5的labels需要将padding token设为-100（忽略计算loss）
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
    for batch in tqdm(loader, desc="valid"):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_raw/AP0004_Midterm&Final_translation_dataset_zh_en")
    parser.add_argument("--use_small", action="store_true", help="use 10k training set instead of 100k")
    parser.add_argument("--model_name", type=str, default="google-t5/t5-base", help="模型名称")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="checkpoints/checkpoints_t5")
    args = parser.parse_args()
    
    # -----------------------------------------------------------
    # 1. 强制设置镜像环境变量，确保从镜像站点加载
    # -----------------------------------------------------------
    MIRROR_ENDPOINT = "https://hf-mirror.com"
    
    # 设置所有相关的环境变量
    os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = MIRROR_ENDPOINT
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
    os.environ.setdefault("REQUESTS_TIMEOUT", "300")
    
    # 禁用本地缓存，强制从镜像站点加载到内存
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    print(f"当前使用的 HuggingFace 镜像端点: {MIRROR_ENDPOINT}")
    print(f"模型名称: {args.model_name}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------------------------------------
    # 2. 从镜像站点加载模型（保存到本地目录，避免重复下载）
    # -----------------------------------------------------------
    # 本地模型缓存目录
    MODEL_CACHE_DIR = "/data/250010066/LLM_course/final_program/t5-model"
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    # 为每个模型创建独立的子目录（使用模型名称的简化版本）
    model_name_safe = args.model_name.replace("/", "_").replace("-", "_")
    model_local_dir = os.path.join(MODEL_CACHE_DIR, model_name_safe)
    
    try:
        print(f"正在加载模型: {args.model_name} ...")
        print(f"本地缓存目录: {model_local_dir}")
        
        # 检查本地是否已有模型文件
        config_file = os.path.join(model_local_dir, "config.json")
        has_local_model = os.path.exists(config_file)
        
        if has_local_model:
            print(f"✓ 发现本地模型缓存，直接从本地加载...")
            model_cache_dir = model_local_dir
            use_local_only = True
        else:
            print(f"步骤1: 从镜像站点 {MIRROR_ENDPOINT} 下载模型文件到本地目录...")
            print(f"      保存位置: {model_local_dir}")
            
            # 使用 snapshot_download 从镜像站点下载所有必需文件
            # endpoint 参数明确指定使用镜像站点，而不是默认的 huggingface.co
            try:
                model_cache_dir = snapshot_download(
                    repo_id=args.model_name,
                    cache_dir=model_local_dir,
                    endpoint=MIRROR_ENDPOINT,  # 关键：指定镜像站点
                    local_files_only=False,
                    resume_download=True,
                    # 1. 既然是 T5，我们需要 spiece.model 和 json 配置文件
                    # 2. 我们只需要 pytorch 权重 (*.bin) 或 safetensors
                    allow_patterns=["*.json", "*.bin", "*.model", "*.safetensors"],
                    ignore_patterns=["*.md", "*.txt", "*.h5", "*.ot", "*.msgpack"]  # 忽略 TF/Rust/Flax 权重和文档
                )
                print(f"✓ 模型文件已成功下载并保存到: {model_cache_dir}")
            except Exception as download_error:
                print(f"❌ 从镜像站点下载失败: {download_error}")
                print("尝试重新下载（强制更新）...")
                # 重试：强制重新下载
                model_cache_dir = snapshot_download(
                    repo_id=args.model_name,
                    cache_dir=model_local_dir,
                    endpoint=MIRROR_ENDPOINT,
                    local_files_only=False,
                    force_download=True,  # 强制重新下载
                    # 同样使用 allow_patterns 确保下载必要的文件
                    allow_patterns=["*.json", "*.bin", "*.model", "*.safetensors"],
                    ignore_patterns=["*.md", "*.txt", "*.h5", "*.ot", "*.msgpack"]  # 忽略 TF/Rust/Flax 权重和文档
                )
                print(f"✓ 模型文件已重新下载并保存到: {model_cache_dir}")
            
            use_local_only = True  # 下载完成后，后续都从本地加载
        
        print(f"步骤2: 正在从本地目录加载模型到内存...")
        
        # 从本地目录加载模型和分词器到内存
        # local_files_only=True 确保只从本地目录读取，不再访问网络
        # 模型权重会被完整加载到内存中
        tokenizer = AutoTokenizer.from_pretrained(
            model_cache_dir,
            local_files_only=use_local_only
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_cache_dir,
            local_files_only=use_local_only
        )
        
        print("✓ 模型和分词器已成功加载到内存！")
        print(f"   模型权重已保存到: {model_local_dir}")
        print(f"   下次运行将直接使用本地缓存，无需重新下载")
        
    except Exception as e:
        print("\n[错误] 模型加载失败。")
        print("请检查：")
        print("1. 是否已安装 sentencepiece (T5需要): pip install sentencepiece")
        print("2. 是否已安装 huggingface_hub: pip install huggingface_hub")
        print("3. 网络是否能连接到 hf-mirror.com")
        print(f"原始错误信息: {e}")
        import traceback
        traceback.print_exc()
        return

    model.to(device)
    
    # 构建数据加载器
    print("正在构建数据加载器...")
    train_loader, valid_loader = build_dataloaders(
        args.data_dir, tokenizer, args.batch_size, args.max_len, args.use_small
    )
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    
    # 训练循环
    print("开始训练...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        valid_loss = eval_epoch(model, valid_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")
        
        # 保存checkpoint (仅保存 state_dict 以节省空间)
        ckpt_path = os.path.join(args.save_dir, f"t5_epoch{epoch}.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")
    
    # 保存最终模型 (HuggingFace 格式，方便后续直接加载)
    final_model_path = os.path.join(args.save_dir, "t5_final")
    print(f"正在保存最终模型至 {final_model_path} ...")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("训练完成。")


if __name__ == "__main__":
    main()
