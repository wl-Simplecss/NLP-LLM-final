"""
总调用脚本：完成RNN和Transformer模型的训练、测试和评估
所有输出将保存到txt文件中
"""
import os
import re
import subprocess
import sys
from datetime import datetime
import torch


def ensure_torch_available():
    """预检查当前 Python 是否已安装 torch，缺失时给出明确指引并退出。"""
    try:
        import torch  # noqa: F401
    except Exception:
        print("\n[致命错误] 当前 Python 环境未安装 torch，无法继续训练/评估。")
        print("请先安装或切换到已安装 torch 的环境，然后重新运行：\n")
        print("  # 推荐使用已有环境 llm_final_pro")
        print("  source /data/250010066/conda/bin/activate llm_final_pro")
        print("  pip install torch==2.3.0 torchtext==0.18.0 -i https://pypi.tuna.tsinghua.edu.cn/simple\n")
        print("或在运行前指定已安装 torch 的解释器：")
        print("  PYTHON_EXEC=/data/250010066/conda/envs/llm_final_pro/bin/python python run_mt_pipeline.py\n")
        sys.exit(1)


class TeeOutput:
    """同时输出到控制台和文件的类"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
    
    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)
        self.file.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr


def run_command(cmd, output_file=None, description=""):
    """运行命令并捕获输出"""
    print(f"\n[{description}]")
    
    if output_file:
        output_file.write(f"\n[{description}]\n")
        output_file.flush()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        output = result.stdout
        error = result.stderr
        
        # 过滤掉不必要的输出（警告、进度条等）
        if output:
            # 过滤掉jieba警告、SentencePiece详细配置、tqdm进度条等
            filtered_output = []
            for line in output.split('\n'):
                line_stripped = line.strip()
                # 跳过空行、警告、详细配置、tqdm进度条
                if not line_stripped:
                    continue
                if 'pkg_resources is deprecated' in line:
                    continue
                if 'sentencepiece_trainer.cc' in line or 'trainer_spec' in line or 'normalizer_spec' in line:
                    continue
                if line_stripped.startswith('Evaluating:') and '%' in line:
                    continue
                # 保留训练epoch输出、BLEU分数、关键结果
                if 'Epoch' in line or 'BLEU' in line or 'Average' in line or \
                   '训练完成' in line or '完成' in line or '训练趋势' in line or \
                   'Early Stopping' in line or '最佳模型' in line:
                    filtered_output.append(line)
            
            clean_output = '\n'.join(filtered_output).strip()
            if clean_output:
                print(clean_output)
                if output_file:
                    output_file.write(clean_output + "\n")
                    output_file.flush()
        
        if error:
            # 只显示真正的错误，过滤警告
            error_lines = [line for line in error.split('\n') 
                          if 'pkg_resources is deprecated' not in line and line.strip()]
            if error_lines:
                error_msg = '\n'.join(error_lines)
                print(f"错误: {error_msg}", file=sys.stderr)
                if output_file:
                    output_file.write(f"错误: {error_msg}\n")
                    output_file.flush()
        
        if result.returncode != 0:
            print(f"命令执行失败，返回码: {result.returncode}")
            if output_file:
                output_file.write(f"命令执行失败，返回码: {result.returncode}\n")
                output_file.flush()
            return False
        
        return True
    except Exception as e:
        error_msg = f"执行命令时出错: {str(e)}"
        print(error_msg)
        if output_file:
            output_file.write(error_msg + "\n")
            output_file.flush()
        return False


def main():
    # 创建输出目录
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建checkpoint目录
    os.makedirs("checkpoints/checkpoints_rnn", exist_ok=True)
    os.makedirs("checkpoints/checkpoints_transformer", exist_ok=True)
    
    # 生成输出文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(output_dir, f"training_log_{timestamp}.txt")
    
    # 打开输出文件
    output_file = open(output_file_path, 'w', encoding='utf-8')
    
    # 写入开始信息
    start_time = datetime.now()
    header = f"""
{'='*80}
中英机器翻译项目 - 完整训练和评估流程
开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

项目目录结构:
- data_raw/AP0004_Midterm&Final_translation_dataset_zh_en/ (数据集)
- checkpoints/checkpoints_rnn/ (RNN模型checkpoint)
- checkpoints/checkpoints_transformer/ (Transformer模型checkpoint)
- outputs/ (输出日志)

{'='*80}

"""
    print(header)
    output_file.write(header)
    output_file.flush()
    
    # 数据路径（相对路径）
    data_dir = "data_raw/AP0004_Midterm&Final_translation_dataset_zh_en"
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        error_msg = f"错误: 数据目录不存在: {data_dir}\n请确保数据集已解压到正确位置。"
        print(error_msg)
        output_file.write(error_msg + "\n")
        output_file.close()
        return
    
    # 预检查 torch 是否可用（避免训练/评估时再报错）
    ensure_torch_available()
    
    # ==================== 第零部分：训练SentencePiece分词模型 ====================
    print("\n" + "="*80)
    print("第零部分：训练SentencePiece分词模型")
    print("="*80 + "\n")
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("第零部分：训练SentencePiece分词模型\n")
    output_file.write("="*80 + "\n\n")
    output_file.flush()
    
    # 检查是否已存在 SentencePiece 模型（在 spm 目录中）
    spm_dir = "spm"
    spm_zh_exists = os.path.exists(os.path.join(spm_dir, "spm_zh.model"))
    spm_en_exists = os.path.exists(os.path.join(spm_dir, "spm_en.model"))
    
    if spm_zh_exists and spm_en_exists:
        print("SentencePiece 模型已存在，跳过训练步骤")
        output_file.write("SentencePiece 模型已存在，跳过训练步骤\n")
        output_file.flush()
    else:
        print("开始训练 SentencePiece 模型...")
        output_file.write("开始训练 SentencePiece 模型...\n")
        output_file.flush()
        
        spm_train_cmd = [
            sys.executable, "train/train_spm.py",
            "--data_dir", data_dir,
            "--vocab_size", "8000"  # 8k-16k 对100k数据比较合适
        ]
        
        success = run_command(
            spm_train_cmd,
            output_file,
            "训练SentencePiece分词模型（BPE）"
        )
        
        if not success:
            print("SentencePiece模型训练失败，将使用传统分词方式")
            output_file.write("SentencePiece模型训练失败，将使用传统分词方式\n")
            output_file.flush()
    
    # ==================== 第一部分：训练RNN模型 ====================
    print("\n" + "="*80)
    print("第一部分：训练RNN模型")
    print("="*80 + "\n")
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("第一部分：训练RNN模型\n")
    output_file.write("="*80 + "\n\n")
    output_file.flush()
    
    # RNN模型训练命令优化（使用SentencePiece，Scheduled Sampling，解决过拟合）
    rnn_train_cmd = [
        sys.executable, "train/train_rnn.py",
        "--data_dir", data_dir,
        "--batch_size", "128",
        "--max_len", "100",
        "--embed_dim", "256",      # 更稳的默认值
        "--hidden_size", "512",
        "--num_layers", "2",
        "--dropout", "0.4",        # 优化：增加Dropout到0.4，解决过拟合问题
        "--attn_type", "additive",  # 优化：使用真正的Additive Attention（已修复代码）
        "--use_spm",               # 使用SentencePiece分词
        "--epochs", "40",
        "--lr", "0.0008",          # 8e-4
        "--tf_start", "1.0",       # Scheduled Sampling起始值
        "--tf_end", "0.6",         # Scheduled Sampling结束值
        "--tf_decay_epochs", "25", # 衰减周期
        "--save_dir", "checkpoints/checkpoints_rnn"
    ]
    
    success = run_command(
        rnn_train_cmd,
        output_file,
        "训练RNN模型（点积注意力，Teacher Forcing）"
    )
    
    if not success:
        print("RNN模型训练失败，继续执行后续步骤...")
        output_file.write("RNN模型训练失败，继续执行后续步骤...\n")
        output_file.flush()
    
    # ==================== 第一部分扩展：RNN模型对比实验 ====================
    print("\n" + "="*80)
    print("第一部分扩展：RNN模型对比实验")
    print("="*80 + "\n")
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("第一部分扩展：RNN模型对比实验\n")
    output_file.write("="*80 + "\n\n")
    output_file.flush()
    
    # 1.1 不同注意力机制对比
    print("\n[实验1.1] 对比不同注意力机制（General vs Additive）")
    output_file.write("\n[实验1.1] 对比不同注意力机制（General vs Additive）\n")
    output_file.flush()
    
    for attn_type in ["general", "additive"]:
        rnn_attn_cmd = [
            sys.executable, "train/train_rnn.py",
            "--data_dir", data_dir,
            "--batch_size", "64",  # 减小batch size以加快实验
            "--max_len", "100",
            "--embed_dim", "256",
            "--hidden_size", "512",
            "--num_layers", "2",
            "--dropout", "0.3",
            "--attn_type", attn_type,
            "--use_spm",
            "--epochs", "20",  # 减少epoch以加快实验
            "--lr", "0.0008",
            "--tf_start", "1.0",
            "--tf_end", "0.6",
            "--tf_decay_epochs", "15",
            "--save_dir", f"checkpoints/checkpoints_rnn_attn_{attn_type}"
        ]
        
        run_command(
            rnn_attn_cmd,
            output_file,
            f"训练RNN模型（{attn_type}注意力）"
        )
    
    # 1.2 Teacher Forcing vs Free Running对比
    print("\n[实验1.2] 对比Teacher Forcing vs Free Running")
    output_file.write("\n[实验1.2] 对比Teacher Forcing vs Free Running\n")
    output_file.flush()
    
    for tf_config in [
        ("teacher_forcing", "1.0", "1.0"),  # 纯Teacher Forcing
        ("free_running", "0.0", "0.0"),     # 纯Free Running
        ("scheduled", "1.0", "0.5")         # Scheduled Sampling
    ]:
        name, tf_start, tf_end = tf_config
        rnn_tf_cmd = [
            sys.executable, "train/train_rnn.py",
            "--data_dir", data_dir,
            "--batch_size", "64",
            "--max_len", "100",
            "--embed_dim", "256",
            "--hidden_size", "512",
            "--num_layers", "2",
            "--dropout", "0.3",
            "--attn_type", "additive",
            "--use_spm",
            "--epochs", "20",
            "--lr", "0.0008",
            "--tf_start", tf_start,
            "--tf_end", tf_end,
            "--tf_decay_epochs", "15",
            "--save_dir", f"checkpoints/checkpoints_rnn_tf_{name}"
        ]
        
        run_command(
            rnn_tf_cmd,
            output_file,
            f"训练RNN模型（{name}）"
        )
    
    # ==================== 第二部分：训练Transformer模型 ====================
    print("\n" + "="*80)
    print("第二部分：训练Transformer模型")
    print("="*80 + "\n")
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("第二部分：训练Transformer模型\n")
    output_file.write("="*80 + "\n\n")
    output_file.flush()
    
    # Transformer模型训练命令优化（使用SentencePiece，减少层数到3，增加Dropout到0.3）
    transformer_train_cmd = [
        sys.executable, "train/train_transformer.py",
        "--data_dir", data_dir,
        "--batch_size", "128",
        "--max_len", "100",
        "--d_model", "512",
        "--num_heads", "8",
        "--num_encoder_layers", "3",  # 减少层数到3，适合100k数据量
        "--num_decoder_layers", "3",  # 减少层数到3，适合100k数据量
        "--dim_ff", "2048",
        "--dropout", "0.3",          # 增加Dropout到0.3，防止过拟合
        "--pos_encoding", "sinusoidal",
        "--norm_type", "layernorm",
        "--use_spm",                 # 使用SentencePiece分词
        "--epochs", "80",
        "--lr", "2.0",               # 提高 Noam factor
        "--warmup_steps", "4000",     # 减少 warmup
        "--save_dir", "checkpoints/checkpoints_transformer"
    ]
    
    success = run_command(
        transformer_train_cmd,
        output_file,
        "训练Transformer模型（sinusoidal位置编码，LayerNorm）"
    )
    
    if not success:
        print("Transformer模型训练失败，继续执行后续步骤...")
        output_file.write("Transformer模型训练失败，继续执行后续步骤...\n")
        output_file.flush()
    
    # ==================== 第二部分扩展：Transformer模型对比实验 ====================
    print("\n" + "="*80)
    print("第二部分扩展：Transformer模型对比实验")
    print("="*80 + "\n")
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("第二部分扩展：Transformer模型对比实验\n")
    output_file.write("="*80 + "\n\n")
    output_file.flush()
    
    # 2.1 不同位置编码对比
    print("\n[实验2.1] 对比不同位置编码（Sinusoidal vs Learned vs Relative）")
    output_file.write("\n[实验2.1] 对比不同位置编码（Sinusoidal vs Learned vs Relative）\n")
    output_file.flush()
    
    for pos_enc in ["sinusoidal", "learned", "relative"]:
        transformer_pos_cmd = [
            sys.executable, "train/train_transformer.py",
            "--data_dir", data_dir,
            "--batch_size", "64",
            "--max_len", "100",
            "--d_model", "512",
            "--num_heads", "8",
            "--num_encoder_layers", "3",
            "--num_decoder_layers", "3",
            "--dim_ff", "2048",
            "--dropout", "0.3",
            "--pos_encoding", pos_enc,
            "--norm_type", "layernorm",
            "--use_spm",
            "--epochs", "30",  # 减少epoch以加快实验
            "--lr", "2.0",
            "--warmup_steps", "4000",
            "--save_dir", f"checkpoints/checkpoints_transformer_pos_{pos_enc}"
        ]
        
        run_command(
            transformer_pos_cmd,
            output_file,
            f"训练Transformer模型（{pos_enc}位置编码）"
        )
    
    # 2.2 不同归一化方法对比
    print("\n[实验2.2] 对比不同归一化方法（LayerNorm vs RMSNorm）")
    output_file.write("\n[实验2.2] 对比不同归一化方法（LayerNorm vs RMSNorm）\n")
    output_file.flush()
    
    for norm_type in ["layernorm", "rmsnorm"]:
        transformer_norm_cmd = [
            sys.executable, "train/train_transformer.py",
            "--data_dir", data_dir,
            "--batch_size", "64",
            "--max_len", "100",
            "--d_model", "512",
            "--num_heads", "8",
            "--num_encoder_layers", "3",
            "--num_decoder_layers", "3",
            "--dim_ff", "2048",
            "--dropout", "0.3",
            "--pos_encoding", "sinusoidal",
            "--norm_type", norm_type,
            "--use_spm",
            "--epochs", "30",
            "--lr", "2.0",
            "--warmup_steps", "4000",
            "--save_dir", f"checkpoints/checkpoints_transformer_norm_{norm_type}"
        ]
        
        run_command(
            transformer_norm_cmd,
            output_file,
            f"训练Transformer模型（{norm_type}归一化）"
        )
    
    # 2.3 超参数敏感性实验
    print("\n[实验2.3] 超参数敏感性实验（Batch Size, Learning Rate, Model Scale）")
    output_file.write("\n[实验2.3] 超参数敏感性实验（Batch Size, Learning Rate, Model Scale）\n")
    output_file.flush()
    
    # 2.3.1 Batch Size对比
    print("\n[实验2.3.1] 对比不同Batch Size")
    output_file.write("\n[实验2.3.1] 对比不同Batch Size\n")
    output_file.flush()
    
    for batch_size in ["32", "64", "128"]:
        transformer_bs_cmd = [
            sys.executable, "train/train_transformer.py",
            "--data_dir", data_dir,
            "--batch_size", batch_size,
            "--max_len", "100",
            "--d_model", "512",
            "--num_heads", "8",
            "--num_encoder_layers", "3",
            "--num_decoder_layers", "3",
            "--dim_ff", "2048",
            "--dropout", "0.3",
            "--pos_encoding", "sinusoidal",
            "--norm_type", "layernorm",
            "--use_spm",
            "--epochs", "20",
            "--lr", "2.0",
            "--warmup_steps", "4000",
            "--save_dir", f"checkpoints/checkpoints_transformer_bs_{batch_size}"
        ]
        
        run_command(
            transformer_bs_cmd,
            output_file,
            f"训练Transformer模型（batch_size={batch_size}）"
        )
    
    # 2.3.2 Learning Rate对比
    print("\n[实验2.3.2] 对比不同Learning Rate")
    output_file.write("\n[实验2.3.2] 对比不同Learning Rate\n")
    output_file.flush()
    
    for lr in ["1.0", "2.0", "3.0"]:
        transformer_lr_cmd = [
            sys.executable, "train/train_transformer.py",
            "--data_dir", data_dir,
            "--batch_size", "64",
            "--max_len", "100",
            "--d_model", "512",
            "--num_heads", "8",
            "--num_encoder_layers", "3",
            "--num_decoder_layers", "3",
            "--dim_ff", "2048",
            "--dropout", "0.3",
            "--pos_encoding", "sinusoidal",
            "--norm_type", "layernorm",
            "--use_spm",
            "--epochs", "20",
            "--lr", lr,
            "--warmup_steps", "4000",
            "--save_dir", f"checkpoints/checkpoints_transformer_lr_{lr}"
        ]
        
        run_command(
            transformer_lr_cmd,
            output_file,
            f"训练Transformer模型（lr={lr}）"
        )
    
    # 2.3.3 Model Scale对比
    print("\n[实验2.3.3] 对比不同Model Scale")
    output_file.write("\n[实验2.3.3] 对比不同Model Scale\n")
    output_file.flush()
    
    for scale in [
        ("small", "256", "4", "2", "2", "1024"),
        ("medium", "512", "8", "3", "3", "2048"),
        ("large", "768", "12", "4", "4", "3072")
    ]:
        name, d_model, num_heads, enc_layers, dec_layers, dim_ff = scale
        transformer_scale_cmd = [
            sys.executable, "train/train_transformer.py",
            "--data_dir", data_dir,
            "--batch_size", "64",
            "--max_len", "100",
            "--d_model", d_model,
            "--num_heads", num_heads,
            "--num_encoder_layers", enc_layers,
            "--num_decoder_layers", dec_layers,
            "--dim_ff", dim_ff,
            "--dropout", "0.3",
            "--pos_encoding", "sinusoidal",
            "--norm_type", "layernorm",
            "--use_spm",
            "--epochs", "20",
            "--lr", "2.0",
            "--warmup_steps", "4000",
            "--save_dir", f"checkpoints/checkpoints_transformer_scale_{name}"
        ]
        
        run_command(
            transformer_scale_cmd,
            output_file,
            f"训练Transformer模型（{name} scale）"
        )
    
    # 2.4 T5预训练模型微调对比
    print("\n[实验2.4] T5预训练模型微调对比")
    output_file.write("\n[实验2.4] T5预训练模型微调对比\n")
    output_file.flush()
    
    t5_train_cmd = [
        sys.executable, "train/train_t5.py",
        "--data_dir", data_dir,
        "--batch_size", "16",
        "--max_len", "128",
        "--epochs", "10",
        "--lr", "5e-5",
        "--warmup_steps", "500",
        "--model_name", "t5-small",
        "--save_dir", "checkpoints/checkpoints_t5"
    ]
    
    run_command(
        t5_train_cmd,
        output_file,
        "微调T5预训练模型（t5-small）"
    )
    
    # ==================== 第三部分：模型测试（推理） ====================
    print("\n" + "="*80)
    print("第三部分：模型测试（推理）")
    print("="*80 + "\n")
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("第三部分：模型测试（推理）\n")
    output_file.write("="*80 + "\n\n")
    output_file.flush()
    
    # 测试句子（基于数据集特点：新闻/政治/经济类正式文本）
    # 注意：使用数据集中的词汇和风格，但确保不与测试集完全一致
    test_sentences = [
        # A. 短句（15-30字符）- 政治/经济话题
        "上海—15年来，中国一直是全球增长的关键引擎。",
        "危机爆发至今已是十年，解决长期性经济弱点的必要性依然存在。",
        "深度中期预算改革是应对国会一再不作为所造成的问题之所必须。",
        # B. 中句（30-60字符）- 复杂句式/专有名词
        "尽管美国经济目前表现出色，但资产价格水平过高严重威胁着稳定。",
        "作为总统代表，克里提出了一揽子解决方案，旨在打破当前南北之间的僵局。",
        "与此同时，欧盟成员国一直在削减防务开支。",
        "这些收益足以满足当前世界基础设施需求的一半。",
        # C. 长句（60-100字符）- 复合句/多从句
        "这个两步走的过程与2008年的情况有点相似，当时国会面对的是另一种危险状况。",
        "上个月，在两个欧盟大国意大利和法国上交其预算计划后，谴责之声不可避免。",
        "尽管有着深厚的科研传统，这一数字在中欧则更为严峻。",
        # D. 包含专有名词和正式表达
        "欧洲必须致力于确保其在新世界秩序中的地位。",
        "用天然气代替煤确能降低碳排放，尽管从长期看天然气本身亦非可持续能源。",
        # E. 基础测试（简单短句）
        "今天天气很好",
        "这是一个测试句子"
    ]
    
    # 自动选择验证集 loss 最低的 checkpoint（而不是最新的）
    def find_best_checkpoint(checkpoint_dir, model_type):
        """查找验证集 loss 最低的 checkpoint"""
        if not os.path.exists(checkpoint_dir):
            return None
        
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if not checkpoint_files:
            return None
        
        best_checkpoint = None
        best_valid_loss = float('inf')
        
        for ckpt_file in checkpoint_files:
            ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                valid_loss = ckpt.get("valid_loss", float('inf'))
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_checkpoint = ckpt_path
            except Exception as e:
                print(f"警告: 无法加载 {ckpt_path}: {e}")
                continue
        
        if best_checkpoint:
            print(f"找到最佳{model_type} checkpoint: {best_checkpoint} (验证损失: {best_valid_loss:.4f})")
            output_file.write(f"找到最佳{model_type} checkpoint: {best_checkpoint} (验证损失: {best_valid_loss:.4f})\n")
            output_file.flush()
        
        return best_checkpoint
    
    rnn_checkpoint = find_best_checkpoint("checkpoints/checkpoints_rnn", "RNN")
    transformer_checkpoint = find_best_checkpoint("checkpoints/checkpoints_transformer", "Transformer")
    
    # RNN模型推理测试
    if rnn_checkpoint:
        print(f"\n使用RNN checkpoint: {rnn_checkpoint}")
        output_file.write(f"\n使用RNN checkpoint: {rnn_checkpoint}\n")
        output_file.write("\nRNN模型推理测试结果:\n")
        output_file.write("-" * 80 + "\n")
        output_file.flush()
        
        for test_sent in test_sentences:
            rnn_inference_cmd = [
                sys.executable, "inference.py",
                "--checkpoint", rnn_checkpoint,
                "--model_type", "rnn",
                "--text", test_sent,
                "--beam_size", "5",
                "--max_len", "80"
            ]
            
            print(f"\n输入: {test_sent}")
            output_file.write(f"\n输入: {test_sent}\n")
            output_file.flush()
            
            result = subprocess.run(
                rnn_inference_cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.stdout:
                print(f"RNN输出: {result.stdout.strip()}")
                output_file.write(f"RNN输出: {result.stdout.strip()}\n")
                output_file.flush()
    else:
        print("未找到RNN模型checkpoint，跳过RNN推理测试")
        output_file.write("未找到RNN模型checkpoint，跳过RNN推理测试\n")
        output_file.flush()
    
    # Transformer模型推理测试
    if transformer_checkpoint:
        print(f"\n使用Transformer checkpoint: {transformer_checkpoint}")
        output_file.write(f"\n使用Transformer checkpoint: {transformer_checkpoint}\n")
        output_file.write("\nTransformer模型推理测试结果:\n")
        output_file.write("-" * 80 + "\n")
        output_file.flush()
        
        for test_sent in test_sentences:
            transformer_inference_cmd = [
                sys.executable, "inference.py",
                "--checkpoint", transformer_checkpoint,
                "--model_type", "transformer",
                "--text", test_sent,
                "--beam_size", "5",
                "--max_len", "80"
            ]
            
            print(f"\n输入: {test_sent}")
            output_file.write(f"\n输入: {test_sent}\n")
            output_file.flush()
            
            result = subprocess.run(
                transformer_inference_cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.stdout:
                print(f"Transformer输出: {result.stdout.strip()}")
                output_file.write(f"Transformer输出: {result.stdout.strip()}\n")
                output_file.flush()
    else:
        print("未找到Transformer模型checkpoint，跳过Transformer推理测试")
        output_file.write("未找到Transformer模型checkpoint，跳过Transformer推理测试\n")
        output_file.flush()
    
    # ==================== 第四部分：模型评估（BLEU分数） ====================
    print("\n" + "="*80)
    print("第四部分：模型评估（BLEU分数）")
    print("="*80 + "\n")
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("第四部分：模型评估（BLEU分数）\n")
    output_file.write("="*80 + "\n\n")
    output_file.flush()
    
    test_file = os.path.join(data_dir, "test.jsonl")
    
    # RNN模型BLEU评估
    if rnn_checkpoint and os.path.exists(test_file):
        print("\n评估RNN模型BLEU分数...")
        output_file.write("\n评估RNN模型BLEU分数...\n")
        output_file.flush()
        
        rnn_eval_cmd = [
            sys.executable, "evaluate.py",
            "--checkpoint", rnn_checkpoint,
            "--model_type", "rnn",
            "--test_file", test_file,
            "--beam_size", "5",
            "--max_len", "80"
        ]
        
        success = run_command(
            rnn_eval_cmd,
            output_file,
            "评估RNN模型BLEU分数"
        )
    else:
        if not rnn_checkpoint:
            print("未找到RNN模型checkpoint，跳过BLEU评估")
            output_file.write("未找到RNN模型checkpoint，跳过BLEU评估\n")
        if not os.path.exists(test_file):
            print(f"测试文件不存在: {test_file}")
            output_file.write(f"测试文件不存在: {test_file}\n")
        output_file.flush()
    
    # Transformer模型BLEU评估
    if transformer_checkpoint and os.path.exists(test_file):
        print("\n评估Transformer模型BLEU分数...")
        output_file.write("\n评估Transformer模型BLEU分数...\n")
        output_file.flush()
        
        transformer_eval_cmd = [
            sys.executable, "evaluate.py",
            "--checkpoint", transformer_checkpoint,
            "--model_type", "transformer",
            "--test_file", test_file,
            "--beam_size", "5",
            "--max_len", "80"
        ]
        
        success = run_command(
            transformer_eval_cmd,
            output_file,
            "评估Transformer模型BLEU分数"
        )
    else:
        if not transformer_checkpoint:
            print("未找到Transformer模型checkpoint，跳过BLEU评估")
            output_file.write("未找到Transformer模型checkpoint，跳过BLEU评估\n")
        if not os.path.exists(test_file):
            print(f"测试文件不存在: {test_file}")
            output_file.write(f"测试文件不存在: {test_file}\n")
        output_file.flush()
    
    # ==================== 第五部分：对比分析 ====================
    print("\n" + "="*80)
    print("第五部分：模型对比分析")
    print("="*80 + "\n")
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("第五部分：模型对比分析\n")
    output_file.write("="*80 + "\n\n")
    output_file.flush()
    
    # 收集评估结果
    rnn_bleu = None
    transformer_bleu = None
    
    # 从评估输出中提取BLEU分数
    if rnn_checkpoint and os.path.exists(test_file):
        print("\n提取RNN模型BLEU分数...")
        output_file.write("\n提取RNN模型BLEU分数...\n")
        rnn_eval_cmd = [
            sys.executable, "evaluate.py",
            "--checkpoint", rnn_checkpoint,
            "--model_type", "rnn",
            "--test_file", test_file,
            "--beam_size", "5",
            "--max_len", "80"
        ]
        result = subprocess.run(
            rnn_eval_cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if result.stdout:
            # 尝试从输出中提取BLEU分数
            for line in result.stdout.split('\n'):
                if 'BLEU' in line or 'bleu' in line.lower():
                    try:
                        # 查找数字
                        numbers = re.findall(r'\d+\.\d+', line)
                        if numbers:
                            rnn_bleu = float(numbers[0])
                            break
                    except:
                        pass
    
    if transformer_checkpoint and os.path.exists(test_file):
        print("\n提取Transformer模型BLEU分数...")
        output_file.write("\n提取Transformer模型BLEU分数...\n")
        transformer_eval_cmd = [
            sys.executable, "evaluate.py",
            "--checkpoint", transformer_checkpoint,
            "--model_type", "transformer",
            "--test_file", test_file,
            "--beam_size", "5",
            "--max_len", "80"
        ]
        result = subprocess.run(
            transformer_eval_cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if result.stdout:
            # 尝试从输出中提取BLEU分数
            for line in result.stdout.split('\n'):
                if 'BLEU' in line or 'bleu' in line.lower():
                    try:
                        numbers = re.findall(r'\d+\.\d+', line)
                        if numbers:
                            transformer_bleu = float(numbers[0])
                            break
                    except:
                        pass
    
    def load_args_from_ckpt(path):
        try:
            ckpt = torch.load(path, map_location="cpu")
            return ckpt.get("args", {})
        except Exception:
            return {}

    rnn_args = load_args_from_ckpt(rnn_checkpoint) if rnn_checkpoint else {}
    transformer_args = load_args_from_ckpt(transformer_checkpoint) if transformer_checkpoint else {}

    rnn_dataset = "100k" if not rnn_args.get("use_small", False) else "10k"
    transformer_dataset = "100k" if not transformer_args.get("use_small", False) else "10k"

    # 生成对比分析报告
    comparison_report = f"""
{'='*80}
模型对比分析报告
{'='*80}

一、模型架构对比
────────────────────────────────────────────────────────────────────────────
1. RNN模型（Seq2Seq with Attention）
   - 编码器：双向GRU，{rnn_args.get('num_layers', '?')}层，隐藏层大小{rnn_args.get('hidden_size', '?')}
   - 解码器：单向GRU，{rnn_args.get('num_layers', '?')}层，隐藏层大小{rnn_args.get('hidden_size', '?')}
   - 注意力机制：{rnn_args.get('attn_type', 'dot')}
   - 嵌入维度：{rnn_args.get('embed_dim', '?')}
   - 训练策略：Teacher Forcing (ratio={rnn_args.get('teacher_forcing_ratio', '?')})
   - 参数量：相对较小，计算效率高

2. Transformer模型
   - 编码器层数：{transformer_args.get('num_encoder_layers', '?')}层
   - 解码器层数：{transformer_args.get('num_decoder_layers', '?')}层
   - 模型维度（d_model）：{transformer_args.get('d_model', '?')}
   - 注意力头数：{transformer_args.get('num_heads', '?')}
   - 前馈网络维度：{transformer_args.get('dim_ff', '?')}
   - 位置编码：{transformer_args.get('pos_encoding', 'sinusoidal')}
   - 归一化方法：{transformer_args.get('norm_type', 'layernorm')}
   - 参数量：相对较大，并行计算能力强

二、训练配置对比
────────────────────────────────────────────────────────────────────────────
RNN模型配置：
  - 数据集：{rnn_dataset}训练集
  - Batch Size：{rnn_args.get('batch_size', '?')}
  - 最大序列长度：{rnn_args.get('max_len', '?')}
  - 学习率：{rnn_args.get('lr', '?')}
  - 训练轮数：{rnn_args.get('epochs', '?')} epochs
  - Dropout：{rnn_args.get('dropout', '?')}

Transformer模型配置：
  - 数据集：{transformer_dataset}训练集
  - Batch Size：{transformer_args.get('batch_size', '?')}
  - 最大序列长度：{transformer_args.get('max_len', '?')}
  - 学习率：{transformer_args.get('lr', '?')}（Noam调度）
  - 训练轮数：{transformer_args.get('epochs', '?')} epochs
  - Dropout：{transformer_args.get('dropout', '?')}

三、性能评估对比
────────────────────────────────────────────────────────────────────────────
"""
    
    if rnn_bleu is not None:
        comparison_report += f"RNN模型BLEU-4分数：{rnn_bleu:.4f}\n"
    else:
        comparison_report += "RNN模型BLEU-4分数：未评估或评估失败\n"
    
    if transformer_bleu is not None:
        comparison_report += f"Transformer模型BLEU-4分数：{transformer_bleu:.4f}\n"
    else:
        comparison_report += "Transformer模型BLEU-4分数：未评估或评估失败\n"
    
    if rnn_bleu is not None and transformer_bleu is not None:
        diff = transformer_bleu - rnn_bleu
        diff_percent = (diff / rnn_bleu * 100) if rnn_bleu > 0 else 0
        comparison_report += f"\n性能差异：Transformer比RNN {'高' if diff > 0 else '低'} {abs(diff):.4f} ({abs(diff_percent):.2f}%)\n"
        
        if diff > 0:
            comparison_report += "\n结论：Transformer模型在翻译质量上优于RNN模型。\n"
        elif diff < 0:
            comparison_report += "\n结论：RNN模型在翻译质量上优于Transformer模型。\n"
        else:
            comparison_report += "\n结论：两个模型性能相当。\n"
    
    comparison_report += f"""
四、模型特点分析
────────────────────────────────────────────────────────────────────────────
RNN模型特点：
  ✓ 序列建模能力强，适合处理序列依赖
  ✓ 参数量相对较小，训练速度快
  ✓ 内存占用较小
  ✓ 使用Teacher Forcing训练，收敛稳定
  ✗ 无法并行计算，训练速度受序列长度限制
  ✗ 长距离依赖建模能力有限

Transformer模型特点：
  ✓ 完全并行计算，训练速度快
  ✓ 多头注意力机制，能捕捉不同层面的信息
  ✓ 位置编码明确，对位置信息敏感
  ✓ 长距离依赖建模能力强
  ✗ 参数量较大，需要更多内存
  ✗ 需要更多训练数据才能充分发挥优势

五、适用场景建议
────────────────────────────────────────────────────────────────────────────
RNN模型适用于：
  - 计算资源有限的环境
  - 需要快速迭代和实验的场景
  - 序列长度较短的任务
  - 对模型大小有严格要求的部署场景

Transformer模型适用于：
  - 有充足计算资源的场景
  - 需要处理长序列的任务
  - 对翻译质量要求较高的应用
  - 可以充分利用GPU并行计算的环境

六、改进建议
────────────────────────────────────────────────────────────────────────────
RNN模型改进方向：
  1. 尝试不同的注意力机制（general, additive）
  2. 调整Teacher Forcing比例，尝试Free Running
  3. 增加模型层数或隐藏层大小
  4. 使用更大的训练数据集（100k）
  5. 调整学习率和训练策略

Transformer模型改进方向：
  1. 尝试不同的位置编码方案（learned, relative）
  2. 尝试RMSNorm归一化方法
  3. 增加模型规模（d_model, num_heads, layers）
  4. 使用更大的训练数据集（100k）
  5. 调整学习率调度策略
  6. 尝试预训练模型（T5）进行微调

{'='*80}
"""
    
    print(comparison_report)
    output_file.write(comparison_report)
    output_file.flush()
    
    # ==================== 总结 ====================
    end_time = datetime.now()
    duration = end_time - start_time
    
    summary = f"""
{'='*80}
训练和评估流程完成
{'='*80}

开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
总耗时: {duration}

输出文件: {output_file_path}

Checkpoint位置:
- RNN模型: checkpoints/checkpoints_rnn/
- Transformer模型: checkpoints/checkpoints_transformer/

{'='*80}
"""
    print(summary)
    output_file.write(summary)
    output_file.close()
    
    print(f"\n所有输出已保存到: {output_file_path}")


# ====================================================================================
# BLEU 统一对比评测（已合并到 run_mt_pipeline.py）
# - 只评测，不训练
# - 自动从各目录挑选 best checkpoint
# - 在同一 test.jsonl 上输出 Greedy/Beam 的 corpus BLEU-4 总表
# ====================================================================================

def bleu_compute_corpus_bleu(candidates, references, max_n: int = 4) -> float:
    import math
    import collections

    p_numerators = collections.defaultdict(int)
    p_denominators = collections.defaultdict(int)
    hyp_lengths = 0
    ref_lengths = 0

    for cand, refs in zip(candidates, references):
        hyp_lengths += len(cand)
        best_ref_len = min((len(ref) for ref in refs), key=lambda x: abs(x - len(cand)))
        ref_lengths += best_ref_len

        for n in range(1, max_n + 1):
            cand_ngrams = [tuple(cand[i : i + n]) for i in range(len(cand) - n + 1)]
            cand_cnt = collections.Counter(cand_ngrams)
            p_denominators[n] += len(cand_ngrams)

            max_ref_cnt = collections.defaultdict(int)
            for ref in refs:
                ref_ngrams = [tuple(ref[i : i + n]) for i in range(len(ref) - n + 1)]
                ref_cnt = collections.Counter(ref_ngrams)
                for gram, count in ref_cnt.items():
                    max_ref_cnt[gram] = max(max_ref_cnt[gram], count)

            for gram, count in cand_cnt.items():
                p_numerators[n] += min(count, max_ref_cnt.get(gram, 0))

    if hyp_lengths > ref_lengths:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_lengths / hyp_lengths) if hyp_lengths > 0 else 0.0

    weights = [0.25] * 4
    log_precisions = []
    for n in range(1, max_n + 1):
        if p_denominators[n] == 0:
            log_precisions.append(float("-inf"))
        else:
            if p_numerators[n] == 0:
                p_n = 1.0 / (2.0 * p_denominators[n])
            else:
                p_n = p_numerators[n] / p_denominators[n]
            log_precisions.append(math.log(p_n))

    if all(p == float("-inf") for p in log_precisions):
        return 0.0
    s = sum(w * lp for w, lp in zip(weights, log_precisions))
    return bp * math.exp(s)


def bleu_safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def bleu_extract_epoch_from_name(path: str):
    m = re.search(r"epoch(\d+)", os.path.basename(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def bleu_find_best_checkpoint(checkpoint_dir: str):
    """
    挑选 best checkpoint：
    - 优先 valid_loss 最小
    - 若没有 valid_loss，则 epoch 最大（从 ckpt['epoch'] 或文件名 epochN 解析）
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
            ep = bleu_extract_epoch_from_name(p)

        if v is not None:
            v = bleu_safe_float(v)
            if v is not None and v < best_valid:
                best_valid = v
                best_path = p
                best_epoch = int(ep) if ep is not None else best_epoch
        else:
            if ep is not None and int(ep) > best_epoch:
                best_epoch = int(ep)
                best_path = p

    return best_path


def bleu_load_rnn_or_transformer(checkpoint_path: str, model_type: str, device):
    from data_utils import SPMVocabWrapper
    from models.models_rnn import EncoderRNN, DecoderRNN, Seq2Seq
    from models.models_transformer import TransformerNMT

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt.get("args", {})
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]

    if model_type == "rnn":
        embed_dim = args.get("embed_dim", 256)
        hidden_size = args.get("hidden_size", 512)
        num_layers = args.get("num_layers", 2)
        dropout = args.get("dropout", 0.1)
        attn_type = args.get("attn_type", "dot")
        encoder = EncoderRNN(len(src_vocab.itos), embed_dim, hidden_size, num_layers, dropout)
        decoder = DecoderRNN(len(tgt_vocab.itos), embed_dim, hidden_size, num_layers, dropout, attn_type)
        model = Seq2Seq(encoder, decoder)
    else:
        model = TransformerNMT(
            src_vocab_size=len(src_vocab.itos),
            tgt_vocab_size=len(tgt_vocab.itos),
            d_model=args.get("d_model", 512),
            num_heads=args.get("num_heads", 8),
            num_encoder_layers=args.get("num_encoder_layers", 3),
            num_decoder_layers=args.get("num_decoder_layers", 3),
            dim_ff=args.get("dim_ff", 2048),
            dropout=args.get("dropout", 0.3),
            max_len=args.get("max_len", 128),
            pos_encoding=args.get("pos_encoding", "sinusoidal"),
            norm_type=args.get("norm_type", "layernorm"),
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, src_vocab, tgt_vocab, ckpt


def bleu_eval_rnn_or_transformer(checkpoint_path: str, model_type: str, test_data, device, beam_size: int, max_len: int):
    from data_utils import tokenize_en, tokenize_zh, SPMVocabWrapper

    model, src_vocab, tgt_vocab, ckpt = bleu_load_rnn_or_transformer(checkpoint_path, model_type, device)

    def _run(beam: int) -> float:
        candidates = []
        references = []

        for item in test_data:
            src_text = item["zh"]
            tgt_text = item["en"]

            if isinstance(src_vocab, SPMVocabWrapper):
                src_ids = src_vocab.encode(src_text, add_sos_eos=True)
            else:
                src_tokens = tokenize_zh(src_text)
                src_ids = src_vocab.encode(src_tokens, add_sos_eos=True)

            src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
            src_mask = src_tensor.ne(0)

            with torch.no_grad():
                if beam > 1:
                    pred_ids = model.beam_search(src_tensor, src_mask, beam_size=beam, max_len=max_len)
                else:
                    pred_ids = model.greedy_decode(src_tensor, src_mask, max_len=max_len)

            if isinstance(tgt_vocab, SPMVocabWrapper):
                pred_str = tgt_vocab.decode_to_sentence(pred_ids[0].tolist())
                pred_tokens = tokenize_en(pred_str)
            else:
                pred_tokens = tgt_vocab.decode(pred_ids[0].tolist(), remove_special=True)

            ref_tokens = tokenize_en(tgt_text)
            candidates.append(pred_tokens)
            references.append([ref_tokens])

        return bleu_compute_corpus_bleu(candidates, references)

    greedy_bleu = _run(beam=1)
    beam_bleu = _run(beam=beam_size)
    return greedy_bleu, beam_bleu, ckpt


def bleu_eval_t5(checkpoint_path: str, test_data, device, beam_size: int, max_len: int):
    from data_utils import tokenize_en
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from huggingface_hub import snapshot_download

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt.get("args", {})
    model_name = args.get("model_name", "google-t5/t5-small")

    cache_root = "/data/250010066/LLM_course/final_program/t5-model"
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    local_cache_dir = os.path.join(cache_root, model_name_safe)

    MIRROR_ENDPOINT = "https://hf-mirror.com"
    os.environ.setdefault("HF_ENDPOINT", MIRROR_ENDPOINT)
    os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", MIRROR_ENDPOINT)

    def _resolve_snapshot_dir():
        if os.path.isdir(local_cache_dir):
            try:
                snap_dir = snapshot_download(
                    repo_id=model_name,
                    cache_dir=local_cache_dir,
                    local_files_only=True,
                    resume_download=True,
                    allow_patterns=["*.json", "*.bin", "*.model", "*.safetensors"],
                    ignore_patterns=["*.md", "*.txt", "*.h5", "*.ot", "*.msgpack"],
                )
                return snap_dir, True
            except Exception:
                repo_dir = os.path.join(local_cache_dir, "models--" + model_name.replace("/", "--"))
                snaps_root = os.path.join(repo_dir, "snapshots")
                if os.path.isdir(snaps_root):
                    snaps = [os.path.join(snaps_root, d) for d in os.listdir(snaps_root)]
                    snaps = [p for p in snaps if os.path.isdir(p)]
                    if snaps:
                        snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                        return snaps[0], True

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

    model_dir, local_only = _resolve_snapshot_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=local_only)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    def _run(beam: int) -> float:
        candidates = []
        references = []

        for item in test_data:
            src_text = item["zh"]
            tgt_text = item["en"]

            input_text = f"translate Chinese to English: {src_text}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.no_grad():
                if beam > 1:
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_len,
                        num_beams=beam,
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

        return bleu_compute_corpus_bleu(candidates, references)

    greedy_bleu = _run(beam=1)
    beam_bleu = _run(beam=beam_size)
    return greedy_bleu, beam_bleu, ckpt


def bleu_compare_main(data_dir: str, beam_size: int = 5, max_len: int = 80, limit: int = 0) -> str:
    from data_utils import load_jsonl

    test_file = os.path.join(data_dir, "test.jsonl")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试集不存在: {test_file}")

    test_data = load_jsonl(test_file)
    if limit and limit > 0:
        test_data = test_data[:limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bleu_compare_{stamp}.txt")

    suites = []
    suites += [
        ("RNN attn=dot (baseline)", "rnn", "checkpoints/checkpoints_rnn"),
        ("RNN attn=general", "rnn", "checkpoints/checkpoints_rnn_attn_general"),
        ("RNN attn=additive", "rnn", "checkpoints/checkpoints_rnn_attn_additive"),
    ]
    suites += [
        ("RNN TF=teacher_forcing", "rnn", "checkpoints/checkpoints_rnn_tf_teacher_forcing"),
        ("RNN TF=free_running", "rnn", "checkpoints/checkpoints_rnn_tf_free_running"),
    ]
    suites += [
        ("TFM pos=sinusoidal", "transformer", "checkpoints/checkpoints_transformer_pos_sinusoidal"),
        ("TFM pos=learned", "transformer", "checkpoints/checkpoints_transformer_pos_learned"),
        ("TFM pos=relative", "transformer", "checkpoints/checkpoints_transformer_pos_relative"),
    ]
    suites += [
        ("TFM norm=layernorm", "transformer", "checkpoints/checkpoints_norm_layernorm"),
        ("TFM norm=rmsnorm", "transformer", "checkpoints/checkpoints_norm_rmsnorm"),
    ]
    suites += [
        ("HParam batch_size=64", "transformer", "checkpoints/checkpoints_sensitivity_bs64"),
        ("HParam batch_size=128", "transformer", "checkpoints/checkpoints_sensitivity_bs128"),
        ("HParam batch_size=256", "transformer", "checkpoints/checkpoints_sensitivity_bs256"),
        ("HParam lr=0.5", "transformer", "checkpoints/checkpoints_sensitivity_lr0.5"),
        ("HParam lr=1.0", "transformer", "checkpoints/checkpoints_sensitivity_lr1.0"),
        ("HParam lr=2.0", "transformer", "checkpoints/checkpoints_sensitivity_lr2.0"),
        ("HParam scale=small", "transformer", "checkpoints/checkpoints_sensitivity_scale_small"),
        ("HParam scale=medium", "transformer", "checkpoints/checkpoints_sensitivity_scale_medium"),
        ("HParam scale=large", "transformer", "checkpoints/checkpoints_sensitivity_scale_large"),
    ]
    suites += [("T5 fine-tuned", "t5", "checkpoints/checkpoints_t5")]

    rows = []

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("统一BLEU对比评测（Corpus BLEU-4）\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"设备: {device}\n")
        f.write(f"测试集: {test_file}\n")
        f.write(f"样本数: {len(test_data)}\n")
        f.write(f"beam_size={beam_size}, max_len={max_len}\n")
        f.write("=" * 110 + "\n\n")

        for name, model_type, ckpt_dir in suites:
            best_ckpt = bleu_find_best_checkpoint(ckpt_dir)
            if not best_ckpt:
                rows.append((name, model_type, ckpt_dir, None, None, None, None, None, "跳过：无checkpoint"))
                continue

            t0 = time.time()
            try:
                if model_type in ("rnn", "transformer"):
                    g, b, ckpt = bleu_eval_rnn_or_transformer(best_ckpt, model_type, test_data, device, beam_size, max_len)
                else:
                    g, b, ckpt = bleu_eval_t5(best_ckpt, test_data, device, beam_size, max_len)

                secs = time.time() - t0
                vl = bleu_safe_float(ckpt.get("valid_loss"))
                ep = ckpt.get("epoch") or bleu_extract_epoch_from_name(best_ckpt)
                ep = int(ep) if ep is not None else None

                rows.append((name, model_type, best_ckpt, vl, ep, float(g), float(b), float(secs), "OK"))

                f.write(f"[OK] {name}\n")
                f.write(f"  - type: {model_type}\n")
                f.write(f"  - checkpoint: {best_ckpt}\n")
                if vl is not None:
                    f.write(f"  - valid_loss: {vl:.6f}\n")
                if ep is not None:
                    f.write(f"  - epoch: {ep}\n")
                f.write(f"  - greedy_bleu: {g:.6f}\n")
                f.write(f"  - beam_bleu:   {b:.6f}\n")
                f.write(f"  - elapsed: {secs:.1f}s\n\n")
            except Exception as e:
                secs = time.time() - t0
                rows.append((name, model_type, best_ckpt, None, bleu_extract_epoch_from_name(best_ckpt), None, None, float(secs), f"失败: {e}"))
                f.write(f"[FAIL] {name}\n")
                f.write(f"  - type: {model_type}\n")
                f.write(f"  - checkpoint: {best_ckpt}\n")
                f.write(f"  - error: {e}\n")
                f.write(f"  - elapsed: {secs:.1f}s\n\n")

        f.write("=" * 110 + "\n")
        f.write("汇总表（可直接粘贴进报告）\n\n")
        f.write("| 实验/模型 | type | checkpoint | valid_loss | epoch | Greedy BLEU | Beam BLEU | 耗时(s) | 状态 |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for name, model_type, ckpt, vl, ep, gb, bb, secs, status in rows:
            vl_s = f"{vl:.4f}" if isinstance(vl, float) else ""
            ep_s = str(ep) if isinstance(ep, int) else ""
            gb_s = f"{gb:.4f}" if isinstance(gb, float) else ""
            bb_s = f"{bb:.4f}" if isinstance(bb, float) else ""
            sec_s = f"{secs:.1f}" if isinstance(secs, float) else ""
            f.write(f"| {name} | {model_type} | `{ckpt}` | {vl_s} | {ep_s} | {gb_s} | {bb_s} | {sec_s} | {status} |\n")
        f.write("\n")

    print(f"✅ 已生成BLEU汇总文件: {out_path}")
    return out_path


if __name__ == "__main__":
    ensure_torch_available()

    import argparse

    parser = argparse.ArgumentParser(description="MT unified runner (train + eval + bleu compare).")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "bleu_compare"],
        help="full: 完整训练+推理+评估；bleu_compare: 仅生成统一 BLEU 对比总表",
    )
    # bleu_compare 参数
    parser.add_argument("--data_dir", type=str, default="data_raw/AP0004_Midterm&Final_translation_dataset_zh_en")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0, help="仅评测前N条测试样本（0表示全量）")

    args = parser.parse_args()

    if args.mode == "bleu_compare":
        bleu_compare_main(args.data_dir, beam_size=args.beam_size, max_len=args.max_len, limit=args.limit)
    else:
        main()

