# 中英机器翻译项目

本项目实现了基于 RNN 和 Transformer 的神经机器翻译模型，满足课程作业的所有要求。

- **训练数据**：100k 中英平行语料
- **训练设备**：NVIDIA H100 80GB GPU
- **模型架构**：RNN (GRU + Attention) / Transformer (from scratch) / T5 (fine-tuned)
- **代码仓库**：https://github.com/wl-Simplecss/NLP-LLM-final

## 模型要求

### 0. 数据预处理与嵌入层配置 (Data & Embeddings)

#### 数据清洗
- ✅ 去除非法字符
- ✅ 过滤低频词
- ✅ 处理超长句子（过滤或截断）

#### 分词策略
- ✅ **英文**: 子词分割 (BPE / SentencePiece) - 已实现
- ✅ **中文**: Jieba 分词 - 已实现
- ✅ 支持传统词级分词和子词分词两种模式

#### 词汇表构建
- ✅ 从分词数据构建词汇表
- ✅ 过滤低频词（通过最小词频阈值）
- ✅ 支持特殊标记：`<pad>`, `<sos>`, `<eos>`, `<unk>`

#### 嵌入层
- ✅ 支持随机初始化嵌入层
- ✅ 支持预训练词向量初始化（可扩展）
- ✅ 嵌入层在训练过程中可微调（Fine-tuning）

### 1. RNN 模型架构要求 (RNN Architecture)

#### 核心单元
- ✅ 使用 **GRU** 实现编码器和解码器
- ✅ 支持 LSTM（可扩展）

#### 层级结构
- ✅ **Encoder**: 2 个单向层 (2 Unidirectional Layers) - 已实现双向GRU
- ✅ **Decoder**: 2 个单向层 (2 Unidirectional Layers)

#### 注意力机制
必须实现 Attention，并对比以下对齐函数的影响：
- ✅ **点积注意力** (Dot Product) - 已实现
- ✅ **乘性注意力** (Multiplicative / General) - 已实现
- ✅ **加性注意力** (Additive / Bahdanau) - 已实现
- ✅ 可通过 `--attn_type` 参数选择不同注意力类型

#### 训练策略
对比 Teacher Forcing vs. Free Running 的有效性：
- ✅ **Teacher Forcing** - 已实现（`--teacher_forcing_ratio=1.0`）
- ✅ **Free Running** - 已实现（`--teacher_forcing_ratio<1.0`）
- ✅ **Scheduled Sampling** - 已实现（动态调整Teacher Forcing比例）

#### 解码策略
对比 Greedy Search vs. Beam Search 的效果：
- ✅ **Greedy Search** - 已实现
- ✅ **Beam Search** - 已实现（`--beam_size` 参数控制）
- ✅ 支持重复惩罚和三元组阻塞（避免重复生成）

### 2. Transformer 模型架构要求 (Transformer Architecture)

#### 模型构建
- ✅ **从零构建** (From Scratch) 标准的 Encoder-Decoder Transformer 架构
- ✅ 完整的自注意力机制和交叉注意力机制
- ✅ 位置编码和前馈网络

#### 消融研究 - 架构对比

**位置编码对比**：
- ✅ **绝对位置编码** (Absolute PE / Sinusoidal) - 已实现
- ✅ **学习位置编码** (Learned Positional Encoding) - 已实现
- ✅ **相对位置编码** (Relative PE) - 已实现
- ✅ 可通过 `--pos_encoding` 参数选择

**归一化对比**：
- ✅ **LayerNorm** - 已实现
- ✅ **RMSNorm** - 已实现
- ✅ 可通过 `--norm_type` 参数选择

#### 超参数敏感性
需评估以下超参数对性能的影响：
- ✅ **Batch Size** - 支持不同批量大小（`--batch_size`）
- ✅ **Learning Rate** - 支持不同学习率（`--lr`，使用Noam调度）
- ✅ **Model Scale** - 支持不同模型规模：
  - `--d_model`: 模型维度
  - `--num_heads`: 注意力头数
  - `--num_encoder_layers`: 编码器层数
  - `--num_decoder_layers`: 解码器层数
  - `--dim_ff`: 前馈网络维度

#### 预训练模型对比
- ✅ **任务**: 微调 (Fine-tune) 现有预训练模型（T5）
- ✅ **目标**: 将其性能与从零训练的 Transformer 进行对比
- ✅ 已实现 T5 微调脚本（`train/train_t5.py`）
- ✅ 支持 t5-small, t5-base 等不同规模的 T5 模型

## 项目结构

```
.
├── run_mt_pipeline.py          # 统一入口脚本（训练/推理/评估/统一BLEU对比）
├── inference.py                # 一键推理脚本（支持 --demo 和 --interactive 模式）
├── evaluate.py                 # BLEU评估脚本（Corpus-level BLEU）
├── data_utils.py               # 数据预处理工具（分词、词表构建等）
├── requirements.txt            # Python依赖
├── environment.yml             # Conda环境配置
├── README.md                   # 项目说明文档
│
├── models/                     # 模型结构定义
│   ├── __init__.py
│   ├── models_rnn.py           # RNN模型实现（Encoder-Decoder + Attention）
│   └── models_transformer.py   # Transformer模型实现
│
├── train/                      # 模型训练脚本
│   ├── __init__.py
│   ├── train_rnn.py            # RNN模型训练脚本
│   ├── train_transformer.py    # Transformer模型训练脚本
│   ├── train_spm.py            # SentencePiece分词模型训练脚本
│   └── train_t5.py             # T5预训练模型微调脚本
│
├── data_raw/                   # 数据集目录（上传GitHub）
│   └── AP0004_Midterm&Final_translation_dataset_zh_en/
│       ├── train_10k.jsonl
│       ├── train_100k.jsonl
│       ├── valid.jsonl
│       └── test.jsonl
│
├── checkpoints_best/           # 最佳模型checkpoint（上传GitHub，供推理使用）
│   ├── rnn_best.pt
│   ├── transformer_best.pt
│   └── t5_best.pt
│
├── spm/                        # SentencePiece分词模型（上传GitHub）
│   ├── spm_zh.model
│   ├── spm_zh.vocab
│   ├── spm_en.model
│   └── spm_en.vocab
│
├── script/                     # 可视化脚本
│   └── plot_*.py
│
│  ===== 以下内容不上传 GitHub =====
│
├── checkpoints/                # 训练过程的checkpoint（不上传）
├── outputs/                    # 训练日志输出（不上传）
├── t5-model/                   # T5模型本地缓存（不上传）
├── report/                     # 报告文件夹（不上传，单独提交）
└── *.sh                        # Shell脚本（不上传）
```

## 功能实现清单

### 1. 基于RNN的NMT模型 ✅

- **模型架构**：
  - ✅ 使用GRU实现编码器和解码器
  - ✅ 编码器和解码器均为两层单向层（`num_layers=2`）
  
- **注意力机制**：
  - ✅ 点积注意力（dot-product）
  - ✅ 乘性注意力（multiplicative/general）
  - ✅ 加性注意力（additive/Bahdanau）
  - ✅ 可通过`--attn_type`参数选择不同注意力类型

- **训练策略**：
  - ✅ Teacher Forcing（默认，`--teacher_forcing_ratio=1.0`）
  - ✅ Free Running（`--teacher_forcing_ratio<1.0`）
  - ✅ 支持混合训练策略

- **解码策略**：
  - ✅ 贪婪解码（greedy）
  - ✅ 集束搜索（beam-search）
  - ✅ 可通过`--beam_size`参数控制

### 2. 基于Transformer的NMT模型 ✅

- **从头训练**：
  - ✅ 完整的Encoder-Decoder Transformer架构
  - ✅ 多头自注意力和交叉注意力机制
  - ✅ 位置编码和前馈网络

- **架构消融实验**：
  - ✅ 位置编码方案：
    - 绝对位置编码（sinusoidal）
    - 学习位置编码（learned）
    - 相对位置编码（relative）
  - ✅ 归一化方法：
    - LayerNorm
    - RMSNorm
  - ✅ 可通过`--pos_encoding`和`--norm_type`参数选择

- **超参数敏感性**：
  - ✅ 支持不同批量大小（`--batch_size`）
  - ✅ 支持不同学习率（`--lr`）
  - ✅ 支持不同模型规模（`--d_model`, `--num_heads`, `--num_encoder_layers`, `--num_decoder_layers`, `--dim_ff`）

- **基于预训练语言模型**：
  - ✅ T5模型微调脚本（`train_t5.py`）
  - ✅ 支持t5-small, t5-base等不同规模的T5模型
  - ✅ 使用HuggingFace Transformers库

### 3. 数据预处理 ✅

- ✅ 数据清洗：过滤过长句子、非法字符
- ✅ 中文分词：使用Jieba
- ✅ 英文分词：空格分词（可扩展为BPE/WordPiece）
- ✅ 词表构建：统计词频，过滤低频词
- ✅ 支持最大长度限制（`--max_len`）

### 4. 评估指标 ✅

- ✅ BLEU-4分数计算（`evaluate.py`）
- ✅ 支持在测试集上评估
- ✅ 支持不同解码策略的评估

### 5. 推理脚本 ✅

- ✅ 一键推理脚本（`inference.py`）
- ✅ 支持RNN、Transformer和T5模型
- ✅ 支持贪婪解码和集束搜索
- ✅ 自动加载checkpoint和词表

## 使用方法

### 环境配置

使用Conda创建环境：
```bash
conda env create -f environment.yml
conda activate llm_final_pro
```

或使用pip安装依赖：
```bash
pip install -r requirements.txt
```

### 一键运行完整流程

```bash
# 运行完整训练和评估流程（推荐）
python run_mt_pipeline.py

# 仅生成统一 BLEU 对比总表（不训练）
python run_mt_pipeline.py --mode bleu_compare --limit 2
```

### 单独训练RNN模型

```bash
# 使用10k数据集，点积注意力，Teacher Forcing
python train/train_rnn.py --use_small --attn_type dot --teacher_forcing_ratio 1.0 --epochs 10

# 使用100k数据集，加性注意力，Free Running
python train/train_rnn.py --attn_type additive --teacher_forcing_ratio 0.5 --epochs 10

# 不同注意力机制对比
python train/train_rnn.py --use_small --attn_type dot --epochs 5
python train/train_rnn.py --use_small --attn_type general --epochs 5
python train/train_rnn.py --use_small --attn_type additive --epochs 5
```

### 单独训练Transformer模型

```bash
# 标准配置（sinusoidal位置编码 + LayerNorm）
python train/train_transformer.py --use_small --epochs 10

# 相对位置编码 + RMSNorm
python train/train_transformer.py --use_small --pos_encoding relative --norm_type rmsnorm --epochs 10

# 不同超参数配置
python train/train_transformer.py --use_small --d_model 512 --num_heads 8 --batch_size 32 --lr 5e-4 --epochs 10
```

### 训练SentencePiece分词模型

```bash
# 训练中英文SentencePiece模型
python train/train_spm.py --data_dir data_raw/AP0004_Midterm&Final_translation_dataset_zh_en --vocab_size 8000
```

### 微调T5模型

```bash
# 使用t5-small
python train/train_t5.py --use_small --model_name t5-small --epochs 5

# 使用t5-base（需要更多资源）
python train/train_t5.py --model_name t5-base --epochs 5
```

### 推理

**一键演示模式**（推荐，使用最佳模型）：
```bash
python inference.py --demo
```

**交互式翻译模式**：
```bash
python inference.py --interactive
```

**指定模型推理**：
```bash
# RNN模型推理（使用最佳模型）
python inference.py --checkpoint checkpoints_best/rnn_best.pt --model_type rnn --text "今天天气很好" --beam_size 5

# Transformer模型推理（使用最佳模型）
python inference.py --checkpoint checkpoints_best/transformer_best.pt --model_type transformer --text "今天天气很好" --beam_size 5

# T5模型推理（使用最佳模型）
python inference.py --checkpoint checkpoints_best/t5_best.pt --model_type t5 --text "今天天气很好" --beam_size 5
```

### 评估

```bash
# 评估RNN最佳模型
python evaluate.py --checkpoint checkpoints_best/rnn_best.pt --model_type rnn --beam_size 5

# 评估Transformer最佳模型
python evaluate.py --checkpoint checkpoints_best/transformer_best.pt --model_type transformer --beam_size 5

# 统一BLEU对比评测（评测所有实验配置）
python run_mt_pipeline.py --mode bleu_compare
```

## 实验建议

### RNN模型实验

1. **注意力机制对比**：
   - 分别使用dot、general、additive训练模型
   - 比较验证集损失和BLEU分数

2. **训练策略对比**：
   - Teacher Forcing vs Free Running
   - 可以尝试不同的`teacher_forcing_ratio`值（0.5, 0.7, 1.0）

3. **解码策略对比**：
   - Greedy vs Beam Search（beam_size=4, 8）

### Transformer模型实验

1. **位置编码对比**：
   - sinusoidal vs learned vs relative
   - 比较训练速度和翻译质量

2. **归一化方法对比**：
   - LayerNorm vs RMSNorm
   - 比较训练稳定性和性能

3. **超参数敏感性**：
   - 不同batch_size（16, 32, 64）
   - 不同学习率（1e-4, 5e-4, 1e-3）
   - 不同模型规模（d_model: 256, 512）

4. **预训练模型对比**：
   - 从头训练的Transformer vs 微调的T5
   - 比较训练时间、资源消耗和最终性能

## 注意事项

1. 本项目使用 **100k 大训练集** 在 **NVIDIA H100 80GB GPU** 上完成训练
2. 如果计算资源有限，可使用 `--use_small` 参数训练 10k 数据集
3. T5 模型需要下载预训练权重，首次运行会较慢（使用 HuggingFace 镜像加速）
4. 建议使用 GPU 训练，CPU 训练会非常慢
5. Checkpoint 会自动保存模型状态、词表和训练参数

## GitHub 上传策略

本项目的 `.gitignore` 配置如下：

| 内容 | 是否上传 | 说明 |
|------|----------|------|
| `*.py` | ✅ 上传 | Python 源码 |
| `models/`, `train/` | ✅ 上传 | 模型定义和训练脚本 |
| `data_raw/` | ✅ 上传 | 数据集 |
| `checkpoints_best/` | ✅ 上传 | 最佳模型（供推理） |
| `spm/` | ✅ 上传 | 分词模型 |
| `README.md` | ✅ 上传 | 项目说明 |
| `checkpoints/` | ❌ 忽略 | 训练过程的检查点 |
| `outputs/` | ❌ 忽略 | 训练日志 |
| `t5-model/` | ❌ 忽略 | T5 模型本地缓存 |
| `report/` | ❌ 忽略 | 报告文件夹（单独提交） |
| `*.sh` | ❌ 忽略 | Shell 脚本 |

## 文件说明

### 根目录文件
- `run_mt_pipeline.py`: 统一入口脚本（训练/推理/评估/统一BLEU对比），推荐只使用该脚本
- `inference.py`: 统一推理接口，支持所有模型类型（RNN、Transformer、T5）
- `evaluate.py`: BLEU分数计算和模型评估（使用Corpus-level BLEU）
- `data_utils.py`: 数据加载、分词、词表构建、数据批处理、SentencePiece支持

### models/ 目录
- `models/models_rnn.py`: RNN编码器、解码器、注意力机制、Seq2Seq模型
- `models/models_transformer.py`: Transformer各组件、位置编码、归一化、完整模型

### train/ 目录
- `train/train_rnn.py`: RNN模型训练循环、Scheduled Sampling实现
- `train/train_transformer.py`: Transformer模型训练循环
- `train/train_spm.py`: SentencePiece分词模型训练脚本
- `train/train_t5.py`: T5模型微调脚本

## 依赖版本

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.40.0
- Jieba >= 0.42.1
- 其他依赖见`requirements.txt`

