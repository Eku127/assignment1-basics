# 训练脚本使用指南

本目录包含用于训练 Transformer 语言模型的便捷脚本。

## 📋 脚本列表

### `train_tinystories.sh`

用于在 TinyStories 数据集上训练 Transformer 语言模型的 Bash 脚本。该脚本已经配置好了符合 CS336 Assignment 1 要求的默认超参数，并会自动创建包含训练参数的 checkpoint 目录名。

### `generate.sh`

用于从训练好的 checkpoint 生成文本的 Bash 脚本。支持多种解码策略（temperature sampling、top-p sampling），可以方便地测试不同生成参数的效果。

---

## 🚀 快速开始

### 基本使用

1. **修改数据路径**（如果默认路径不正确）：
   
   编辑脚本第 13-14 行，修改默认的数据路径：
   ```bash
   TRAIN_DATA="./data/tinystories/train_tokens.npy"
   VAL_DATA="./data/tinystories/val_tokens.npy"
   ```

2. **直接运行脚本**：
   ```bash
   bash cs336_basics/training/scripts/train_tinystories_bs.sh
   ```
   
   或者使其可执行后直接运行：
   ```bash
   chmod +x cs336_basics/training/scripts/train_tinystories.sh
   ./cs336_basics/training/scripts/train_tinystories.sh
   ```

---

## ⚙️ 配置超参数

### 方式 1：修改脚本中的默认值

直接编辑 `train_tinystories.sh` 文件，修改相应的变量值。例如：

```bash
# 修改学习率
LEARNING_RATE=1e-3

# 修改批次大小
BATCH_SIZE=128

# 修改 warmup 步数
WARMUP_STEPS=3000
```

### 方式 2：通过环境变量覆盖

在运行脚本前设置环境变量，可以覆盖脚本中的默认值：

```bash
# 只修改学习率
LEARNING_RATE=1e-3 bash cs336_basics/training/scripts/train_tinystories.sh

# 修改多个参数
LEARNING_RATE=5e-4 \
WARMUP_STEPS=3000 \
BETA1=0.9 \
BETA2=0.999 \
WEIGHT_DECAY=0.01 \
bash cs336_basics/training/scripts/train_tinystories.sh

# 使用自定义数据路径
TRAIN_DATA=/path/to/your/train.npy \
VAL_DATA=/path/to/your/val.npy \
bash cs336_basics/training/scripts/train_tinystories.sh
```

---

## 📊 默认超参数配置

脚本已经配置了符合任务要求的默认超参数：

### 模型参数
- `vocab_size`: 10000
- `context_length`: 256
- `d_model`: 512
- `num_layers`: 4
- `num_heads`: 16
- `d_ff`: 1344
- `use_rope`: true

### 训练参数
- `batch_size`: 64
- `max_steps`: 20000
- `learning_rate`: 3e-4（可通过环境变量覆盖）
- `min_lr`: 3e-5
- `warmup_steps`: 2000（可通过环境变量覆盖）
- `beta1`: 0.9（可通过环境变量覆盖）
- `beta2`: 0.95（可通过环境变量覆盖）
- `weight_decay`: 0.1（可通过环境变量覆盖）
- `grad_clip`: 1.0

### 日志和检查点
- `log_every`: 100（每 100 步打印一次训练日志）
- `val_every`: 1000（每 1000 步进行一次验证）
- `save_every`: 5000（每 5000 步保存一次 checkpoint）
- `checkpoint_dir`: `./checkpoints/tinystories/`（可通过环境变量覆盖）

---

## 📁 Checkpoint 目录命名规则

脚本会自动创建包含关键训练参数的 checkpoint 目录名，格式如下：

```
lr{learning_rate}_bs{batch_size}_layers{num_layers}_heads{num_heads}_d{d_model}_warmup{warmup_steps}_beta{beta1}-{beta2}_wd{weight_decay}
```

**示例**：
- `lr3e-4_bs64_layers4_heads16_d512_warmup2000_beta0.9-0.95_wd0.1`

这样的命名方式使得你可以轻松识别不同超参数配置对应的 checkpoint。

---

## 🔧 可配置的环境变量

以下参数可以通过环境变量进行覆盖：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `TRAIN_DATA` | `./data/tinystories/train_tokens.npy` | 训练数据路径 |
| `VAL_DATA` | `./data/tinystories/val_tokens.npy` | 验证数据路径 |
| `LEARNING_RATE` | `3e-4` | 最大学习率 |
| `MIN_LR` | `3e-5` | 最小学习率 |
| `WARMUP_STEPS` | `2000` | Warmup 步数 |
| `BETA1` | `0.9` | AdamW β1 参数 |
| `BETA2` | `0.95` | AdamW β2 参数 |
| `WEIGHT_DECAY` | `0.1` | 权重衰减系数 |
| `CHECKPOINT_BASE_DIR` | `./checkpoints/tinystories` | Checkpoint 基础目录 |
| `USE_WANDB` | `false` | 是否使用 Weights & Biases |
| `WANDB_PROJECT` | `CS336_TinyStories` | W&B 项目名称 |
| `WANDB_NAME` | （空）| W&B 运行名称（可选）|
| `DEVICE` | `cuda` | 训练设备（`cuda`, `cpu`, `mps`）|

---

## 📈 使用 Weights & Biases (W&B)

如果需要使用 W&B 进行实验跟踪，设置以下环境变量：

```bash
USE_WANDB=true \
WANDB_PROJECT="MyExperiment" \
WANDB_NAME="lr_3e-4_baseline" \
bash cs336_basics/training/scripts/train_tinystories.sh
```

**参数说明**：
- `USE_WANDB=true`: 启用 W&B 日志记录
- `WANDB_PROJECT`: W&B 项目名称（用于组织不同的实验）
- `WANDB_NAME`: 本次运行的名称（可选，如果不设置会使用自动生成的名称）

---

## 💡 使用示例

### 示例 1：基本训练

```bash
# 使用默认参数训练
bash cs336_basics/training/scripts/train_tinystories.sh
```

### 示例 2：学习率调优

```bash
# 测试不同的学习率
for lr in 1e-4 3e-4 1e-3 3e-3; do
    LEARNING_RATE=$lr \
    WANDB_NAME="lr_${lr}" \
    USE_WANDB=true \
    bash cs336_basics/training/scripts/train_tinystories.sh
done
```

### 示例 3：完整参数自定义

```bash
LEARNING_RATE=5e-4 \
WARMUP_STEPS=3000 \
BETA1=0.9 \
BETA2=0.999 \
WEIGHT_DECAY=0.01 \
USE_WANDB=true \
WANDB_PROJECT="HyperparameterSearch" \
WANDB_NAME="lr5e-4_warmup3k_beta0.9-0.999" \
TRAIN_DATA=/custom/path/train.npy \
VAL_DATA=/custom/path/val.npy \
bash cs336_basics/training/scripts/train_tinystories.sh
```

---

## ⚠️ 注意事项

1. **数据路径**：确保训练和验证数据文件存在，否则脚本会报错并退出。

2. **Checkpoint 目录**：脚本会自动创建 checkpoint 目录，确保有足够的磁盘空间。

3. **GPU 内存**：如果遇到 GPU 内存不足的问题，可以尝试减小 `batch_size`。

4. **训练时间**：完整的训练（20000 步）可能需要数小时，建议使用 `screen` 或 `tmux` 在后台运行。

5. **恢复训练**：如果需要从 checkpoint 恢复训练，需要在 `train.py` 中使用 `--resume_from` 参数（本脚本暂不支持，可以手动调用 `train.py`）。

---

## 🔍 故障排除

### 问题：找不到数据文件

**错误信息**：
```
Error: Training data file not found: ./data/tinystories/train_tokens.npy
```

**解决方法**：
1. 检查数据文件是否存在
2. 修改脚本中的默认路径，或
3. 通过环境变量设置正确的路径：
   ```bash
   TRAIN_DATA=/correct/path/train.npy VAL_DATA=/correct/path/val.npy bash train_tinystories.sh
   ```

### 问题：CUDA 内存不足

**解决方法**：
- 减小 `batch_size`（例如改为 32 或 16）
- 减小 `context_length`（如果允许的话）

---

---

## 📝 文本生成脚本 (`generate.sh`)

### 基本使用

```bash
# 基本生成（使用默认参数）
bash cs336_basics/training/scripts/generate.sh \
    --checkpoint ./checkpoints/model.pt \
    --prompt "Once upon a time"

# 使用自定义参数
bash cs336_basics/training/scripts/generate.sh \
    --checkpoint ./checkpoints/model.pt \
    --prompt "The robot" \
    --temperature 1.5 \
    --max_new_tokens 200 \
    --top_p 0.9
```

### 必需参数

- `--checkpoint PATH`: 模型 checkpoint 文件路径（`.pt` 文件）

### 可选参数

#### Tokenizer 路径
- `--vocab PATH`: 词汇表 JSON 文件路径（默认：`./data/bpe/vocab.json`）
- `--merges PATH`: BPE merges 文件路径（默认：`./data/bpe/merges.txt`）

#### 生成参数
- `--prompt TEXT`: 输入提示文本（默认：`"Once upon a time"`）
- `--max_new_tokens N`: 最大生成 token 数（默认：`100`）
- `--temperature T`: 采样温度（默认：`1.0`）
  - 较低值（0.1-0.5）：更保守、更确定性的输出
  - 较高值（1.0-2.0）：更随机、更有创意的输出
- `--top_p P`: Top-p (nucleus) 采样阈值（默认：`1.0`，即不使用）
  - 范围：(0, 1]
  - 较小的值会限制候选 token 集合

#### 模型架构参数
这些参数需要与训练时使用的参数一致：
- `--vocab_size N`: 词汇表大小（默认：`10000`）
- `--context_length N`: 上下文长度（默认：`256`）
- `--d_model N`: 模型维度（默认：`512`）
- `--num_layers N`: Transformer 层数（默认：`4`）
- `--num_heads N`: 注意力头数（默认：`16`）
- `--d_ff N`: Feed-forward 维度（默认：`1344`）
- `--no_rope`: 禁用 RoPE 位置编码（默认启用）

#### 其他选项
- `--device DEVICE`: 运行设备（`cuda`, `cpu`, `mps`，默认：`cuda:0`）
- `--seed N`: 随机种子（用于可重现性）
- `--eos_token TOKEN`: 结束 token（默认：`<|endoftext|>`）
- `--verbose`: 打印详细信息
- `-h, --help`: 显示帮助信息

### 使用环境变量

也可以通过环境变量设置参数：

```bash
VOCAB_FILE=./data/bpe/vocab.json \
MERGES_FILE=./data/bpe/merges.txt \
PROMPT="The robot walked" \
TEMPERATURE=0.8 \
MAX_NEW_TOKENS=150 \
bash cs336_basics/training/scripts/generate.sh \
    --checkpoint ./checkpoints/model.pt
```

### 使用示例

#### 示例 1：确定性生成（低温度）

```bash
bash cs336_basics/training/scripts/generate.sh \
    --checkpoint ./checkpoints/model.pt \
    --prompt "Once upon a time" \
    --temperature 0.1 \
    --max_new_tokens 100
```

#### 示例 2：创意生成（高温度 + top-p）

```bash
bash cs336_basics/training/scripts/generate.sh \
    --checkpoint ./checkpoints/model.pt \
    --prompt "In a distant galaxy" \
    --temperature 1.5 \
    --top_p 0.9 \
    --max_new_tokens 200
```

#### 示例 3：批量测试不同温度

```bash
for temp in 0.1 0.5 1.0 1.5; do
    echo "=== Temperature: $temp ==="
    bash cs336_basics/training/scripts/generate.sh \
        --checkpoint ./checkpoints/model.pt \
        --prompt "The robot" \
        --temperature $temp \
        --max_new_tokens 50
    echo ""
done
```

### 注意事项

1. **模型架构参数**：确保 `--vocab_size`、`--d_model`、`--num_layers` 等参数与训练时使用的参数一致，否则无法正确加载 checkpoint。

2. **Tokenizer 路径**：确保 `--vocab` 和 `--merges` 文件路径正确，且与训练时使用的 tokenizer 一致。

3. **Checkpoint 格式**：脚本支持两种 checkpoint 格式：
   - 包含 `'model'` 键的字典格式
   - 直接是模型 state_dict 的格式

4. **设备选择**：如果指定 `--device cuda` 但 CUDA 不可用，脚本会自动回退到 CPU。

---

## 📚 相关文档

- 训练脚本主文档：`../README.md`
- 训练函数文档：`../train.py`
- 生成函数文档：`../generate.py`
- 数据加载文档：`../docs/dataloader_strategy_cn.md`
- Checkpoint 文档：`../docs/checkpointing_cn.md`

