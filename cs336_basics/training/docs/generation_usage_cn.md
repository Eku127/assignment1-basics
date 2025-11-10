# 文本生成使用指南

本文档详细介绍如何使用已实现的文本生成功能。

## 📂 实现文件

### 1. `decode.py` - 核心解码功能

实现了三个关键函数：

#### `generate_text()` - 主生成函数
自回归生成文本，支持所有解码策略。

**参数说明**：
- `model`: 训练好的 TransformerLM 模型
- `tokenizer`: BPE tokenizer 用于编码/解码
- `prompt`: 输入提示文本
- `max_tokens`: 最大生成 token 数（默认：100）
- `temperature`: 采样温度（默认：1.0）
  - → 0: 接近贪心解码（确定性）
  - = 1.0: 标准采样
  - > 1.0: 更随机/创造性
- `top_p`: Nucleus 采样阈值（默认：1.0，无过滤）
  - = 1.0: 使用完整词汇表
  - = 0.9: 只使用概率累积到 90% 的 token
- `device`: 运行设备（'cpu', 'cuda', 'mps'）
- `eos_token_id`: 结束 token ID（可选）

**返回**：生成的完整文本（包含 prompt）

#### `apply_temperature()` - 温度缩放
对 logits 应用温度缩放，控制采样的随机性。

**公式**：
```
softmax(v, τ)_i = exp(v_i / τ) / Σ exp(v_j / τ)
```

#### `top_p_filtering()` - Nucleus 采样过滤
实现 Top-p（核）采样，只保留累积概率达到 p 的最小 token 集合。

**算法步骤**：
1. 计算概率分布（softmax）
2. 按概率降序排序
3. 计算累积概率
4. 找到累积概率超过 p 的截断点
5. 过滤掉低概率 token

### 2. `generate.py` - 命令行生成脚本

功能完整的命令行工具，用于从检查点生成文本。

## 🚀 使用方法

### 方法一：Python API

```python
import torch
from cs336_basics.transformer import TransformerLM
from cs336_basics.bpe import Tokenizer
from cs336_basics.training import generate_text

# 1. 加载 tokenizer
tokenizer = Tokenizer.from_files(
    vocab_filepath="data/vocab.json",
    merges_filepath="data/merges.txt",
    special_tokens=["<|endoftext|>"]
)

# 2. 加载模型
model = TransformerLM(
    vocab_size=10000,
    context_length=1024,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    use_rope=True,
)
checkpoint = torch.load("model_checkpoint.pt")
model.load_state_dict(checkpoint['model'])
model.eval()

# 3. 生成文本
text = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time, in a land far away,",
    max_tokens=150,
    temperature=0.8,
    top_p=0.9,
    device='cuda'
)

print(text)
```

### 方法二：命令行脚本

#### 基础用法

```bash
python cs336_basics/training/generate.py \
    --checkpoint /path/to/checkpoint.pt \
    --vocab /path/to/vocab.json \
    --merges /path/to/merges.txt \
    --prompt "Once upon a time" \
    --max_new_tokens 100
```

#### 示例 1: 贪心解码（确定性输出）

```bash
python cs336_basics/training/generate.py \
    --checkpoint model.pt \
    --vocab vocab.json \
    --merges merges.txt \
    --prompt "The quick brown fox" \
    --temperature 0.01 \
    --max_new_tokens 50 \
    --verbose
```

**适用场景**：
- 需要确定性输出
- 评估任务
- 希望模型输出最可能的延续

#### 示例 2: 平衡的创造性采样

```bash
python cs336_basics/training/generate.py \
    --checkpoint model.pt \
    --vocab vocab.json \
    --merges merges.txt \
    --prompt "Once upon a time, in a land far away," \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_new_tokens 200
```

**适用场景**：
- 故事生成
- 对话系统
- 需要多样性但保持连贯性

#### 示例 3: 高创造性生成

```bash
python cs336_basics/training/generate.py \
    --checkpoint model.pt \
    --vocab vocab.json \
    --merges merges.txt \
    --prompt "In the year 3000," \
    --temperature 1.5 \
    --top_p 0.95 \
    --max_new_tokens 150 \
    --seed 42
```

**适用场景**：
- 创意写作
- 头脑风暴
- 探索模型的多样性

#### 示例 4: 指定模型架构

如果你的模型使用非默认配置：

```bash
python cs336_basics/training/generate.py \
    --checkpoint my_model.pt \
    --vocab vocab.json \
    --merges merges.txt \
    --prompt "Hello world" \
    --vocab_size 50000 \
    --context_length 2048 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --d_ff 3072 \
    --temperature 0.8 \
    --top_p 0.9
```

## 📊 参数调优建议

### Temperature (温度)

| 温度值 | 效果 | 适用场景 |
|--------|------|----------|
| 0.01 - 0.3 | 保守、确定性强 | 事实性任务、代码生成 |
| 0.5 - 0.8 | 平衡 | 故事生成、对话 |
| 0.9 - 1.2 | 创造性 | 创意写作、诗歌 |
| 1.3 - 2.0 | 非常随机 | 实验、头脑风暴 |

### Top-p (Nucleus Sampling)

| top_p 值 | 效果 | 适用场景 |
|----------|------|----------|
| 0.5 - 0.7 | 非常保守 | 需要高质量、低风险输出 |
| 0.8 - 0.9 | 平衡（推荐） | 大多数应用 |
| 0.92 - 0.95 | 更多样化 | 创意任务 |
| 1.0 | 无过滤 | 通常与低温度结合 |

### 推荐组合

```python
# 组合 1: 高质量、连贯的输出
temperature = 0.7
top_p = 0.9

# 组合 2: 非常保守（接近确定性）
temperature = 0.1
top_p = 0.95

# 组合 3: 创造性、多样化
temperature = 1.0
top_p = 0.95

# 组合 4: 纯贪心（完全确定性）
temperature = 0.001  # 接近 0
top_p = 1.0
```

## 🔍 常见问题

### Q1: 生成的文本重复怎么办？

**解决方案**：
1. 增加温度（例如从 0.5 → 0.8）
2. 降低 top_p（例如从 1.0 → 0.9）
3. 检查模型是否训练充分

### Q2: 生成的文本不连贯/胡言乱语？

**解决方案**：
1. 降低温度（例如从 1.5 → 0.8）
2. 使用 top-p 过滤（设置 top_p=0.9）
3. 检查模型质量

### Q3: 如何加快生成速度？

**建议**：
1. 使用 GPU（`--device cuda`）
2. 减少 `max_new_tokens`
3. 考虑批量生成多个序列
4. 使用较小的模型

### Q4: 如何实现确定性输出？

```bash
python cs336_basics/training/generate.py \
    --checkpoint model.pt \
    --vocab vocab.json \
    --merges merges.txt \
    --prompt "Hello" \
    --temperature 0.001 \
    --seed 42
```

设置固定的 seed 和极低的温度。

### Q5: 如何处理超长上下文？

代码会自动截断输入以适应模型的 `context_length`：
```python
if input_ids.size(1) > model.context_length:
    model_input = input_ids[:, -model.context_length:]
```

只保留最近的 token。

## 📝 技术细节

### 自回归生成流程

```
1. 编码 prompt → token IDs: [1, 2, 3]
2. 循环生成:
   a. 前向传播: logits = model([1, 2, 3])
   b. 取最后一个位置: logits[-1]
   c. 应用温度: logits / temperature
   d. 应用 top-p 过滤
   e. Softmax + 采样: next_token = 4
   f. 检查是否为 EOS token
   g. 追加: [1, 2, 3, 4]
3. 重复直到生成 EOS 或达到最大长度
4. 解码 token IDs → 文本
```

### Top-p 过滤算法

```python
1. 计算概率: probs = softmax(logits)
2. 排序: sorted_probs, sorted_indices = sort(probs, descending=True)
3. 累积概率: cumsum = cumulative_sum(sorted_probs)
4. 找截断点: keep = (cumsum <= top_p)
5. 创建 mask: 过滤掉低概率 token
6. 重新归一化并采样
```

### 内存优化

- 使用 `model.eval()` 和 `torch.no_grad()` 节省内存
- 自动截断超长序列以适应 context_length
- 单个序列生成（batch_size=1）

## 🎯 任务要求对照

根据 CS336 Assignment 1 的要求，我们已实现：

✅ **生成补全**：接受 prompt，生成直到 `<|endoftext|>`  
✅ **最大长度控制**：`--max_new_tokens` 参数  
✅ **温度采样**：`apply_temperature()` 函数 + `--temperature` 参数  
✅ **Top-p 采样**：`top_p_filtering()` 函数 + `--top_p` 参数  

所有功能均已完整实现且可通过命令行和 Python API 使用。

## 📚 参考文献

- Holtzman et al. (2020). "The Curious Case of Neural Text Degeneration"
  - 提出了 Nucleus (top-p) 采样方法
- Fan et al. (2018). "Hierarchical Neural Story Generation"
  - 温度采样的应用研究

