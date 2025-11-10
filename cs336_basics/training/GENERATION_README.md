# 文本生成（Decoding）实现总结

## ✅ 任务完成情况

根据 CS336 Assignment 1 的要求，已完整实现文本生成功能（3分）：

| 要求 | 实现 | 文件位置 |
|------|------|----------|
| ✅ 生成补全（prompt → completion） | `generate_text()` | `decode.py` |
| ✅ 控制最大生成 token 数 | `max_tokens` 参数 | `decode.py` |
| ✅ 温度采样（Temperature Scaling） | `apply_temperature()` | `decode.py` |
| ✅ Top-p 采样（Nucleus Sampling） | `top_p_filtering()` | `decode.py` |

## 📂 实现文件

```
cs336_basics/training/
├── decode.py                    # 核心解码函数（3个函数）
├── generate.py                  # 命令行生成脚本
├── docs/
│   ├── generation_cn.md         # 解码策略理论指南
│   └── generation_usage_cn.md   # 详细使用文档
└── demo/
    └── test_generation.py       # 测试脚本（已通过 ✅）
```

## 🔑 核心函数

### 1. `generate_text()` - 自回归文本生成

```python
from cs336_basics.training import generate_text

text = generate_text(
    model=model,              # TransformerLM 模型
    tokenizer=tokenizer,      # BPE tokenizer
    prompt="Once upon a time",
    max_tokens=100,           # 最大生成 token 数
    temperature=0.8,          # 采样温度
    top_p=0.9,               # Nucleus 采样阈值
    device='cuda',
    eos_token_id=None        # 自动从 tokenizer 获取
)
```

**功能**：
- 接受提示文本，自回归生成补全
- 自动截断超长输入以适应 context_length
- 遇到 `<|endoftext|>` 或达到最大长度时停止

### 2. `apply_temperature()` - 温度缩放

```python
from cs336_basics.training.decode import apply_temperature

scaled_logits = apply_temperature(logits, temperature=0.8)
```

**公式**: `softmax(v, τ)ᵢ = exp(vᵢ/τ) / Σⱼ exp(vⱼ/τ)`

**效果**:
- `τ → 0`: 接近贪心解码（确定性）
- `τ = 1.0`: 标准 softmax
- `τ > 1.0`: 更随机/创造性

### 3. `top_p_filtering()` - Nucleus 采样

```python
from cs336_basics.training.decode import top_p_filtering

filtered_logits = top_p_filtering(logits, top_p=0.9)
```

**算法**:
1. 计算概率分布（softmax）
2. 按概率降序排序
3. 累积概率直到 ≥ p
4. 过滤掉不在"核"中的 token

**效果**: 动态调整候选集大小，平衡多样性和连贯性

## 🚀 快速开始

### 方法一：命令行脚本

```bash
# 基础用法
python cs336_basics/training/generate.py \
    --checkpoint model.pt \
    --vocab vocab.json \
    --merges merges.txt \
    --prompt "Once upon a time" \
    --max_new_tokens 100 \
    --temperature 0.8 \
    --top_p 0.9

# 贪心解码（确定性）
python cs336_basics/training/generate.py \
    --checkpoint model.pt \
    --vocab vocab.json \
    --merges merges.txt \
    --prompt "Hello world" \
    --temperature 0.01 \
    --max_new_tokens 50

# 创造性生成
python cs336_basics/training/generate.py \
    --checkpoint model.pt \
    --vocab vocab.json \
    --merges merges.txt \
    --prompt "In the year 3000" \
    --temperature 1.5 \
    --top_p 0.95 \
    --max_new_tokens 200
```

### 方法二：Python API

```python
import torch
from cs336_basics.transformer import TransformerLM
from cs336_basics.bpe import Tokenizer
from cs336_basics.training import generate_text

# 加载 tokenizer
tokenizer = Tokenizer.from_files(
    vocab_filepath="vocab.json",
    merges_filepath="merges.txt",
    special_tokens=["<|endoftext|>"]
)

# 加载模型
model = TransformerLM(vocab_size=10000, context_length=1024, ...)
checkpoint = torch.load("model.pt")
model.load_state_dict(checkpoint['model'])
model.eval()

# 生成文本
text = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.8,
    top_p=0.9,
    device='cuda'
)
print(text)
```

## 🧪 测试

运行测试脚本验证实现：

```bash
python cs336_basics/training/demo/test_generation.py
```

**测试内容**:
- ✅ 温度缩放正确性
- ✅ Top-p 过滤正确性
- ✅ 组合策略效果
- ✅ 边界情况处理

所有测试已通过！

## 📊 参数调优建议

### 常用组合

| 场景 | Temperature | Top-p | 适用任务 |
|------|------------|-------|----------|
| 确定性 | 0.01 - 0.1 | 1.0 | 事实性任务、代码生成 |
| 保守 | 0.5 - 0.7 | 0.9 | 一般性文本生成 |
| 平衡（推荐）| 0.7 - 0.9 | 0.9 | 故事、对话 |
| 创造性 | 1.0 - 1.5 | 0.95 | 创意写作、头脑风暴 |

### 示例

```python
# 高质量、连贯输出（推荐）
temperature = 0.8
top_p = 0.9

# 接近确定性（评估用）
temperature = 0.01
top_p = 1.0

# 最大多样性
temperature = 1.2
top_p = 0.95
```

## 🔍 技术实现细节

### 自回归生成流程

```
输入: prompt = "Hello"
→ 编码: [101, 102]
→ 循环:
    1. 模型前向: logits = model([101, 102])
    2. 取最后位置: logits[-1]
    3. 温度缩放: logits / temperature
    4. Top-p 过滤: 保留累积概率 ≤ p 的 token
    5. Softmax + 采样: next_token = 103
    6. 检查 EOS: 如果 token == <|endoftext|> 则停止
    7. 追加: [101, 102, 103]
→ 解码: "Hello world"
```

### Top-p 过滤实现

```python
# 1. 计算概率
probs = softmax(logits)

# 2. 排序（降序）
sorted_probs, sorted_indices = sort(probs, descending=True)

# 3. 累积概率
cumsum = cumulative_sum(sorted_probs)

# 4. 找截断点（保留 cumsum ≤ top_p 的 token）
indices_to_remove = cumsum > top_p
indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
indices_to_remove[..., 0] = False  # 保留至少第一个

# 5. 应用 mask
filtered_logits[indices_to_remove] = -inf
```

### 内存优化

- 使用 `torch.no_grad()` 禁用梯度计算
- 自动截断超长序列: `input_ids[:, -context_length:]`
- 单序列生成避免不必要的批处理开销

## 📚 文档

| 文档 | 内容 |
|------|------|
| `docs/generation_cn.md` | 解码策略理论（贪心、温度、Top-k、Top-p）|
| `docs/generation_usage_cn.md` | 详细使用教程、参数调优、FAQ |

## 🎯 与任务要求对照

根据作业文档 `cs336_spring2025_assignment1_basics.txt` (第 1845-1857 行)：

> Deliverable: Implement a function to decode from your language model. We recommend that you support the following features:
> 
> • Generate completions for a user-provided prompt (i.e., take in some x1...t and sample a completion until you hit an <|endoftext|> token).
> • Allow the user to control the maximum number of generated tokens.
> • Given a desired temperature value, apply softmax temperature scaling to the predicted next-word distributions before sampling.
> • Top-p sampling (Holtzman et al., 2020; also referred to as nucleus sampling), given a user-specified threshold value.

**全部完成 ✅**

## 🔗 参考

- Holtzman et al. (2020). "The Curious Case of Neural Text Degeneration"
- 任务文档: `cs336_spring2025_assignment1_basics.txt` (第 1788-1858 行)

---

**作者**: AI Assistant  
**日期**: 2025-11-09  
**测试状态**: ✅ 全部通过

