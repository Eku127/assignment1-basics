# BPE (Byte-Pair Encoding) Tokenizer

## 📋 概述

本模块实现了基于字节的BPE (Byte-Pair Encoding) tokenizer，遵循CS336 Assignment 1的规范。BPE是一种subword tokenization方法，在词级和字符级tokenization之间取得平衡。

## 📚 背景知识

### 为什么需要Tokenization？

神经网络语言模型需要将文本转换为数字序列才能处理。Tokenization就是这个转换过程：
- **输入**: `"Hello world"`
- **输出**: `[104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]`

### Tokenization方法对比

| 方法 | 词汇表大小 | 序列长度 | OOV问题 | 示例 |
|------|-----------|---------|---------|------|
| **Word-level** | 很大(~100K) | 短 | 有 | `["Hello", "world"]` |
| **Character-level** | 小(~256) | 很长 | 无 | `['H','e','l','l','o',' ','w','o','r','l','d']` |
| **Byte-level** | 256 | 非常长 | 无 | `[72, 101, 108, ...]` |
| **BPE (Subword)** | 中等(~32K) | 适中 | 无 | `["Hello", " world"]` |

### BPE的优势

✅ **无OOV问题**: 通过字节级回退，可以表示任何Unicode文本  
✅ **平衡效率**: 常见词作为单个token，罕见词拆分为subwords  
✅ **可扩展**: 词汇表大小可配置，适应不同需求

---

## 🏗️ 模块结构

```
bpe/
├── __init__.py              # 模块入口
├── training.py              # BPE训练（学习merges）
├── tokenizer.py             # Tokenizer类（编码/解码）
├── utils.py                 # 工具函数（预分词等）
└── README.md                # 本文档
```

---

## 🎯 核心组件

### 1️⃣ BPE训练 (training.py)

**功能**: 从文本语料中学习BPE合并规则

**函数**: `train_bpe(input_path, vocab_size, special_tokens)`

**算法流程**:
```
1. 初始化词汇表: 256个字节 + 特殊tokens
2. 预分词: 使用GPT-2正则表达式
3. 迭代合并:
   a. 统计所有相邻字节对的频率
   b. 选择最频繁的对（相同频率时按字典序）
   c. 合并该对，添加到词汇表
   d. 重复直到达到目标词汇表大小
4. 返回词汇表和合并规则列表
```

**示例**:
```python
from cs336_basics.bpe import train_bpe

# 训练BPE tokenizer
vocab, merges = train_bpe(
    input_path="corpus.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"]
)

print(f"Vocabulary size: {len(vocab)}")
print(f"Number of merges: {len(merges)}")

# 查看学到的tokens
for merge_idx, (a, b) in enumerate(merges[:5]):
    print(f"Merge {merge_idx}: {a!r} + {b!r} -> {a+b!r}")
```

**输出示例**:
```
Vocabulary size: 10000
Number of merges: 9743
Merge 0: b't' + b'h' -> b'th'
Merge 1: b'e' + b'r' -> b'er'
Merge 2: b'in' + b'g' -> b'ing'
...
```

### 2️⃣ Tokenizer类 (tokenizer.py)

**功能**: 使用学到的vocab和merges编码/解码文本

**主要方法**:

#### 构造函数
```python
from cs336_basics.bpe import Tokenizer

# 方法1: 直接构造
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

# 方法2: 从文件加载
tokenizer = Tokenizer.from_files(
    vocab_filepath="vocab.json",
    merges_filepath="merges.txt",
    special_tokens=["<|endoftext|>"]
)
```

#### encode() - 编码
```python
text = "Hello world"
token_ids = tokenizer.encode(text)
print(token_ids)
# [9906, 995]  # 示例输出
```

#### decode() - 解码
```python
token_ids = [9906, 995]
text = tokenizer.decode(token_ids)
print(text)
# "Hello world"
```

#### encode_iterable() - 流式编码
```python
# 内存高效的大文件tokenization
with open("large_corpus.txt", "r") as f:
    for token_id in tokenizer.encode_iterable(f):
        # 逐个处理token，不需要一次加载整个文件
        process(token_id)
```

### 3️⃣ 工具函数 (utils.py)

**核心功能**:

#### 预分词 (Pre-tokenization)
使用GPT-2风格的正则表达式分割文本：
```python
from cs336_basics.bpe.utils import pretokenize_string

text = "some text that i'll pre-tokenize"
tokens = pretokenize_string(text)
print(tokens)
# ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

**正则表达式模式**:
```python
GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

匹配规则：
- 缩写: `'s`, `'t`, `'m`, `'ll`, `'ve`, `'re`
- 字母: `?\p{L}+` (可选前导空格 + Unicode字母)
- 数字: `?\p{N}+` (可选前导空格 + Unicode数字)
- 其他字符: `?[^\s\p{L}\p{N}]+`
- 空格: `\s+(?!\S)|\s+`

#### 特殊Token处理
```python
from cs336_basics.bpe.utils import split_on_special_tokens

text = "Doc1<|endoftext|>Doc2<|endoftext|>Doc3"
segments = split_on_special_tokens(text, ["<|endoftext|>"])
print(segments)
# ['Doc1', 'Doc2', 'Doc3']
```

---

## 📖 使用指南

### 完整工作流程

#### Step 1: 训练BPE
```python
from cs336_basics.bpe import train_bpe
import json

# 训练tokenizer
vocab, merges = train_bpe(
    input_path="/data/TinyStories_train.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"]
)

# 保存vocab (JSON格式)
with open("vocab.json", "w") as f:
    # 将bytes转为list[int]以便JSON序列化
    json_vocab = {str(k): list(v) for k, v in vocab.items()}
    json.dump(json_vocab, f)

# 保存merges (文本格式)
with open("merges.txt", "w") as f:
    for a, b in merges:
        f.write(f"{a!r} {b!r}\n")
```

#### Step 2: 加载并使用
```python
from cs336_basics.bpe import Tokenizer

# 加载tokenizer
tokenizer = Tokenizer.from_files(
    vocab_filepath="vocab.json",
    merges_filepath="merges.txt",
    special_tokens=["<|endoftext|>"]
)

# 编码文本
text = "Once upon a time, there was a little girl named Lily."
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")
print(f"Number of tokens: {len(token_ids)}")

# 解码回文本
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")
assert text == decoded_text  # 验证往返一致性
```

#### Step 3: Tokenize整个数据集
```python
import numpy as np

# 内存映射方式处理大文件
def tokenize_file(input_path, output_path, tokenizer):
    """将文本文件tokenize并保存为numpy数组"""
    token_ids = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            ids = tokenizer.encode(line)
            token_ids.extend(ids)
    
    # 保存为uint16数组（词汇表<65536时）
    arr = np.array(token_ids, dtype=np.uint16)
    np.save(output_path, arr)
    print(f"Saved {len(arr):,} tokens to {output_path}")

# Tokenize训练集和验证集
tokenize_file("train.txt", "train_tokens.npy", tokenizer)
tokenize_file("val.txt", "val_tokens.npy", tokenizer)
```

---

## 🧪 测试

### 运行测试

```bash
# 测试BPE训练
uv run pytest tests/test_train_bpe.py -v

# 测试Tokenizer类
uv run pytest tests/test_tokenizer.py -v

# 测试特定功能
uv run pytest tests/test_tokenizer.py::test_encode -v
uv run pytest tests/test_tokenizer.py::test_decode -v
uv run pytest tests/test_tokenizer.py::test_special_tokens -v
```

### 测试覆盖

| 测试文件 | 测试内容 |
|---------|---------|
| `test_train_bpe.py` | BPE训练算法、特殊token处理、tie-breaking |
| `test_tokenizer.py` | 编码、解码、特殊token、streaming、往返一致性 |

---

## 📊 性能优化

### BPE训练优化

**问题**: 朴素实现每次merge都要扫描整个语料，复杂度O(n²)

**解决方案**: 增量更新
1. **维护数据结构**:
   - `total_pair_counts`: 全局pair计数
   - `pair_to_words`: pair到包含它的words的反向索引
   - `word_pair_counters`: 每个word的pair计数器

2. **增量更新**:
   - Merge时只更新受影响的words
   - 更新相关的pair计数
   - 复杂度降至O(n)

**加速效果**:
- TinyStories (2.1M文档): ~2分钟 (多进程)
- OpenWebText (8M文档): ~30分钟

### Tokenization优化

**内存优化**:
```python
# ❌ 不好: 一次加载整个文件
with open("large_file.txt") as f:
    text = f.read()
    tokens = tokenizer.encode(text)

# ✅ 好: 流式处理
with open("large_file.txt") as f:
    for token_id in tokenizer.encode_iterable(f):
        save_token(token_id)
```

---

## 🎓 实现细节

### 编码算法

```
输入: "hello world"
输出: [token_ids]

步骤:
1. 处理特殊tokens
   - 查找文本中的特殊tokens
   - 分割并保留特殊token边界

2. 预分词
   - 使用GPT-2正则表达式
   - "hello world" -> ["hello", " world"]

3. 转换为字节
   - "hello" -> [b'h', b'e', b'l', b'l', b'o']
   - " world" -> [b' ', b'w', b'o', b'r', b'l', b'd']

4. 应用BPE merges（按训练时顺序）
   - 扫描: [b'h', b'e', b'l', b'l', b'o']
   - 发现merge (b'h', b'e') -> b'he'
   - 结果: [b'he', b'l', b'l', b'o']
   - 继续应用剩余merges...

5. 转换为token IDs
   - 使用bytes_to_id映射
   - [b'hello'] -> [9906]
```

### 解码算法

```
输入: [9906, 995]
输出: "hello world"

步骤:
1. 查找vocabulary
   - 9906 -> b'hello'
   - 995 -> b' world'

2. 连接bytes
   - b'hello' + b' world' = b'hello world'

3. UTF-8解码
   - b'hello world' -> "hello world"
   - 无效字节用U+FFFD替换
```

---

## ⚠️ 常见问题

### Q1: 为什么使用字节级BPE？

**A**: 三个原因：
1. **无OOV**: 任何Unicode文本都可以表示为UTF-8字节
2. **小初始词汇表**: 只需256个基础tokens
3. **语言无关**: 不需要针对特定语言调整

### Q2: 为什么需要预分词？

**A**: 
- **效率**: 避免跨词边界的无意义merges
- **语义**: 保留重要的词边界信息
- **标点**: 合理处理标点符号

### Q3: 特殊tokens如何处理？

**A**:
```python
# 特殊tokens不参与BPE merges
text = "Hello<|endoftext|>World"

# 编码过程:
# 1. 分割: ["Hello", "<|endoftext|>", "World"]
# 2. 只对 "Hello" 和 "World" 应用BPE
# 3. <|endoftext|> 保持为单个token
```

### Q4: 如何选择vocab_size？

| 应用场景 | 推荐大小 | 说明 |
|---------|---------|------|
| 小模型/简单数据 | 10K | 如TinyStories |
| 中型模型 | 32K | 如GPT-2 |
| 大型模型 | 50-100K | 如GPT-3 |

权衡：
- **太小**: 序列太长，训练慢
- **太大**: 罕见词得不到充分训练

---

## 📚 参考资料

### 论文
1. **Sennrich et al. (2016)** - "Neural Machine Translation of Rare Words with Subword Units"
   - BPE用于NMT的原始论文

2. **Wang et al. (2019)** - "Neural Machine Translation with Byte-Level Subwords"
   - 字节级BPE

3. **Radford et al. (2019)** - "Language Models are Unsupervised Multitask Learners"
   - GPT-2使用的BPE + 预分词

### 相关代码
- OpenAI tiktoken: https://github.com/openai/tiktoken
- HuggingFace tokenizers: https://github.com/huggingface/tokenizers

---

## 📄 文档信息

- **课程**: CS336 - Language Modeling from Scratch
- **作业**: Assignment 1 - Basics (Section 2: BPE Tokenizer)
- **组件**: BPE Training & Tokenization
- **语言**: Python 3.10+
- **依赖**: `regex` (for GPT-2 pattern), `numpy` (optional, for saving tokens)

---

*Built for CS336 Spring 2025 - Stanford University*

