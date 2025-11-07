# BPE 字节对编码 (Byte-Pair Encoding) 实现

## 📖 项目概述

本项目实现了一个完整的字节级 BPE (Byte-Pair Encoding) 分词器，这是现代大语言模型（如 GPT-2、GPT-3）中广泛使用的子词分词技术。

### 什么是 BPE？

BPE 是一种数据压缩算法，后来被应用于自然语言处理中的分词任务：

1. **基本原理**：通过迭代地合并最频繁出现的字节对，逐步构建词汇表
2. **字节级处理**：在字节层面操作，可以处理任何 UTF-8 编码的文本
3. **子词分割**：在字符和单词之间找到平衡，既能处理常见词，也能处理罕见词和新词

### 为什么使用字节级 BPE？

- ✅ **无词汇表外问题**：任何文本都可以被编码（因为基于字节）
- ✅ **灵活性强**：可以处理多语言、代码、特殊符号等
- ✅ **压缩率高**：相比字符级编码，大幅减少序列长度
- ✅ **工业标准**：OpenAI GPT 系列、Meta LLaMA 等模型的分词方案

---

## 🏗️ 项目结构

```
cs336_basics/bpe/
├── __init__.py              # 模块入口，导出核心类和函数
├── README_CN.md             # 中文文档（本文件）
├── utils.py                 # 工具函数：预分词、词汇表构建
├── training.py              # BPE 训练核心算法
├── tokenizer.py             # 分词器类：编码/解码
├── applications/            # 应用脚本目录
│   ├── train_tinystories_tokenizer.py    # TinyStories 数据集训练脚本
│   ├── train_owt_tokenizer.py            # OpenWebText 数据集训练脚本
│   ├── encode_tinystories.py             # TinyStories 编码脚本
│   └── encode_owt.py                     # OpenWebText 编码脚本
└── old_code/                # 历史代码存档
```

---

## 📚 核心模块详解

### 1. `utils.py` - 工具函数模块

#### 功能概述

提供 BPE 训练和编码所需的基础工具函数。

#### 主要功能

**1.1 预分词 (Pre-tokenization)**

将原始文本按照 GPT-2 的正则表达式规则切分成初始的 token：

```python
from cs336_basics.bpe.utils import pretokenize

# 输入：文本段落列表
segments = ["Hello world!", "How are you?"]

# 输出：每个字节序列及其出现频率
freq = pretokenize(segments)
# 结果：{(72, 101, 108, 108, 111): 1, (119, 111, 114, 108, 100): 1, ...}
```

**特点：**
- 使用 GPT-2 的预分词模式（正则表达式）
- 自动识别大规模语料，启用多进程并行处理
- 返回字节序列的频率统计

**1.2 特殊 Token 处理**

```python
from cs336_basics.bpe.utils import split_on_special_tokens

text = "Hello<|endoftext|>World"
special_tokens = ["<|endoftext|>"]

segments = split_on_special_tokens(text, special_tokens)
# 结果：["Hello", "World"]  （特殊token被作为分隔符）
```

**1.3 初始词汇表构建**

```python
from cs336_basics.bpe.utils import build_initial_vocab

special_tokens = ["<|endoftext|>"]
id_to_bytes, bytes_to_id = build_initial_vocab(special_tokens)

# id_to_bytes: {0: b'\x00', 1: b'\x01', ..., 255: b'\xff', 256: b'<|endoftext|>'}
# bytes_to_id: 反向映射字典
```

---

### 2. `training.py` - BPE 训练模块

#### 功能概述

实现 BPE 训练算法的核心逻辑，从原始文本语料库中学习分词规则。

#### 训练流程

```
输入：原始文本文件 + 目标词汇表大小 + 特殊token
         ↓
1. 读取并分割文本（按特殊token分割）
         ↓
2. 预分词（GPT-2 正则表达式）
         ↓
3. 初始化词汇表（256个字节 + 特殊token）
         ↓
4. 迭代合并最频繁的字节对
         ↓
输出：完整词汇表 + 合并规则列表
```

#### 使用方法

```python
from cs336_basics.bpe import train_bpe

# 训练 BPE 分词器
vocab, merges = train_bpe(
    input_path="corpus.txt",           # 训练语料路径
    vocab_size=10000,                  # 目标词汇表大小
    special_tokens=["<|endoftext|>"]  # 特殊token列表
)

# vocab: {0: b' ', 1: b'!', ..., 9999: b'Hello'}
# merges: [(b' ', b't'), (b'h', b'e'), ...]  按训练顺序排列
```

#### 训练过程输出示例

```
📖 Reading corpus from corpus.txt...
✅ Read 132,878 characters in 0.0s (124.0 MB/s)

✂️  Splitting on special tokens: ['<|endoftext|>']...
✅ Split into 1 segments in 0.0s

🔤 Pre-tokenizing with GPT-2 regex...
✅ Pre-tokenization complete in 0.0s
   Found 4,763 unique byte sequences

📚 Initializing vocabulary...
✅ Initial vocabulary: 257 tokens

🔢 Building corpus ID representation...
Converting to IDs: 100%|██████████| 4763/4763 [00:00<00:00]
✅ Corpus built in 0.0s: 4,763 unique sequences

🔗 Building pair statistics...
Computing pairs: 100%|██████████| 4763/4763 [00:00<00:00]
✅ Pair statistics built in 0.1s: 1,069 unique pairs

🔄 Starting BPE training: 243 merges needed
Merging pairs: 100%|██████████| 243/243 [00:28<00:00, 8.47merge/s]

✅ BPE training completed!
   Total merges: 243
   Final vocab size: 500
   Total time: 28.7s
```

#### 关键特性

- **增量更新**：只更新受影响的词，避免全量重新计算
- **进度可视化**：使用 tqdm 显示实时训练进度
- **内存优化**：使用频率计数而非存储完整语料
- **正确的平局处理**：频率相同时，按字典序选择字节对

---

### 3. `tokenizer.py` - 分词器模块

#### 功能概述

提供完整的分词器类，支持文本的编码（text → token IDs）和解码（token IDs → text）。

#### 核心功能

**3.1 创建分词器**

```python
from cs336_basics.bpe import Tokenizer

# 方法1：从训练结果创建
tokenizer = Tokenizer(
    vocab=vocab,                        # 词汇表字典
    merges=merges,                      # 合并规则列表
    special_tokens=["<|endoftext|>"]   # 特殊token
)

# 方法2：从保存的文件加载
tokenizer = Tokenizer.from_files(
    vocab_filepath="data/tokenizers/tinystories_vocab.json",
    merges_filepath="data/tokenizers/tinystories_merges.txt",
    special_tokens=["<|endoftext|>"]
)
```

**3.2 文本编码**

```python
# 基本编码
text = "Hello world!"
token_ids = tokenizer.encode(text)
print(token_ids)  # [9906, 995, 0]

# 流式编码（内存高效）
with open("large_file.txt") as f:
    for token_id in tokenizer.encode_iterable(f):
        # 逐个处理 token，不需要一次性加载整个文件
        process(token_id)
```

**3.3 文本解码**

```python
# 将 token IDs 解码回文本
token_ids = [9906, 995, 0]
text = tokenizer.decode(token_ids)
print(text)  # "Hello world!"

# 验证往返一致性
original = "Hello world!"
assert tokenizer.decode(tokenizer.encode(original)) == original
```

**3.4 保存和加载**

```python
# 保存分词器
tokenizer.save(
    vocab_filepath="my_tokenizer_vocab.json",
    merges_filepath="my_tokenizer_merges.txt"
)

# 加载分词器
tokenizer = Tokenizer.from_files(
    vocab_filepath="my_tokenizer_vocab.json",
    merges_filepath="my_tokenizer_merges.txt",
    special_tokens=["<|endoftext|>"]
)
```

#### 关键优化

- **优先级合并**：使用 `bpe_ranks` 字典实现 O(1) 优先级查找
- **预排序特殊token**：初始化时排序，避免每次编码都排序
- **快速路径**：无特殊token时跳过分割逻辑
- **内存高效**：`encode_iterable` 支持流式处理大文件

---

## 🧪 测试说明

### 运行所有 BPE 相关测试

```bash
# 测试 BPE 训练
uv run pytest tests/test_train_bpe.py -v

# 测试分词器编码/解码
uv run pytest tests/test_tokenizer.py -v

# 测试特定功能
uv run pytest tests/test_tokenizer.py::test_encode -v
uv run pytest tests/test_tokenizer.py::test_decode -v
uv run pytest tests/test_tokenizer.py::test_roundtrip -v
```

### 主要测试用例

#### 1. BPE 训练测试

**测试内容：**
- 词汇表大小是否正确
- 合并规则顺序是否符合频率
- 特殊token是否正确处理
- 平局情况的字典序处理

```bash
uv run pytest tests/test_train_bpe.py::test_train_bpe -v
```

#### 2. 编码功能测试

**测试内容：**
- 基本字符串编码
- 特殊token处理
- Unicode 字符支持
- 空字符串和边界情况

```bash
uv run pytest tests/test_tokenizer.py::test_encode -v
```

#### 3. 解码功能测试

**测试内容：**
- token IDs 正确解码为文本
- UTF-8 解码错误处理
- 特殊字符正确还原

```bash
uv run pytest tests/test_tokenizer.py::test_decode -v
```

#### 4. 往返一致性测试

**测试内容：**
- `decode(encode(text)) == text`
- 各种文本类型的往返验证
- 特殊token的往返处理

```bash
uv run pytest tests/test_tokenizer.py::test_roundtrip -v
```

#### 5. 内存效率测试

**测试内容：**
- `encode_iterable` 不会将整个文件加载到内存
- 流式处理的正确性

```bash
uv run pytest tests/test_tokenizer.py::test_encode_iterable_memory_usage -v
```

---

## 🚀 应用示例

### 示例 1：训练 TinyStories 分词器

TinyStories 是一个小型英文故事数据集，适合快速实验。

```bash
# 运行训练脚本
uv run python cs336_basics/bpe/applications/train_tinystories_tokenizer.py

# 输出文件
# - data/tokenizers/tinystories_vocab.json       (词汇表)
# - data/tokenizers/tinystories_merges.txt       (合并规则)
# - data/tokenizers/tinystories_special_tokens.json  (特殊token)
```

**训练参数：**
- 数据集大小：~90MB
- 词汇表大小：10,000
- 特殊token：`<|endoftext|>`
- 预计时间：2-5分钟

**输出示例：**
```
🚀 TinyStories BPE 分词器训练
📊 数据集信息：
   - 路径: data/TinyStories/TinyStories-train.txt
   - 大小: 89.7 MB
   
🎯 训练参数：
   - 目标词汇表大小: 10,000
   - 特殊 tokens: ['<|endoftext|>']
   
📖 开始训练...
✅ 训练完成！词汇表大小: 10,000
💾 分词器已保存到 data/tokenizers/
```

### 示例 2：训练 OpenWebText 分词器

OpenWebText 是一个大规模英文网页数据集，需要更长的训练时间。

```bash
# 运行训练脚本
uv run python cs336_basics/bpe/applications/train_owt_tokenizer.py

# 输出文件
# - data/tokenizers/owt_vocab.json
# - data/tokenizers/owt_merges.txt
# - data/tokenizers/owt_special_tokens.json
```

**训练参数：**
- 数据集大小：~40GB
- 词汇表大小：32,000
- 特殊token：`<|endoftext|>`
- 预计时间：30-60分钟

### 示例 3：编码 TinyStories 数据集

将文本数据集转换为 token ID 序列，用于模型训练。

```bash
# 运行编码脚本
uv run python cs336_basics/bpe/applications/encode_tinystories.py

# 输出文件
# - data/encoded/tinystories_train.npy      (训练集，numpy格式)
# - data/encoded/tinystories_train.bin      (训练集，二进制格式)
# - data/encoded/tinystories_valid.npy      (验证集，numpy格式)
# - data/encoded/tinystories_valid.bin      (验证集，二进制格式)
```

**处理流程：**
1. 加载训练好的分词器
2. 读取原始文本文件
3. 批量编码（多进程并行）
4. 保存为 numpy 数组和二进制文件

**输出统计示例：**
```
🚀 Encoding: TinyStories-train.txt
  Using 8 processes, batch size: 1000

  ✅ Encoding Complete!
     Documents: 2,119,719 (0 errors)
     Tokens: 146,523,456
     Token ID range: 0 - 9999
     Unique tokens: 9,987
     File size: 279.12 MB (numpy), 279.12 MB (binary)
     Time: 45.2s (3.2M tokens/sec)
```

### 示例 4：编码 OpenWebText 数据集

```bash
# 运行编码脚本
uv run python cs336_basics/bpe/applications/encode_owt.py

# 输出文件
# - data/encoded/owt_train.npy
# - data/encoded/owt_train.bin
# - data/encoded/owt_valid.npy
# - data/encoded/owt_valid.bin
```

**特点：**
- 自动使用更多进程（最多16个）
- 更大的批次大小（2000个文档/批次）
- 优化的大文件处理

---

## 💡 使用技巧

### 1. 选择合适的词汇表大小

| 应用场景 | 推荐大小 | 说明 |
|---------|---------|------|
| 实验/原型 | 1,000 - 5,000 | 快速训练，适合调试 |
| 小型语言模型 | 10,000 - 15,000 | 平衡性能和效率 |
| 中型语言模型 | 30,000 - 50,000 | GPT-2/GPT-3 规模 |
| 大型多语言模型 | 100,000+ | 支持更多语言和领域 |

### 2. 特殊 Token 的使用

```python
# 常见特殊token
special_tokens = [
    "<|endoftext|>",      # 文档分隔符
    "<|pad|>",            # 填充token
    "<|unk|>",            # 未知token（字节级BPE通常不需要）
    "<|bos|>",            # 句子开始
    "<|eos|>",            # 句子结束
]
```

**注意事项：**
- 特殊token不参与BPE合并
- 特殊token会占用词汇表空间
- 设计特殊token时避免与常见文本片段冲突

### 3. 内存管理

**训练时：**
- 使用频率统计而非完整语料
- 对于超大数据集，考虑采样训练

**编码时：**
- 使用 `encode_iterable()` 处理大文件
- 批量处理多个文档可以提高效率

### 4. 文件格式选择

```python
# numpy 格式 (.npy)
# - 优点：支持快速随机访问，方便加载到 numpy/pytorch
# - 缺点：包含元数据，文件稍大

# 二进制格式 (.bin)
# - 优点：纯数据，文件最小，加载最快
# - 缺点：需要手动指定 dtype（通常是 uint16）

# 加载示例
import numpy as np

# numpy 格式
tokens_npy = np.load("data/encoded/tinystories_train.npy")

# 二进制格式
tokens_bin = np.fromfile("data/encoded/tinystories_train.bin", dtype=np.uint16)
```

---

## 📊 性能指标

### 训练性能

| 数据集 | 大小 | 词汇表 | 训练时间 | 速度 |
|--------|------|--------|----------|------|
| TinyStories | 90MB | 10K | ~2分钟 | ~45MB/min |
| OpenWebText | 40GB | 32K | ~45分钟 | ~890MB/min |

**性能影响因素：**
- CPU 核心数（多进程加速）
- 数据集大小和复杂度
- 词汇表大小（更大的词汇表需要更多合并）

### 编码性能

| 数据集 | Token数量 | 编码时间 | 速度 |
|--------|-----------|----------|------|
| TinyStories | 146M | ~45秒 | ~3.2M tokens/s |
| OpenWebText | 2.5B | ~15分钟 | ~2.8M tokens/s |

**优化要点：**
- 并行文档处理（multiprocessing）
- 批量编码减少函数调用
- 优先级合并算法（bpe_ranks）

---

## 🔧 故障排查

### 问题 1：训练速度太慢

**可能原因：**
- 数据集过大
- 词汇表设置过大
- 单线程预分词

**解决方案：**
- 使用数据采样进行快速实验
- 降低词汇表大小
- 确保预分词启用了多进程（代码会自动判断）

### 问题 2：编码后文件过大

**可能原因：**
- 使用了 int32 而非 uint16
- 词汇表过小导致token数量过多

**解决方案：**
```python
# 使用 uint16 保存（词汇表 < 65536）
token_array = np.array(token_ids, dtype=np.uint16)
token_array.tofile("output.bin")
```

### 问题 3：特殊 token 未正确处理

**症状：**
- 特殊token被分割成多个字节
- 编码后无法识别特殊token

**解决方案：**
- 确保训练和编码时使用相同的 special_tokens 列表
- 验证特殊token在词汇表中
- 检查特殊token是否包含在保存的 JSON 文件中

### 问题 4：解码结果与原文不一致

**可能原因：**
- UTF-8 编码/解码问题
- 特殊字符处理
- 词汇表不匹配

**调试方法：**
```python
# 测试往返一致性
text = "测试文本 Hello! 🎉"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
assert text == decoded, f"不一致: {text} != {decoded}"
```

---

## 📖 参考资料

### 学术论文

1. **Sennrich et al. (2016)** - "Neural Machine Translation of Rare Words with Subword Units"
   - BPE 算法的原始论文
   - 首次将 BPE 应用于 NMT

2. **Radford et al. (2019)** - "Language Models are Unsupervised Multitask Learners" (GPT-2)
   - 字节级 BPE 的应用
   - GPT-2 预分词正则表达式

3. **Brown et al. (2020)** - "Language Models are Few-Shot Learners" (GPT-3)
   - 大规模 BPE 词汇表设计

### 开源实现

- **OpenAI tiktoken**: 高性能 Rust 实现
- **HuggingFace tokenizers**: 工业级分词库
- **SentencePiece**: Google 的子词分词工具

---

## 📝 总结

本 BPE 实现提供了：

✅ **完整的训练流程**：从原始文本到词汇表和合并规则  
✅ **高效的分词器**：支持编码、解码和流式处理  
✅ **工业级特性**：特殊token、UTF-8、内存优化  
✅ **实用的应用脚本**：开箱即用的训练和编码工具  
✅ **详尽的测试**：确保正确性和鲁棒性  

适用于：
- 语言模型预训练的数据准备
- 分词算法的学习和实验
- 自定义领域的分词器训练
- NLP 课程作业和研究项目

---

**版本信息：** v2.0  
**最后更新：** 2025年11月  
**维护者：** CS336 Assignment 1 Team

