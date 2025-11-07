# BPE学习框架

这个目录包含了学习BPE（Byte-Pair Encoding）的完整框架，帮助你理解BPE的工作原理和实现细节。

## 文件结构

- `bpe_framework.py` - 主要学习文件，包含空白函数和测试用例
- `step_by_step_guide.py` - 分步骤实现指南
- `example_solution.py` - 示例解决方案（供参考）
- `README.md` - 本文件

## 学习步骤

### 1. 理解BPE概念
首先运行指南文件来理解BPE的基本概念：
```bash
cd /data4/jjj/workspace/assignment1-basics
uv run python cs336_basics/learn_bpe/step_by_step_guide.py
```

### 2. 实现BPE函数
在 `bpe_framework.py` 中实现以下函数：

1. `simple_pretokenize()` - 预分词函数
2. `text_to_byte_sequences()` - 字节转换函数
3. `count_byte_pairs()` - 字节对统计函数
4. `find_most_frequent_pair()` - 找最频繁字节对
5. `merge_byte_pair()` - 合并字节对函数
6. `simple_bpe_train()` - 完整BPE训练函数

### 3. 测试你的实现
运行测试来验证你的实现：
```bash
uv run python cs336_basics/learn_bpe/bpe_framework.py
```

### 4. 参考解决方案
如果遇到困难，可以查看 `example_solution.py` 中的示例解决方案。

## 测试用例

框架包含以下测试用例：
- 预分词测试
- 字节转换测试
- 字节对统计测试
- 合并函数测试
- 完整BPE训练测试

## 学习目标

通过这个框架，你将学会：
1. 如何将文本分割成词和标点符号
2. 如何将字符串转换为字节序列
3. 如何统计字节对频率
4. 如何合并最频繁的字节对
5. 如何实现完整的BPE训练流程

## 提示

- 按顺序实现函数，每个函数都有详细的提示
- 先理解每个函数的目标和输入输出
- 使用测试用例来验证你的实现
- 如果测试失败，仔细检查你的逻辑
- 可以参考示例解决方案，但建议先自己尝试
