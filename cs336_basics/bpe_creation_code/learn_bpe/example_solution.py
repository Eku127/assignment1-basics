#!/usr/bin/env python3
"""
BPE学习框架 - 示例解决方案（供参考）
"""

from collections import Counter
from typing import List, Dict, Tuple
import re


def simple_pretokenize(text: str) -> List[str]:
    """
    简单的预分词函数 - 示例解决方案
    """
    # 方法1：简单按空格分割，然后处理标点符号
    words = text.split()
    result = []
    
    for word in words:
        # 分离词和标点符号
        # 使用正则表达式找到词和标点符号
        parts = re.findall(r'\w+|[^\w\s]', word)
        result.extend(parts)
    
    return result


def text_to_byte_sequences(pretokens: List[str]) -> List[List[int]]:
    """
    将预分词结果转换为字节序列 - 示例解决方案
    """
    result = []
    for word in pretokens:
        # 将字符串转换为UTF-8字节，再转换为整数列表
        bytes_data = word.encode('utf-8')
        byte_list = list(bytes_data)
        result.append(byte_list)
    return result


def count_byte_pairs(byte_sequences: List[List[int]]) -> Counter:
    """
    统计字节对频率 - 示例解决方案
    """
    pair_counts = Counter()
    
    for seq in byte_sequences:
        # 遍历序列中的相邻字节对
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            pair_counts[pair] += 1
    
    return pair_counts


def find_most_frequent_pair(pair_counts: Counter) -> Tuple[int, int]:
    """
    找到最频繁的字节对 - 示例解决方案
    """
    if not pair_counts:
        return None
    
    # 使用 most_common(1) 找到频率最高的字节对
    most_common = pair_counts.most_common(1)[0]
    return most_common[0]


def merge_byte_pair(byte_sequences: List[List[int]], pair: Tuple[int, int], new_token_id: int) -> List[List[int]]:
    """
    合并指定的字节对 - 示例解决方案
    """
    result = []
    
    for seq in byte_sequences:
        new_seq = []
        i = 0
        
        while i < len(seq):
            # 检查当前位置是否匹配要合并的字节对
            if (i < len(seq) - 1 and 
                seq[i] == pair[0] and 
                seq[i+1] == pair[1]):
                # 找到匹配，添加新token
                new_seq.append(new_token_id)
                i += 2  # 跳过已处理的字节
            else:
                # 没有匹配，添加原字节
                new_seq.append(seq[i])
                i += 1
        
        result.append(new_seq)
    
    return result


def simple_bpe_train(text: str, num_merges: int = 5) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    简单的BPE训练函数 - 示例解决方案
    """
    # 1. 预分词
    pretokens = simple_pretokenize(text)
    
    # 2. 转换为字节序列
    byte_sequences = text_to_byte_sequences(pretokens)
    
    # 3. 初始化词汇表（0-255字节）
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    next_token_id = 256
    
    # 4. 循环合并最频繁的字节对
    for _ in range(num_merges):
        # 统计字节对频率
        pair_counts = count_byte_pairs(byte_sequences)
        if not pair_counts:
            break
            
        # 找到最频繁的字节对
        most_frequent = find_most_frequent_pair(pair_counts)
        if not most_frequent:
            break
        
        # 合并字节对
        byte_sequences = merge_byte_pair(byte_sequences, most_frequent, next_token_id)
        
        # 更新词汇表和合并规则
        vocab[next_token_id] = vocab[most_frequent[0]] + vocab[most_frequent[1]]
        merges.append((vocab[most_frequent[0]], vocab[most_frequent[1]]))
        next_token_id += 1
    
    return vocab, merges


def demo_solution():
    """
    演示解决方案
    """
    print("=== BPE解决方案演示 ===\n")
    
    # 测试数据
    test_text = "hello world hello there"
    print(f"测试文本: '{test_text}'\n")
    
    # 运行BPE训练
    vocab, merges = simple_bpe_train(test_text, num_merges=3)
    
    print("训练结果:")
    print(f"词汇表大小: {len(vocab)}")
    print(f"合并规则数量: {len(merges)}")
    print()
    
    print("合并规则:")
    for i, (left, right) in enumerate(merges):
        print(f"  {i+1}. {left} + {right} -> {left + right}")
    print()
    
    print("学习到的词汇表（非字节部分）:")
    learned_vocab = {k: v for k, v in vocab.items() if k >= 256}
    for k, v in sorted(learned_vocab.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    demo_solution()
