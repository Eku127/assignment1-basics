#!/usr/bin/env python3
"""
BPE学习框架 - 包含空白函数和测试用例
"""

from collections import Counter
from typing import List, Dict, Tuple
import regex as regex_mod
import re


def simple_pretokenize(text: str) -> List[str]:
    """
    简单的预分词函数
    
    任务：将文本分割成词和标点符号
    提示：
    1. 可以先用空格分割
    2. 然后处理标点符号（如逗号、句号、感叹号等）
    3. 返回一个词列表
    
    输入：文本字符串
    输出：预分词后的词列表
    
    示例：
    simple_pretokenize("hello world!") -> ["hello", "world", "!"]
    """
    # TODO: 在这里实现你的预分词逻辑

    words = text.split()
    result = []
    for word in words:
        # 处理标点符号
        # 将词和标点符号分别添加到结果中
        parts = re.findall(r'\w+|[^\w\s]', word)
        result.extend(parts)
    return result


def text_to_byte_sequences(pretokens: List[str]) -> List[List[int]]:
    """
    将预分词结果转换为字节序列
    
    任务：将每个词转换为UTF-8字节序列
    提示：
    1. 使用 word.encode('utf-8') 将字符串转换为字节
    2. 使用 list() 将字节对象转换为整数列表
    
    输入：预分词后的词列表
    输出：字节序列列表
    
    示例：
    text_to_byte_sequences(["hello", "!"]) -> [[104, 101, 108, 108, 111], [33]]
    """
    # TODO: 在这里实现你的字节转换逻辑
    
    result = []
    for word in pretokens:
        bytes_data = word.encode('utf-8')
        print(bytes_data)
        byte_list = list(bytes_data)
        result.append(byte_list)
    return result


def count_byte_pairs(byte_sequences: List[List[int]]) -> Counter:
    """
    统计字节对频率
    
    任务：统计所有相邻字节对的频率
    提示：
    1. 遍历每个字节序列
    2. 对于每个序列，统计相邻的字节对
    3. 使用 Counter 来统计频率
    
    输入：字节序列列表
    输出：字节对频率计数器
    
    示例：
    count_byte_pairs([[104, 101, 108], [104, 101, 111]]) -> Counter({(104, 101): 2, (101, 108): 1, (101, 111): 1})
    """
    # TODO: 在这里实现你的字节对统计逻辑
    
    results = Counter()

    for sequence in byte_sequences:
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i+1])
            results[pair] += 1

    return results


def find_most_frequent_pair(pair_counts: Counter) -> Tuple[int, int]:
    """
    找到最频繁的字节对
    
    任务：从字节对计数器中找到频率最高的字节对
    提示：
    1. 使用 most_common(1) 方法
    2. 返回字节对元组
    
    输入：字节对频率计数器
    输出：最频繁的字节对
    
    示例：
    find_most_frequent_pair(Counter({(104, 101): 3, (101, 108): 2})) -> (104, 101)
    """
    # TODO: 在这里实现你的查找逻辑

    if not pair_counts:
        return None

    most_common = pair_counts.most_common(1)
    return most_common[0][0]


    


def merge_byte_pair(byte_sequences: List[List[int]], pair: Tuple[int, int], new_token_id: int) -> List[List[int]]:
    """
    合并指定的字节对
    
    任务：在所有字节序列中查找并合并指定的字节对
    提示：
    1. 遍历每个字节序列
    2. 查找相邻的字节对
    3. 将匹配的字节对替换为新token
    4. 返回更新后的序列列表
    
    输入：字节序列列表，要合并的字节对，新token的ID
    输出：更新后的字节序列列表
    
    示例：
    merge_byte_pair([[104, 101, 108]], (104, 101), 256) -> [[256, 108]]
    """
    # TODO: 在这里实现你的合并逻辑
    
    
    result = []
    for seq in byte_sequences:

        new_seq = []
        i =0
        while i < len(seq):
            if (i < len(seq) -1  and seq[i] == pair[0] and seq[i+1] == pair[1]):
                new_seq.append(new_token_id)
                i+=2
            else:
                new_seq.append(seq[i])
                i+=1
        result.append(new_seq)

    return result




def simple_bpe_train(text: str, num_merges: int = 5) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    简单的BPE训练函数
    
    任务：实现完整的BPE训练流程
    提示：
    1. 预分词
    2. 转换为字节序列
    3. 初始化词汇表（0-255字节 + 特殊token）
    4. 循环合并最频繁的字节对
    5. 返回词汇表和合并规则
    
    输入：训练文本，合并次数
    输出：词汇表和合并规则
    """
    # TODO: 在这里实现你的BPE训练逻辑
    
    pretokens = simple_pretokenize(text)
    byte_sequences = text_to_byte_sequences(pretokens)
    vocab = {i: bytes([i]) for i in range(256)}
    print(vocab)
    merges = []
    next_token_id = 256
    for _ in range(num_merges):
        pair_counts = count_byte_pairs(byte_sequences)
        most_frequent = find_most_frequent_pair(pair_counts)
        byte_sequences = merge_byte_pair(byte_sequences, most_frequent, next_token_id)
        vocab[next_token_id] = vocab[most_frequent[0]] + vocab[most_frequent[1]]
        merges.append((vocab[most_frequent[0]], vocab[most_frequent[1]]))
        next_token_id += 1

    print(vocab)
    return vocab, merges


def test_bpe_functions():
    """
    测试BPE函数的正确性
    """
    print("=== BPE函数测试 ===\n")
    
    # 测试数据
    test_text = "hello world hello there"
    print(f"测试文本: '{test_text}'\n")
    
    # 测试1：预分词
    print("测试1：预分词")
    pretokens = simple_pretokenize(test_text)
    print(f"预分词结果: {pretokens}")
    expected = ["hello", "world", "hello", "there"]
    print(f"期望结果: {expected}")
    print(f"测试通过: {pretokens == expected}\n")
    
    # 测试2：字节转换
    print("测试2：字节转换")
    byte_sequences = text_to_byte_sequences(pretokens)
    print(f"字节序列: {byte_sequences}")
    expected = [[104, 101, 108, 108, 111], [119, 111, 114, 108, 100], 
                [104, 101, 108, 108, 111], [116, 104, 101, 114, 101]]
    print(f"期望结果: {expected}")
    print(f"测试通过: {byte_sequences == expected}\n")
    
    # 测试3：字节对统计
    print("测试3：字节对统计")
    pair_counts = count_byte_pairs(byte_sequences)
    print(f"字节对频率: {dict(pair_counts.most_common(4))}")
    expected_pairs = {(104, 101): 3, (101, 108): 2, (108, 108): 2, (108, 111): 2}
    print(f"期望的主要字节对: {expected_pairs}")
    print(f"测试通过: {all(pair_counts[pair] == count for pair, count in expected_pairs.items())}\n")
    
    # 测试4：找最频繁的字节对
    print("测试4：找最频繁的字节对")
    most_frequent = find_most_frequent_pair(pair_counts)
    print(f"最频繁的字节对: {most_frequent}")
    print(f"测试通过: {most_frequent in [(104, 101), (101, 108), (108, 108), (108, 111)]}\n")
    
    # 测试5：合并字节对
    print("测试5：合并字节对")
    if most_frequent:
        print(most_frequent)
        merged_sequences = merge_byte_pair(byte_sequences, most_frequent, 256)
        print(f"合并后的序列: {merged_sequences}")
        print("测试通过: 检查是否包含新token 256\n")
    
    # 测试6：完整BPE训练
    print("测试6：完整BPE训练")
    vocab, merges = simple_bpe_train(test_text, num_merges=3)
    print(f"词汇表大小: {len(vocab)}")
    print(f"合并规则数量: {len(merges)}")
    print(f"前几个合并规则: {merges[:3]}")
    print("测试通过: 检查词汇表是否包含原始字节和新token\n")


if __name__ == "__main__":
    test_bpe_functions()
