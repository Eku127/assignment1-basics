#!/usr/bin/env python3
"""
BPE学习指南 - 分步骤实现指南
"""

def step1_guide():
    """
    第一步：实现预分词函数
    """
    print("=== 第一步：实现预分词函数 ===\n")
    
    print("目标：将文本分割成词和标点符号")
    print("示例：'hello world!' -> ['hello', 'world', '!']\n")
    
    print("提示：")
    print("1. 使用 split() 方法按空格分割")
    print("2. 处理标点符号，可以用简单的字符检查")
    print("3. 或者使用正则表达式进行更精确的分割\n")
    
    print("代码框架：")
    print("""
def simple_pretokenize(text: str) -> List[str]:
    words = text.split()  # 按空格分割
    result = []
    for word in words:
        # 处理标点符号
        # 将词和标点符号分别添加到结果中
        pass
    return result
    """)
    
    print("测试用例：")
    test_cases = [
        ("hello world", ["hello", "world"]),
        ("hello world!", ["hello", "world", "!"]),
        ("hello, world.", ["hello", ",", "world", "."]),
        ("hello there", ["hello", "there"])
    ]
    
    for text, expected in test_cases:
        print(f"  '{text}' -> {expected}")


def step2_guide():
    """
    第二步：实现字节转换函数
    """
    print("\n=== 第二步：实现字节转换函数 ===\n")
    
    print("目标：将字符串转换为UTF-8字节序列")
    print("示例：'hello' -> [104, 101, 108, 108, 111]\n")
    
    print("提示：")
    print("1. 使用 word.encode('utf-8') 将字符串转换为字节")
    print("2. 使用 list() 将字节对象转换为整数列表")
    print("3. 对每个词都进行转换\n")
    
    print("代码框架：")
    print("""
def text_to_byte_sequences(pretokens: List[str]) -> List[List[int]]:
    result = []
    for word in pretokens:
        # 将字符串转换为字节，再转换为整数列表
        pass
    return result
    """)
    
    print("测试用例：")
    test_cases = [
        (["hello"], [[104, 101, 108, 108, 111]]),
        (["!"], [[33]]),
        (["hello", "!"], [[104, 101, 108, 108, 111], [33]])
    ]
    
    for pretokens, expected in test_cases:
        print(f"  {pretokens} -> {expected}")


def step3_guide():
    """
    第三步：实现字节对统计函数
    """
    print("\n=== 第三步：实现字节对统计函数 ===\n")
    
    print("目标：统计所有相邻字节对的频率")
    print("示例：[[104, 101, 108], [104, 101, 111]] -> Counter({(104, 101): 2, (101, 108): 1, (101, 111): 1})\n")
    
    print("提示：")
    print("1. 遍历每个字节序列")
    print("2. 对于每个序列，检查相邻的字节对")
    print("3. 使用 Counter 来统计频率\n")
    
    print("代码框架：")
    print("""
def count_byte_pairs(byte_sequences: List[List[int]]) -> Counter:
    pair_counts = Counter()
    for seq in byte_sequences:
        # 遍历序列中的相邻字节对
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            pair_counts[pair] += 1
    return pair_counts
    """)


def step4_guide():
    """
    第四步：实现合并函数
    """
    print("\n=== 第四步：实现合并函数 ===\n")
    
    print("目标：在所有字节序列中查找并合并指定的字节对")
    print("示例：[[104, 101, 108]], (104, 101), 256 -> [[256, 108]]\n")
    
    print("提示：")
    print("1. 遍历每个字节序列")
    print("2. 查找相邻的字节对")
    print("3. 将匹配的字节对替换为新token")
    print("4. 注意处理重叠的情况\n")
    
    print("代码框架：")
    print("""
def merge_byte_pair(byte_sequences: List[List[int]], pair: Tuple[int, int], new_token_id: int) -> List[List[int]]:
    result = []
    for seq in byte_sequences:
        new_seq = []
        i = 0
        while i < len(seq):
            # 检查当前位置是否匹配要合并的字节对
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i+1] == pair[1]:
                # 找到匹配，添加新token
                new_seq.append(new_token_id)
                i += 2  # 跳过已处理的字节
            else:
                # 没有匹配，添加原字节
                new_seq.append(seq[i])
                i += 1
        result.append(new_seq)
    return result
    """)


def step5_guide():
    """
    第五步：实现完整BPE训练
    """
    print("\n=== 第五步：实现完整BPE训练 ===\n")
    
    print("目标：实现完整的BPE训练流程")
    print("步骤：")
    print("1. 预分词")
    print("2. 转换为字节序列")
    print("3. 初始化词汇表（0-255字节）")
    print("4. 循环合并最频繁的字节对")
    print("5. 返回词汇表和合并规则\n")
    
    print("代码框架：")
    print("""
def simple_bpe_train(text: str, num_merges: int = 5) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 1. 预分词
    pretokens = simple_pretokenize(text)
    
    # 2. 转换为字节序列
    byte_sequences = text_to_byte_sequences(pretokens)
    
    # 3. 初始化词汇表
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    next_token_id = 256
    
    # 4. 循环合并
    for _ in range(num_merges):
        # 统计字节对频率
        pair_counts = count_byte_pairs(byte_sequences)
        if not pair_counts:
            break
            
        # 找到最频繁的字节对
        most_frequent = find_most_frequent_pair(pair_counts)
        
        # 合并字节对
        byte_sequences = merge_byte_pair(byte_sequences, most_frequent, next_token_id)
        
        # 更新词汇表和合并规则
        vocab[next_token_id] = vocab[most_frequent[0]] + vocab[most_frequent[1]]
        merges.append((vocab[most_frequent[0]], vocab[most_frequent[1]]))
        next_token_id += 1
    
    return vocab, merges
    """)


def main():
    """
    主函数：显示所有步骤的指南
    """
    print("BPE学习指南 - 分步骤实现")
    print("=" * 50)
    
    step1_guide()
    step2_guide()
    step3_guide()
    step4_guide()
    step5_guide()
    
    print("\n=== 开始实现 ===")
    print("现在你可以开始实现 bpe_framework.py 中的函数了！")
    print("建议按顺序实现：")
    print("1. simple_pretokenize")
    print("2. text_to_byte_sequences")
    print("3. count_byte_pairs")
    print("4. find_most_frequent_pair")
    print("5. merge_byte_pair")
    print("6. simple_bpe_train")
    print("\n运行 python bpe_framework.py 来测试你的实现！")


if __name__ == "__main__":
    main()
