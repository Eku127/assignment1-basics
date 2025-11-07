#!/usr/bin/env python3
"""
Counter 演示 - 理解 Counter 的用法
"""

from collections import Counter

def counter_basic_demo():
    """Counter 基本用法演示"""
    print("=== Counter 基本用法演示 ===\n")
    
    # 1. 创建 Counter 的几种方式
    print("1. 创建 Counter 的几种方式:")
    
    # 方式1：从列表创建
    words = ['hello', 'world', 'hello', 'python', 'hello', 'world']
    word_counts = Counter(words)
    print(f"   从列表创建: {word_counts}")
    
    # 方式2：从字符串创建（统计字符频率）
    text = "hello"
    char_counts = Counter(text)
    print(f"   从字符串创建: {char_counts}")
    
    # 方式3：手动添加
    manual_counter = Counter()
    manual_counter['a'] += 1
    manual_counter['b'] += 2
    manual_counter['a'] += 1
    print(f"   手动添加: {manual_counter}")
    print()
    
    # 2. 常用方法
    print("2. 常用方法:")
    
    # most_common() - 获取最常见的元素
    print(f"   most_common(3): {word_counts.most_common(3)}")
    print(f"   most_common(1): {word_counts.most_common(1)}")
    
    # 访问元素
    print(f"   'hello' 出现次数: {word_counts['hello']}")
    print(f"   'python' 出现次数: {word_counts['python']}")
    print(f"   '不存在' 出现次数: {word_counts['不存在']}")  # 不存在的元素返回0
    print()
    
    # 3. 更新 Counter
    print("3. 更新 Counter:")
    print(f"   更新前: {word_counts}")
    word_counts.update(['hello', 'new', 'word'])
    print(f"   更新后: {word_counts}")
    print()
    
    # 4. 数学运算
    print("4. 数学运算:")
    counter1 = Counter(['a', 'b', 'c', 'a'])
    counter2 = Counter(['a', 'b', 'b', 'd'])
    print(f"   Counter1: {counter1}")
    print(f"   Counter2: {counter2}")
    print(f"   Counter1 + Counter2: {counter1 + counter2}")
    print(f"   Counter1 - Counter2: {counter1 - counter2}")
    print()

def counter_in_bpe_demo():
    """Counter 在 BPE 中的应用演示"""
    print("=== Counter 在 BPE 中的应用 ===\n")
    
    # 模拟字节序列
    byte_sequences = [
        [104, 101, 108, 108, 111],  # "hello"
        [119, 111, 114, 108, 100],  # "world"
        [104, 101, 108, 108, 111],  # "hello" (重复)
    ]
    
    print("字节序列:")
    for i, seq in enumerate(byte_sequences):
        print(f"  序列 {i}: {seq}")
    print()
    
    # 统计字节对频率
    print("统计字节对频率:")
    pair_counts = Counter()
    
    for seq in byte_sequences:
        print(f"  处理序列: {seq}")
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            pair_counts[pair] += 1
            print(f"    字节对 {pair}: 频率 +1")
    
    print(f"\n最终字节对频率: {dict(pair_counts)}")
    print(f"最频繁的字节对: {pair_counts.most_common(1)[0]}")
    print()
    
    # 展示如何找到最频繁的字节对
    print("找到最频繁的字节对:")
    if pair_counts:
        most_frequent = pair_counts.most_common(1)[0][0]
        frequency = pair_counts.most_common(1)[0][1]
        print(f"  最频繁的字节对: {most_frequent}")
        print(f"  出现次数: {frequency}")
        
        # 转换为字节显示
        byte1, byte2 = most_frequent
        print(f"  对应字节: {bytes([byte1])} + {bytes([byte2])}")
        print(f"  合并后: {bytes([byte1]) + bytes([byte2])}")

def why_use_counter():
    """为什么使用 Counter"""
    print("=== 为什么使用 Counter ===\n")
    
    print("1. 简洁性:")
    print("   不用 Counter:")
    print("   pair_counts = {}")
    print("   for pair in pairs:")
    print("       if pair in pair_counts:")
    print("           pair_counts[pair] += 1")
    print("       else:")
    print("           pair_counts[pair] = 1")
    print()
    print("   使用 Counter:")
    print("   pair_counts = Counter()")
    print("   for pair in pairs:")
    print("       pair_counts[pair] += 1")
    print()
    
    print("2. 便利方法:")
    print("   - most_common(): 直接获取最常见的元素")
    print("   - 不存在的元素返回 0，不会报错")
    print("   - 支持数学运算（+、-、&、|）")
    print("   - 可以像字典一样使用")
    print()
    
    print("3. 性能:")
    print("   - 内部使用字典实现，查找和更新都是 O(1)")
    print("   - 专门为计数优化")

if __name__ == "__main__":
    counter_basic_demo()
    counter_in_bpe_demo()
    why_use_counter()
