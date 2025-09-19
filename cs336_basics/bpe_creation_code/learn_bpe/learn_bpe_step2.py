#!/usr/bin/env python3
"""
BPE学习 - 第二步：手动演示BPE合并过程
"""

def manual_bpe_demo():
    """手动演示BPE的合并过程"""
    
    # 假设我们有这样的训练数据
    training_data = ["hello", "world", "hello", "there"]
    print("训练数据:", training_data)
    print()
    
    # 步骤1：转换为字节序列
    print("步骤1：转换为字节序列")
    byte_sequences = []
    for word in training_data:
        bytes_data = word.encode('utf-8')
        byte_list = list(bytes_data)
        byte_sequences.append(byte_list)
        print(f"  '{word}' -> {byte_list}")
    print()
    
    # 步骤2：统计字节对频率
    print("步骤2：统计字节对频率")
    from collections import Counter
    
    pair_counts = Counter()
    for seq in byte_sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            pair_counts[pair] += 1
    
    print("字节对频率:")
    for pair, count in pair_counts.most_common():
        print(f"  {pair}: {count}次")
    print()
    
    # 步骤3：找到最频繁的字节对
    most_frequent_pair = pair_counts.most_common(1)[0][0]
    print(f"最频繁的字节对: {most_frequent_pair} (出现{pair_counts[most_frequent_pair]}次)")
    print()
    
    # 步骤4：合并最频繁的字节对
    print("步骤4：合并最频繁的字节对")
    new_token_id = 256  # 新token的ID
    
    # 在序列中查找并合并
    for i, seq in enumerate(byte_sequences):
        if len(seq) < 2:
            continue
            
        new_seq = []
        j = 0
        merged = False
        
        while j < len(seq):
            if (j < len(seq) - 1 and 
                seq[j] == most_frequent_pair[0] and 
                seq[j+1] == most_frequent_pair[1]):
                # 找到匹配的pair，进行合并
                new_seq.append(new_token_id)
                j += 2
                merged = True
            else:
                new_seq.append(seq[j])
                j += 1
        
        if merged:
            print(f"  序列 {i}: {seq} -> {new_seq}")
            byte_sequences[i] = new_seq
    
    print(f"  新token {new_token_id} 代表: {bytes(most_frequent_pair)}")
    print()
    
    # 步骤5：显示合并后的结果
    print("步骤5：合并后的结果")
    for i, seq in enumerate(byte_sequences):
        print(f"  '{training_data[i]}' -> {seq}")
    print()
    
    # 步骤6：继续下一轮合并
    print("步骤6：继续下一轮合并")
    pair_counts = Counter()
    for seq in byte_sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            pair_counts[pair] += 1
    
    print("新的字节对频率:")
    for pair, count in pair_counts.most_common():
        print(f"  {pair}: {count}次")
    
    if pair_counts:
        most_frequent_pair = pair_counts.most_common(1)[0][0]
        print(f"最频繁的字节对: {most_frequent_pair} (出现{pair_counts[most_frequent_pair]}次)")

if __name__ == "__main__":
    manual_bpe_demo()
