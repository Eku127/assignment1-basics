#!/usr/bin/env python3
"""
BPE学习 - 第三步：理解预分词的重要性
"""

def demo_pretokenization():
    """演示预分词的作用"""
    
    # 假设我们有这样的文本
    text = "hello world! hello there."
    print(f"原始文本: '{text}'")
    print()
    
    # 方法1：直接按字节处理（错误的方法）
    print("方法1：直接按字节处理")
    bytes_data = text.encode('utf-8')
    print(f"字节序列: {list(bytes_data)}")
    print("问题：'hello' 和 'world!' 之间没有边界，可能会错误合并")
    print()
    
    # 方法2：先预分词，再按字节处理（正确的方法）
    print("方法2：先预分词，再按字节处理")
    
    # 简单的预分词：按空格分割
    words = text.split()
    print(f"预分词结果: {words}")
    
    # 对每个词单独处理
    for word in words:
        bytes_data = word.encode('utf-8')
        print(f"  '{word}' -> {list(bytes_data)}")
    
    print("优势：每个词独立处理，不会跨词边界合并")
    print()
    
    # 方法3：使用GPT-2风格的预分词
    print("方法3：使用GPT-2风格的预分词")
    import regex as regex_mod
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat = regex_mod.compile(PAT)
    
    pretokens = [m.group(0) for m in pat.finditer(text)]
    print(f"GPT-2预分词结果: {pretokens}")
    
    for token in pretokens:
        bytes_data = token.encode('utf-8')
        print(f"  '{token}' -> {list(bytes_data)}")

if __name__ == "__main__":
    demo_pretokenization()
