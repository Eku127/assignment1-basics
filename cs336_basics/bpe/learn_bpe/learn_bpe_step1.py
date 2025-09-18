#!/usr/bin/env python3
"""
BPE学习 - 第一步：理解字符和字节的区别
"""

def show_character_level(text):
    """显示字符级别的分词"""
    print(f"文本: '{text}'")
    print("字符级别分词:")
    for i, char in enumerate(text):
        print(f"  {i}: '{char}' (Unicode: {ord(char)})")
    print()

def show_byte_level(text):
    """显示字节级别的分词"""
    print(f"文本: '{text}'")
    print("字节级别分词:")
    bytes_data = text.encode('utf-8')
    for i, byte_val in enumerate(bytes_data):
        print(f"  {i}: {byte_val} (字节: {bytes([byte_val])})")
    print()

if __name__ == "__main__":
    # 测试不同的文本
    test_texts = ["hello", "世界", "hello world"]
    
    for text in test_texts:
        show_character_level(text)
        show_byte_level(text)
        print("-" * 50)
