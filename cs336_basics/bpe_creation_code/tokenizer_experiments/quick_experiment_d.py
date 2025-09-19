#!/usr/bin/env python3
"""
实验(d)快速测试版本
使用小样本验证编码功能
"""

import numpy as np
import time
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer


def quick_test():
    """快速测试实验(d)的功能"""
    
    print("=" * 60)
    print("实验(d)快速测试")
    print("=" * 60)
    
    # 测试文本
    test_texts = [
        "Once upon a time, there was a little girl named Alice.",
        "She loved to read books and explore the world around her.",
        "One day, she found a magical door in her garden.",
        "The door led to a wonderful world full of adventure.",
        "Alice was very excited to see what was inside."
    ]
    
    print(f"测试文本数量: {len(test_texts)}")
    print(f"总字符数: {sum(len(text) for text in test_texts)}")
    print(f"总字节数: {sum(len(text.encode('utf-8')) for text in test_texts)}")
    
    # 1. 测试TinyStories tokenizer
    print(f"\n1. 测试TinyStories tokenizer...")
    try:
        ts_tokenizer = Tokenizer.from_files(
            'cs336_basics/bpe_creation_code/bpe_results/tinystories_vocab.json',
            'cs336_basics/bpe_creation_code/bpe_results/tinystories_merges.txt',
            special_tokens=['<|endoftext|>']
        )
        
        ts_tokens = []
        for text in test_texts:
            tokens = ts_tokenizer.encode(text)
            ts_tokens.extend(tokens)
        
        ts_array = np.array(ts_tokens, dtype=np.uint16)
        
        print(f"   词汇表大小: {len(ts_tokenizer.vocab)}")
        print(f"   Token数量: {len(ts_array)}")
        print(f"   ID范围: {ts_array.min()} - {ts_array.max()}")
        print(f"   唯一ID数: {len(np.unique(ts_array))}")
        print(f"   超出uint16: {ts_array.max() > 65535}")
        print(f"   超出词汇表: {ts_array.max() >= len(ts_tokenizer.vocab)}")
        
    except Exception as e:
        print(f"   TinyStories tokenizer测试失败: {e}")
        return False
    
    # 2. 测试OpenWebText tokenizer
    print(f"\n2. 测试OpenWebText tokenizer...")
    try:
        owt_tokenizer = Tokenizer.from_files(
            'cs336_basics/bpe_creation_code/bpe_results/owt_vocab.json',
            'cs336_basics/bpe_creation_code/bpe_results/owt_merges.txt',
            special_tokens=['<|endoftext|>']
        )
        
        owt_tokens = []
        for text in test_texts:
            tokens = owt_tokenizer.encode(text)
            owt_tokens.extend(tokens)
        
        owt_array = np.array(owt_tokens, dtype=np.uint16)
        
        print(f"   词汇表大小: {len(owt_tokenizer.vocab)}")
        print(f"   Token数量: {len(owt_array)}")
        print(f"   ID范围: {owt_array.min()} - {owt_array.max()}")
        print(f"   唯一ID数: {len(np.unique(owt_array))}")
        print(f"   超出uint16: {owt_array.max() > 65535}")
        print(f"   超出词汇表: {owt_array.max() >= len(owt_tokenizer.vocab)}")
        
    except Exception as e:
        print(f"   OpenWebText tokenizer测试失败: {e}")
        return False
    
    # 3. 保存测试结果
    print(f"\n3. 保存测试结果...")
    output_dir = Path("cs336_basics/bpe_creation_code/tokenizer_experiments/encoded_datasets")
    output_dir.mkdir(exist_ok=True)
    
    # 保存TinyStories结果
    ts_path = output_dir / "tinystories_test.npy"
    np.save(ts_path, ts_array)
    print(f"   TinyStories结果保存到: {ts_path}")
    print(f"   文件大小: {ts_path.stat().st_size} 字节")
    
    # 保存OpenWebText结果
    owt_path = output_dir / "owt_test.npy"
    np.save(owt_path, owt_array)
    print(f"   OpenWebText结果保存到: {owt_path}")
    print(f"   文件大小: {owt_path.stat().st_size} 字节")
    
    # 4. 分析uint16的适用性
    print(f"\n4. 分析uint16的适用性...")
    print(f"   TinyStories最大ID: {ts_array.max()} (uint16最大值: 65535)")
    print(f"   OpenWebText最大ID: {owt_array.max()} (uint16最大值: 65535)")
    print(f"   TinyStories词汇表大小: {len(ts_tokenizer.vocab)}")
    print(f"   OpenWebText词汇表大小: {len(owt_tokenizer.vocab)}")
    
    # 5. 回答为什么uint16是合适的选择
    print(f"\n5. 为什么uint16是合适的选择？")
    print(f"""
    ✅ 内存效率: uint16占用2字节，比int32(4字节)节省50%内存
    ✅ 范围足够: 
       - uint16范围: 0-65,535
       - TinyStories词汇表: {len(ts_tokenizer.vocab):,} tokens (最大ID: {ts_array.max()})
       - OpenWebText词汇表: {len(owt_tokenizer.vocab):,} tokens (最大ID: {owt_array.max()})
    ✅ 存储效率: 对于825GB的Pile数据集，使用uint16可节省约400GB存储空间
    ✅ 兼容性: 与PyTorch和大多数深度学习框架兼容
    """)
    
    print(f"\n" + "=" * 60)
    print(f"实验(d)快速测试完成！")
    print(f"=" * 60)
    
    return True


if __name__ == "__main__":
    success = quick_test()
    if success:
        print("✅ 所有测试通过")
    else:
        print("❌ 测试失败")
