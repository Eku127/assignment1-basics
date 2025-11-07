#!/usr/bin/env python3
"""
实验(d)批量处理脚本
同时处理TinyStories和OpenWebText数据集
"""

import subprocess
import sys
from pathlib import Path


def run_experiment_d():
    """运行实验(d)的所有任务"""
    
    print("=" * 80)
    print("实验(d): 将数据集编码为token ID序列")
    print("=" * 80)
    
    # 配置路径
    base_dir = Path("cs336_basics/bpe_creation_code")
    results_dir = base_dir / "bpe_results"
    experiments_dir = base_dir / "tokenizer_experiments"
    data_dir = Path("data")
    
    # 检查必要文件是否存在
    required_files = [
        data_dir / "TinyStoriesV2-GPT4-train.txt",
        data_dir / "owt_train.txt",
        results_dir / "tinystories_vocab.json",
        results_dir / "tinystories_merges.txt",
        results_dir / "owt_vocab.json",
        results_dir / "owt_merges.txt"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("❌ 缺少必要文件:")
        for f in missing_files:
            print(f"   {f}")
        return False
    
    print("✅ 所有必要文件都存在")
    
    # 创建输出目录
    output_dir = experiments_dir / "encoded_datasets"
    output_dir.mkdir(exist_ok=True)
    
    # 1. 处理TinyStories数据集
    print(f"\n{'='*60}")
    print("1. 处理TinyStories数据集")
    print(f"{'='*60}")
    
    tinystories_cmd = [
        "uv", "run", "python", str(experiments_dir / "encode_datasets.py"),
        "--data_path", str(data_dir / "TinyStoriesV2-GPT4-train.txt"),
        "--vocab_path", str(results_dir / "tinystories_vocab.json"),
        "--merges_path", str(results_dir / "tinystories_merges.txt"),
        "--special_token", "<|endoftext|>",
        "--output_path", str(output_dir / "tinystories_tokens.npy"),
        "--max_docs", "1000",  # 限制文档数量以加快处理
        "--batch_size", "50"
    ]
    
    print(f"运行命令: {' '.join(tinystories_cmd)}")
    result1 = subprocess.run(tinystories_cmd, capture_output=True, text=True)
    
    if result1.returncode == 0:
        print("✅ TinyStories编码完成")
        print(result1.stdout)
    else:
        print("❌ TinyStories编码失败")
        print(result1.stderr)
        return False
    
    # 2. 处理OpenWebText数据集
    print(f"\n{'='*60}")
    print("2. 处理OpenWebText数据集")
    print(f"{'='*60}")
    
    owt_cmd = [
        "uv", "run", "python", str(experiments_dir / "encode_datasets.py"),
        "--data_path", str(data_dir / "owt_train.txt"),
        "--vocab_path", str(results_dir / "owt_vocab.json"),
        "--merges_path", str(results_dir / "owt_merges.txt"),
        "--special_token", "<|endoftext|>",
        "--output_path", str(output_dir / "owt_tokens.npy"),
        "--max_docs", "1000",  # 限制文档数量以加快处理
        "--batch_size", "50"
    ]
    
    print(f"运行命令: {' '.join(owt_cmd)}")
    result2 = subprocess.run(owt_cmd, capture_output=True, text=True)
    
    if result2.returncode == 0:
        print("✅ OpenWebText编码完成")
        print(result2.stdout)
    else:
        print("❌ OpenWebText编码失败")
        print(result2.stderr)
        return False
    
    # 3. 分析结果
    print(f"\n{'='*60}")
    print("3. 分析结果")
    print(f"{'='*60}")
    
    analyze_results(output_dir)
    
    # 4. 回答为什么uint16是合适的选择
    print(f"\n{'='*60}")
    print("4. 为什么uint16是合适的选择？")
    print(f"{'='*60}")
    
    print_uint16_justification()
    
    print(f"\n{'='*80}")
    print("实验(d)完成！")
    print(f"{'='*80}")
    
    return True


def analyze_results(output_dir: Path):
    """分析编码结果"""
    import numpy as np
    
    print("分析编码结果...")
    
    # 分析TinyStories结果
    ts_path = output_dir / "tinystories_tokens.npy"
    if ts_path.exists():
        ts_tokens = np.load(ts_path)
        print(f"\nTinyStories结果:")
        print(f"  Token数量: {len(ts_tokens):,}")
        print(f"  ID范围: {ts_tokens.min()} - {ts_tokens.max()}")
        print(f"  唯一ID数: {len(np.unique(ts_tokens)):,}")
        print(f"  文件大小: {ts_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 分析OpenWebText结果
    owt_path = output_dir / "owt_tokens.npy"
    if owt_path.exists():
        owt_tokens = np.load(owt_path)
        print(f"\nOpenWebText结果:")
        print(f"  Token数量: {len(owt_tokens):,}")
        print(f"  ID范围: {owt_tokens.min()} - {owt_tokens.max()}")
        print(f"  唯一ID数: {len(np.unique(owt_tokens)):,}")
        print(f"  文件大小: {owt_path.stat().st_size / 1024 / 1024:.2f} MB")


def print_uint16_justification():
    """解释为什么uint16是合适的选择"""
    print("""
为什么uint16是合适的选择？

1. 内存效率：
   - uint16占用2字节，比int32(4字节)节省50%内存
   - 对于大型数据集，这能显著减少内存使用

2. 范围足够：
   - uint16范围: 0 - 65,535
   - TinyStories词汇表: 10,000 tokens (ID: 0-9,999)
   - OpenWebText词汇表: 32,000 tokens (ID: 0-31,999)
   - 两个词汇表都远小于uint16的最大值

3. 存储效率：
   - 减少磁盘存储空间
   - 加快I/O操作速度
   - 减少网络传输时间

4. 兼容性：
   - 大多数深度学习框架支持uint16
   - 可以轻松转换为其他数据类型
   - 与PyTorch的索引操作兼容

5. 实际考虑：
   - 825GB的Pile数据集使用uint16可以节省约400GB存储空间
   - 对于大规模训练，这种节省是显著的
    """)


if __name__ == "__main__":
    success = run_experiment_d()
    sys.exit(0 if success else 1)
