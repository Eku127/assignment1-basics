#!/usr/bin/env python3
"""
测量 tokenizer 吞吐量的脚本
"""
import time
import argparse
from typing import List
from cs336_basics.tokenizer import Tokenizer


def measure_tokenizer_throughput(
    tokenizer: Tokenizer, 
    test_texts: List[str], 
    warmup_runs: int = 3,
    test_runs: int = 10
) -> tuple[float, float, float]:
    """
    测量 tokenizer 的吞吐量
    
    Args:
        tokenizer: 要测试的 tokenizer
        test_texts: 测试文本列表
        warmup_runs: 预热运行次数
        test_runs: 实际测试运行次数
        
    Returns:
        (平均吞吐量 MB/s, 平均编码时间秒, 总字节数)
    """
    print(f"开始测量 tokenizer 吞吐量...")
    print(f"测试文本数量: {len(test_texts)}")
    print(f"预热运行: {warmup_runs} 次")
    print(f"测试运行: {test_runs} 次")
    
    # 计算总字节数
    total_bytes = sum(len(text.encode('utf-8')) for text in test_texts)
    print(f"总字节数: {total_bytes:,} bytes ({total_bytes / 1024 / 1024:.2f} MB)")
    
    # 预热运行
    print(f"\n预热阶段...")
    for i in range(warmup_runs):
        start_time = time.time()
        for text in test_texts:
            try:
                tokenizer.encode(text)
            except ValueError as e:
                print(f"警告: 编码失败: {e}")
        end_time = time.time()
        print(f"预热运行 {i+1}: {end_time - start_time:.3f} 秒")
    
    # 实际测试
    print(f"\n测试阶段...")
    times = []
    for i in range(test_runs):
        start_time = time.time()
        successful_encodes = 0
        failed_encodes = 0
        
        for text in test_texts:
            try:
                tokenizer.encode(text)
                successful_encodes += 1
            except ValueError:
                failed_encodes += 1
        
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        throughput = total_bytes / elapsed / 1024 / 1024  # MB/s
        print(f"测试运行 {i+1}: {elapsed:.3f} 秒, {throughput:.2f} MB/s, "
              f"成功: {successful_encodes}, 失败: {failed_encodes}")
    
    # 计算平均结果
    avg_time = sum(times) / len(times)
    avg_throughput = total_bytes / avg_time / 1024 / 1024  # MB/s
    
    print(f"\n结果:")
    print(f"平均编码时间: {avg_time:.3f} 秒")
    print(f"平均吞吐量: {avg_throughput:.2f} MB/s")
    print(f"平均吞吐量: {avg_throughput * 1024:.2f} KB/s")
    
    return avg_throughput, avg_time, total_bytes


def estimate_pile_processing_time(throughput_mb_s: float) -> dict:
    """
    估算处理 The Pile 数据集所需的时间
    
    Args:
        throughput_mb_s: tokenizer 吞吐量 (MB/s)
        
    Returns:
        包含各种时间估算的字典
    """
    pile_size_gb = 825  # The Pile 数据集大小
    pile_size_mb = pile_size_gb * 1024
    
    # 计算处理时间
    processing_time_seconds = pile_size_mb / throughput_mb_s
    processing_time_minutes = processing_time_seconds / 60
    processing_time_hours = processing_time_minutes / 60
    processing_time_days = processing_time_hours / 24
    
    return {
        'pile_size_gb': pile_size_gb,
        'throughput_mb_s': throughput_mb_s,
        'processing_time_seconds': processing_time_seconds,
        'processing_time_minutes': processing_time_minutes,
        'processing_time_hours': processing_time_hours,
        'processing_time_days': processing_time_days
    }


def load_sample_texts(data_path: str, num_docs: int = 100) -> List[str]:
    """加载样本文本用于测试"""
    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        current_doc = ""
        doc_count = 0
        
        for line in f:
            if line.strip() == "<|endoftext|>":
                if current_doc.strip():
                    texts.append(current_doc.strip())
                    current_doc = ""
                    doc_count += 1
                    if doc_count >= num_docs:
                        break
            else:
                current_doc += line
    
    print(f"加载了 {len(texts)} 个文档")
    return texts


def main():
    parser = argparse.ArgumentParser(description='测量 tokenizer 吞吐量')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    parser.add_argument('--vocab_path', type=str, required=True, help='词汇表文件路径')
    parser.add_argument('--merges_path', type=str, required=True, help='合并规则文件路径')
    parser.add_argument('--special_token', type=str, default='<|endoftext|>', help='特殊token')
    parser.add_argument('--num_docs', type=int, default=100, help='测试文档数量')
    parser.add_argument('--warmup_runs', type=int, default=3, help='预热运行次数')
    parser.add_argument('--test_runs', type=int, default=10, help='测试运行次数')
    
    args = parser.parse_args()
    
    # 加载 tokenizer
    print(f"加载 tokenizer...")
    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=[args.special_token]
    )
    print(f"词汇表大小: {len(tokenizer.vocab)}")
    
    # 加载测试文本
    print(f"加载测试文本...")
    test_texts = load_sample_texts(args.data_path, args.num_docs)
    
    # 测量吞吐量
    throughput, avg_time, total_bytes = measure_tokenizer_throughput(
        tokenizer, test_texts, args.warmup_runs, args.test_runs
    )
    
    # 估算 The Pile 处理时间
    print(f"\n" + "="*50)
    print(f"The Pile 数据集处理时间估算:")
    print(f"="*50)
    
    estimates = estimate_pile_processing_time(throughput)
    
    print(f"数据集大小: {estimates['pile_size_gb']} GB")
    print(f"Tokenizer 吞吐量: {estimates['throughput_mb_s']:.2f} MB/s")
    print(f"")
    print(f"预计处理时间:")
    print(f"  {estimates['processing_time_seconds']:.0f} 秒")
    print(f"  {estimates['processing_time_minutes']:.1f} 分钟")
    print(f"  {estimates['processing_time_hours']:.2f} 小时")
    print(f"  {estimates['processing_time_days']:.2f} 天")
    
    # 不同吞吐量下的对比
    print(f"\n不同吞吐量下的处理时间对比:")
    print(f"{'吞吐量 (MB/s)':<15} {'处理时间 (小时)':<15} {'处理时间 (天)':<15}")
    print(f"-" * 50)
    
    for target_throughput in [10, 50, 100, 200, 500, 1000]:
        target_estimates = estimate_pile_processing_time(target_throughput)
        print(f"{target_throughput:<15} {target_estimates['processing_time_hours']:<15.2f} {target_estimates['processing_time_days']:<15.2f}")


if __name__ == "__main__":
    main()
