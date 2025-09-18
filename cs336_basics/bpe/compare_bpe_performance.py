#!/usr/bin/env python3
"""
BPE训练性能对比脚本
比较原始版本和优化版本的性能差异
"""

import time
import tempfile
from pathlib import Path
from train_bpe_with_progress import train_bpe_with_dataset
from train_bpe_optimized import train_bpe_optimized


def create_test_data(size_mb: int = 10) -> str:
    """创建测试数据"""
    # 创建包含多个文档的测试文本
    base_text = """This is a sample document for testing BPE performance.
It contains various words and patterns that will be used for tokenization.
The quick brown fox jumps over the lazy dog.
Machine learning and natural language processing are fascinating fields.
<|endoftext|>
Another document with different content and vocabulary.
Deep learning models require large amounts of training data.
Transformers have revolutionized the field of NLP.
Attention mechanisms allow models to focus on relevant parts of the input.
<|endoftext|>
Final document with numbers and special characters: 123, 456, 789!
@#$%^&*() symbols and punctuation marks.
This helps test the robustness of the tokenizer.
<|endoftext|>"""
    
    # 重复文本以达到指定大小
    target_size = size_mb * 1024 * 1024  # 转换为字节
    repeated_text = base_text * (target_size // len(base_text) + 1)
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(repeated_text)
        return f.name


def compare_performance(test_file: str, vocab_size: int = 1000):
    """比较两个版本的性能"""
    print("=== BPE训练性能对比 ===\n")
    
    print(f"测试配置:")
    print(f"  测试文件: {test_file}")
    print(f"  文件大小: {Path(test_file).stat().st_size / (1024*1024):.2f} MB")
    print(f"  词汇表大小: {vocab_size}")
    print()
    
    # 测试原始版本
    print("🔄 测试原始版本...")
    start_time = time.time()
    
    try:
        results_original = train_bpe_with_dataset(
            input_path=test_file,
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"]
        )
        original_time = time.time() - start_time
        
        if results_original:
            print(f"✅ 原始版本完成: {original_time:.2f} 秒")
            print(f"   词汇表大小: {len(results_original['vocab'])}")
            print(f"   合并规则数: {len(results_original['merges'])}")
        else:
            print("❌ 原始版本失败")
            original_time = None
    except Exception as e:
        print(f"❌ 原始版本失败: {e}")
        original_time = None
    
    print()
    
    # 测试优化版本
    print("🚀 测试优化版本...")
    start_time = time.time()
    
    try:
        results_optimized = train_bpe_optimized(
            input_path=test_file,
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"]
        )
        optimized_time = time.time() - start_time
        
        if results_optimized:
            print(f"✅ 优化版本完成: {optimized_time:.2f} 秒")
            print(f"   词汇表大小: {len(results_optimized['vocab'])}")
            print(f"   合并规则数: {len(results_optimized['merges'])}")
        else:
            print("❌ 优化版本失败")
            optimized_time = None
    except Exception as e:
        print(f"❌ 优化版本失败: {e}")
        optimized_time = None
    
    print()
    
    # 性能比较
    if original_time and optimized_time:
        speedup = original_time / optimized_time
        time_saved = original_time - optimized_time
        improvement_percent = (time_saved / original_time) * 100
        
        print(f"📊 性能比较结果:")
        print(f"   原始版本: {original_time:.2f} 秒")
        print(f"   优化版本: {optimized_time:.2f} 秒")
        print(f"   加速比: {speedup:.2f}x")
        print(f"   时间节省: {time_saved:.2f} 秒 ({improvement_percent:.1f}%)")
        
        # 检查结果一致性
        if results_original and results_optimized:
            vocab_match = results_original['vocab'] == results_optimized['vocab']
            merges_match = results_original['merges'] == results_optimized['merges']
            
            print(f"\n🔍 结果验证:")
            print(f"   词汇表一致: {vocab_match}")
            print(f"   合并规则一致: {merges_match}")
            
            if not vocab_match:
                print(f"   ⚠️  词汇表不一致！")
            if not merges_match:
                print(f"   ⚠️  合并规则不一致！")
        
        # 性能评级
        if speedup >= 2.0:
            print(f"\n🏆 优秀！优化效果显著 (加速 {speedup:.1f}x)")
        elif speedup >= 1.5:
            print(f"\n👍 良好！优化效果明显 (加速 {speedup:.1f}x)")
        elif speedup >= 1.2:
            print(f"\n✅ 一般！有一定优化效果 (加速 {speedup:.1f}x)")
        else:
            print(f"\n⚠️  优化效果有限 (加速 {speedup:.1f}x)")
    
    return original_time, optimized_time


def main():
    """主函数"""
    import sys
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "small":
            size_mb = 5
            vocab_size = 500
        elif sys.argv[1] == "medium":
            size_mb = 20
            vocab_size = 1000
        elif sys.argv[1] == "large":
            size_mb = 50
            vocab_size = 2000
        else:
            size_mb = int(sys.argv[1])
            vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    else:
        size_mb = 10
        vocab_size = 1000
    
    print(f"创建 {size_mb}MB 测试数据...")
    test_file = create_test_data(size_mb)
    
    try:
        # 运行性能对比
        original_time, optimized_time = compare_performance(test_file, vocab_size)
        
        if original_time and optimized_time:
            print(f"\n🎯 总结:")
            print(f"   在 {size_mb}MB 数据上，优化版本比原始版本快 {original_time/optimized_time:.2f} 倍")
            print(f"   节省时间: {original_time - optimized_time:.2f} 秒")
        
    finally:
        # 清理临时文件
        try:
            Path(test_file).unlink()
            print(f"\n🧹 已清理临时文件: {test_file}")
        except:
            pass


if __name__ == "__main__":
    print("BPE训练性能对比工具")
    print("使用方法:")
    print("  python compare_bpe_performance.py           # 默认10MB测试")
    print("  python compare_bpe_performance.py small     # 5MB测试")
    print("  python compare_bpe_performance.py medium    # 20MB测试")
    print("  python compare_bpe_performance.py large     # 50MB测试")
    print("  python compare_bpe_performance.py <size_mb> [vocab_size]")
    print()
    
    main()
