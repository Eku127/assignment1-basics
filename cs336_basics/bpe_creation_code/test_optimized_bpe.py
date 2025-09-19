#!/usr/bin/env python3
"""
测试优化版BPE训练器
"""

import time
import tempfile
from pathlib import Path
from train_bpe_optimized import train_bpe_optimized


def test_small_dataset():
    """测试小数据集"""
    print("=== 测试优化版BPE训练器 ===\n")
    
    # 创建测试数据
    test_text = """Hello world! This is a test document.
<|endoftext|>
Another document with different content.
Machine learning is fascinating.
<|endoftext|>
Final document with numbers: 123 456 789.
Special characters: @#$%^&*()
<|endoftext|>"""
    
    # 重复多次以增加数据量
    repeated_text = test_text * 100
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(repeated_text)
        test_file = f.name
    
    print(f"测试文件: {test_file}")
    print(f"文件大小: {len(repeated_text):,} 字符")
    
    try:
        # 测试优化版本
        print("\n🚀 测试优化版BPE训练器...")
        start_time = time.time()
        
        results = train_bpe_optimized(
            input_path=test_file,
            vocab_size=100,
            special_tokens=["<|endoftext|>"]
        )
        
        training_time = time.time() - start_time
        
        if results:
            print(f"✅ 训练成功！")
            print(f"   训练时间: {training_time:.3f} 秒")
            print(f"   词汇表大小: {len(results['vocab'])}")
            print(f"   合并规则数: {len(results['merges'])}")
            
            # 显示一些学习到的token
            learned_tokens = {k: v for k, v in results['vocab'].items() if k >= 256}
            print(f"\n📝 学习到的token示例（前5个）:")
            for i, (token_id, token_bytes) in enumerate(sorted(learned_tokens.items())[:5]):
                try:
                    token_str = token_bytes.decode('utf-8', errors='replace')
                    print(f"     {token_id}: {repr(token_bytes)} -> '{token_str}'")
                except:
                    print(f"     {token_id}: {repr(token_bytes)}")
            
            return True
        else:
            print("❌ 训练失败！")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时文件
        try:
            Path(test_file).unlink()
            print(f"\n🧹 已清理临时文件")
        except:
            pass


def test_performance():
    """测试性能"""
    print("\n=== 性能测试 ===\n")
    
    # 创建更大的测试数据
    base_text = """This is a performance test document for BPE training.
It contains various words and patterns to test tokenization efficiency.
The quick brown fox jumps over the lazy dog multiple times.
Machine learning and natural language processing are complex fields.
<|endoftext|>
Another document with different vocabulary and sentence structures.
Deep learning models require extensive training data and computational resources.
Transformers have revolutionized the field of natural language processing.
Attention mechanisms allow models to focus on relevant parts of input sequences.
<|endoftext|>
Final document with numbers, symbols, and special characters: 123, 456, 789!
@#$%^&*() symbols and punctuation marks test robustness.
This comprehensive test helps evaluate the tokenizer's performance.
<|endoftext|>"""
    
    # 创建10MB的测试数据
    target_size = 10 * 1024 * 1024  # 10MB
    repeated_text = base_text * (target_size // len(base_text) + 1)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(repeated_text)
        test_file = f.name
    
    print(f"性能测试文件: {test_file}")
    print(f"文件大小: {len(repeated_text) / (1024*1024):.2f} MB")
    
    try:
        print("\n🚀 开始性能测试...")
        start_time = time.time()
        
        results = train_bpe_optimized(
            input_path=test_file,
            vocab_size=500,
            special_tokens=["<|endoftext|>"]
        )
        
        training_time = time.time() - start_time
        
        if results:
            print(f"✅ 性能测试成功！")
            print(f"   训练时间: {training_time:.2f} 秒")
            print(f"   处理速度: {len(repeated_text) / training_time / (1024*1024):.2f} MB/s")
            print(f"   词汇表大小: {len(results['vocab'])}")
            print(f"   合并规则数: {len(results['merges'])}")
            
            # 性能评级
            if training_time < 30:
                print("🏆 优秀！训练速度很快")
            elif training_time < 60:
                print("👍 良好！训练速度不错")
            else:
                print("⚠️  训练速度较慢，可能需要进一步优化")
            
            return True
        else:
            print("❌ 性能测试失败！")
            return False
            
    except Exception as e:
        print(f"❌ 性能测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时文件
        try:
            Path(test_file).unlink()
            print(f"\n🧹 已清理临时文件")
        except:
            pass


if __name__ == "__main__":
    print("优化版BPE训练器测试")
    print("=" * 50)
    
    # 基本功能测试
    success1 = test_small_dataset()
    
    if success1:
        # 性能测试
        success2 = test_performance()
        
        if success2:
            print("\n🎉 所有测试通过！优化版BPE训练器工作正常。")
        else:
            print("\n⚠️  基本功能正常，但性能测试失败。")
    else:
        print("\n❌ 基本功能测试失败，请检查代码。")
