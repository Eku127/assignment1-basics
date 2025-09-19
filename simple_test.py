"""
简化的对比测试
"""

import sys
sys.path.append('.')

def test_basic_comparison():
    print("=== 基本对比测试 ===")
    
    try:
        from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
        print("✅ 原始tokenizer导入成功")
    except Exception as e:
        print(f"❌ 原始tokenizer导入失败: {e}")
        return
    
    try:
        from cs336_basics.tokenizer_improved import Tokenizer as ImprovedTokenizer
        print("✅ 改进tokenizer导入成功")
    except Exception as e:
        print(f"❌ 改进tokenizer导入失败: {e}")
        return
    
    try:
        import tiktoken
        print("✅ tiktoken导入成功")
    except Exception as e:
        print(f"❌ tiktoken导入失败: {e}")
        return
    
    # 测试基本功能
    vocab_path = 'tests/fixtures/gpt2_vocab.json'
    merges_path = 'tests/fixtures/gpt2_merges.txt'
    special_tokens = ['<|endoftext|>']
    
    print("\n创建tokenizer...")
    
    try:
        original_tokenizer = get_tokenizer_from_vocab_merges_path(
            vocab_path=vocab_path,
            merges_path=merges_path,
            special_tokens=special_tokens
        )
        print("✅ 原始tokenizer创建成功")
    except Exception as e:
        print(f"❌ 原始tokenizer创建失败: {e}")
        return
    
    try:
        improved_tokenizer = ImprovedTokenizer.from_files(
            vocab_filepath=vocab_path,
            merges_filepath=merges_path,
            special_tokens=special_tokens
        )
        print("✅ 改进tokenizer创建成功")
    except Exception as e:
        print(f"❌ 改进tokenizer创建失败: {e}")
        return
    
    # 简单测试
    test_text = "hello"
    print(f"\n测试文本: {repr(test_text)}")
    
    try:
        original_result = original_tokenizer.encode(test_text)
        print(f"原始tokenizer: {original_result}")
    except Exception as e:
        print(f"原始tokenizer错误: {e}")
    
    try:
        improved_result = improved_tokenizer.encode(test_text)
        print(f"改进tokenizer: {improved_result}")
    except Exception as e:
        print(f"改进tokenizer错误: {e}")

if __name__ == "__main__":
    test_basic_comparison()
