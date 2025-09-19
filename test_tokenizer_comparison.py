"""
对比测试 tokenizer_improved 和原始 tokenizer 的区别
特别针对换行符处理
"""

import sys
sys.path.append('.')

from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
from cs336_basics.tokenizer_improved import Tokenizer as ImprovedTokenizer
import tiktoken


def test_newline_handling():
    """测试换行符处理差异"""
    print("=== 换行符处理对比测试 ===")
    
    # 使用GPT-2的vocab和merges
    vocab_path = 'tests/fixtures/gpt2_vocab.json'
    merges_path = 'tests/fixtures/gpt2_merges.txt'
    special_tokens = ['<|endoftext|>']
    
    # 创建两个tokenizer
    print("创建tokenizer...")
    original_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=special_tokens
    )
    
    improved_tokenizer = ImprovedTokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens
    )
    
    # 创建tiktoken参考
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    
    print("✅ 所有tokenizer创建成功")
    
    # 测试用例
    test_cases = [
        "hello",                    # 普通文本
        "\n",                      # 单个换行符
        "\n\n",                    # 两个换行符
        "hello\nworld",            # 文本+换行符
        "hello\n\nworld",          # 文本+两个换行符
        "<|endoftext|>\n\ntest",   # 特殊token+换行符
        "test\n\n<|endoftext|>",   # 换行符+特殊token
    ]
    
    print(f"\n{'测试用例':<25} {'原始tokenizer':<20} {'改进tokenizer':<20} {'tiktoken':<20} {'匹配?'}")
    print("-" * 100)
    
    for test_text in test_cases:
        # 编码
        try:
            original_result = original_tokenizer.encode(test_text)
        except Exception as e:
            original_result = f"ERROR: {e}"
            
        try:
            improved_result = improved_tokenizer.encode(test_text)
        except Exception as e:
            improved_result = f"ERROR: {e}"
            
        try:
            reference_result = reference_tokenizer.encode(test_text, allowed_special=set(special_tokens))
        except Exception as e:
            reference_result = f"ERROR: {e}"
        
        # 检查匹配
        original_match = "✅" if str(original_result) == str(reference_result) else "❌"
        improved_match = "✅" if str(improved_result) == str(reference_result) else "❌"
        
        print(f"{repr(test_text):<25} {str(original_result):<20} {str(improved_result):<20} {str(reference_result):<20} {original_match}/{improved_match}")


def test_specific_failing_case():
    """测试具体失败的测试用例"""
    print("\n=== 具体失败测试用例 ===")
    
    # 读取测试文件
    with open('tests/fixtures/special_token_double_newlines_non_whitespace.txt', 'r') as f:
        test_content = f.read()
    
    print(f"测试文件内容: {repr(test_content)}")
    
    # 创建tokenizer
    vocab_path = 'tests/fixtures/gpt2_vocab.json'
    merges_path = 'tests/fixtures/gpt2_merges.txt'
    special_tokens = ['<|endoftext|>']
    
    original_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=special_tokens
    )
    
    improved_tokenizer = ImprovedTokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens
    )
    
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    
    # 编码测试
    print("\n编码结果对比:")
    try:
        original_ids = original_tokenizer.encode(test_content)
        print(f"原始tokenizer: {original_ids}")
        print(f"解码: {[original_tokenizer.decode([x]) for x in original_ids]}")
    except Exception as e:
        print(f"原始tokenizer ERROR: {e}")
    
    try:
        improved_ids = improved_tokenizer.encode(test_content)
        print(f"改进tokenizer: {improved_ids}")
        print(f"解码: {[improved_tokenizer.decode([x]) for x in improved_ids]}")
    except Exception as e:
        print(f"改进tokenizer ERROR: {e}")
    
    try:
        reference_ids = reference_tokenizer.encode(test_content, allowed_special=set(special_tokens))
        print(f"tiktoken参考: {reference_ids}")
        print(f"解码: {[reference_tokenizer.decode([x]) for x in reference_ids]}")
    except Exception as e:
        print(f"tiktoken参考 ERROR: {e}")
    
    # 检查匹配
    print(f"\n匹配检查:")
    try:
        original_match = original_tokenizer.encode(test_content) == reference_tokenizer.encode(test_content, allowed_special=set(special_tokens))
        print(f"原始tokenizer vs tiktoken: {'✅ 匹配' if original_match else '❌ 不匹配'}")
    except:
        print(f"原始tokenizer vs tiktoken: ❌ 错误")
    
    try:
        improved_match = improved_tokenizer.encode(test_content) == reference_tokenizer.encode(test_content, allowed_special=set(special_tokens))
        print(f"改进tokenizer vs tiktoken: {'✅ 匹配' if improved_match else '❌ 不匹配'}")
    except:
        print(f"改进tokenizer vs tiktoken: ❌ 错误")


def test_merge_behavior():
    """测试合并行为差异"""
    print("\n=== 合并行为分析 ===")
    
    vocab_path = 'tests/fixtures/gpt2_vocab.json'
    merges_path = 'tests/fixtures/gpt2_merges.txt'
    special_tokens = ['<|endoftext|>']
    
    # 创建tokenizer
    original_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=special_tokens
    )
    
    improved_tokenizer = ImprovedTokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens
    )
    
    # 检查换行符合并规则
    print("检查merges中的换行符规则:")
    newline_merges = []
    for i, merge in enumerate(original_tokenizer.merges):
        if b'\n' in merge[0] or b'\n' in merge[1]:
            newline_merges.append((i, merge))
            print(f"  Merge {i}: {merge[0]} + {merge[1]} -> {merge[0] + merge[1]}")
    
    # 测试换行符处理
    test_text = "\n\n"
    print(f"\n测试文本: {repr(test_text)}")
    
    # 原始tokenizer
    try:
        original_result = original_tokenizer.encode(test_text)
        print(f"原始tokenizer结果: {original_result}")
        print(f"解码: {[original_tokenizer.decode([x]) for x in original_result]}")
    except Exception as e:
        print(f"原始tokenizer错误: {e}")
    
    # 改进tokenizer
    try:
        improved_result = improved_tokenizer.encode(test_text)
        print(f"改进tokenizer结果: {improved_result}")
        print(f"解码: {[improved_tokenizer.decode([x]) for x in improved_result]}")
    except Exception as e:
        print(f"改进tokenizer错误: {e}")
    
    # tiktoken参考
    try:
        reference_result = tiktoken.get_encoding("gpt2").encode(test_text)
        print(f"tiktoken参考结果: {reference_result}")
        print(f"解码: {[tiktoken.get_encoding('gpt2').decode([x]) for x in reference_result]}")
    except Exception as e:
        print(f"tiktoken参考错误: {e}")


if __name__ == "__main__":
    test_newline_handling()
    test_specific_failing_case()
    test_merge_behavior()
