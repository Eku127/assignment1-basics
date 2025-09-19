#!/usr/bin/env python3
"""
实验(d): 流式编码数据集为token ID序列
使用encode_iterable进行内存高效的编码
"""

import argparse
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

from cs336_basics.tokenizer_improved import Tokenizer


def encode_dataset_streaming(
    data_path: str,
    vocab_path: str,
    merges_path: str,
    special_token: str,
    output_path: str,
    max_docs: int = None
) -> dict:
    """
    使用流式处理编码数据集
    
    Args:
        data_path: 数据文件路径
        vocab_path: 词汇表文件路径
        merges_path: 合并规则文件路径
        special_token: 特殊token
        output_path: 输出文件路径
        max_docs: 最大文档数量
        
    Returns:
        编码统计信息
    """
    print("=" * 60)
    print("实验(d): 流式编码数据集")
    print("=" * 60)
    
    # 1. 加载tokenizer
    print(f"\n1. 加载tokenizer...")
    start_time = time.time()
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=[special_token]
    )
    load_time = time.time() - start_time
    print(f"   Tokenizer加载时间: {load_time:.2f} 秒")
    print(f"   词汇表大小: {len(tokenizer.vocab)}")
    
    # 2. 流式编码
    print(f"\n2. 流式编码文档...")
    start_time = time.time()
    
    # 收集所有token IDs
    all_token_ids = []
    doc_count = 0
    current_doc = ""
    
    # 创建输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        # 使用tqdm显示进度
        for line in tqdm(f, desc="处理文件"):
            # 检查当前行是否包含特殊token
            if special_token in line:
                # 分割行，处理特殊token前后的内容
                parts = line.split(special_token)
                
                # 添加特殊token之前的内容到当前文档
                if parts[0].strip():
                    current_doc += parts[0]
                
                # 如果当前文档不为空，编码它
                if current_doc.strip():
                    try:
                        for token_id in tokenizer.encode_iterable([current_doc.strip()]):
                            all_token_ids.append(token_id)
                        doc_count += 1
                        current_doc = ""
                        
                        if max_docs and doc_count >= max_docs:
                            break
                    except Exception as e:
                        print(f"警告: 文档编码失败: {e}")
                        current_doc = ""
                
                # 处理特殊token之后的内容作为新文档的开始
                if len(parts) > 1 and parts[1].strip():
                    current_doc = parts[1]
            else:
                # 普通行，直接添加到当前文档
                current_doc += line
    
    # 处理最后一个文档
    if current_doc.strip():
        try:
            for token_id in tokenizer.encode_iterable([current_doc.strip()]):
                all_token_ids.append(token_id)
            doc_count += 1
        except Exception as e:
            print(f"警告: 最后一个文档编码失败: {e}")
    
    encode_time = time.time() - start_time
    print(f"   编码时间: {encode_time:.2f} 秒")
    print(f"   处理文档数: {doc_count}")
    print(f"   总token数: {len(all_token_ids):,}")
    
    # 3. 转换为numpy数组
    print(f"\n3. 转换为numpy数组...")
    token_array = np.array(all_token_ids, dtype=np.uint16)
    
    # 4. 分析结果
    print(f"\n4. 分析结果...")
    min_id = np.min(token_array)
    max_id = np.max(token_array)
    unique_ids = len(np.unique(token_array))
    
    print(f"   Token ID范围: {min_id} - {max_id}")
    print(f"   唯一token数量: {unique_ids:,}")
    print(f"   总token数量: {len(token_array):,}")
    print(f"   词汇表覆盖率: {unique_ids / len(tokenizer.vocab) * 100:.2f}%")
    print(f"   超出uint16范围: {max_id > 65535}")
    print(f"   超出词汇表范围: {max_id >= len(tokenizer.vocab)}")
    
    # 5. 保存数据
    print(f"\n5. 保存数据...")
    np.save(output_path, token_array)
    
    # 也保存为二进制格式
    binary_path = output_path.replace('.npy', '.bin')
    token_array.tofile(binary_path)
    
    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024
    binary_size_mb = Path(binary_path).stat().st_size / 1024 / 1024
    
    print(f"   保存完成:")
    print(f"     NumPy格式: {output_path} ({file_size_mb:.2f} MB)")
    print(f"     二进制格式: {binary_path} ({binary_size_mb:.2f} MB)")
    
    # 6. 返回统计信息
    return {
        'doc_count': doc_count,
        'total_tokens': len(token_array),
        'unique_tokens': unique_ids,
        'min_id': min_id,
        'max_id': max_id,
        'vocab_size': len(tokenizer.vocab),
        'coverage': unique_ids / len(tokenizer.vocab) * 100,
        'exceeds_uint16': max_id > 65535,
        'exceeds_vocab': max_id >= len(tokenizer.vocab),
        'file_size_mb': file_size_mb,
        'processing_time': encode_time,
        'tokens_per_second': len(token_array) / encode_time
    }


def main():
    parser = argparse.ArgumentParser(description='流式编码数据集为token ID序列')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    parser.add_argument('--vocab_path', type=str, required=True, help='词汇表文件路径')
    parser.add_argument('--merges_path', type=str, required=True, help='合并规则文件路径')
    parser.add_argument('--special_token', type=str, default='<|endoftext|>', help='特殊token')
    parser.add_argument('--output_path', type=str, required=True, help='输出文件路径')
    parser.add_argument('--max_docs', type=int, default=None, help='最大文档数量')
    
    args = parser.parse_args()
    
    # 执行编码
    stats = encode_dataset_streaming(
        args.data_path,
        args.vocab_path,
        args.merges_path,
        args.special_token,
        args.output_path,
        args.max_docs
    )
    
    # 打印总结
    print(f"\n" + "=" * 60)
    print(f"编码完成总结:")
    print(f"  处理文档数: {stats['doc_count']}")
    print(f"  总token数: {stats['total_tokens']:,}")
    print(f"  唯一token数: {stats['unique_tokens']:,}")
    print(f"  词汇表覆盖率: {stats['coverage']:.2f}%")
    print(f"  数据类型: uint16")
    print(f"  处理时间: {stats['processing_time']:.2f} 秒")
    print(f"  处理速度: {stats['tokens_per_second']:.0f} tokens/秒")
    print(f"  文件大小: {stats['file_size_mb']:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
