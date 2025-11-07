#!/usr/bin/env python3
"""
实验(d): 将数据集编码为token ID序列并序列化为uint16格式
"""

import argparse
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer


def load_documents(data_path: str, max_docs: int = None) -> List[str]:
    """
    加载文档，按<|endoftext|>分割
    
    Args:
        data_path: 数据文件路径
        max_docs: 最大文档数量（None表示加载所有）
        
    Returns:
        文档列表
    """
    documents = []
    current_doc = ""
    
    print(f"加载文档从: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取文件"):
            if line.strip() == "<|endoftext|>":
                if current_doc.strip():
                    documents.append(current_doc.strip())
                    current_doc = ""
                    if max_docs and len(documents) >= max_docs:
                        break
            else:
                current_doc += line
    
    # 添加最后一个文档（如果没有以<|endoftext|>结尾）
    if current_doc.strip():
        documents.append(current_doc.strip())
    
    print(f"加载了 {len(documents)} 个文档")
    return documents


def encode_documents(tokenizer: Tokenizer, documents: List[str], batch_size: int = 100) -> np.ndarray:
    """
    将文档编码为token ID序列
    
    Args:
        tokenizer: 用于编码的tokenizer
        documents: 文档列表
        batch_size: 批处理大小
        
    Returns:
        token ID的numpy数组
    """
    print(f"开始编码 {len(documents)} 个文档...")
    
    all_token_ids = []
    failed_docs = 0
    
    # 分批处理以避免内存问题
    for i in tqdm(range(0, len(documents), batch_size), desc="编码文档"):
        batch = documents[i:i + batch_size]
        
        for doc in batch:
            try:
                token_ids = tokenizer.encode(doc)
                all_token_ids.extend(token_ids)
            except Exception as e:
                print(f"警告: 文档编码失败: {e}")
                failed_docs += 1
                continue
    
    print(f"编码完成: {len(all_token_ids)} 个tokens, {failed_docs} 个失败文档")
    
    # 转换为numpy数组
    token_array = np.array(all_token_ids, dtype=np.uint16)
    
    return token_array


def analyze_token_range(token_array: np.ndarray, vocab_size: int) -> dict:
    """
    分析token ID的范围和分布
    
    Args:
        token_array: token ID数组
        vocab_size: 词汇表大小
        
    Returns:
        分析结果字典
    """
    min_id = np.min(token_array)
    max_id = np.max(token_array)
    unique_ids = len(np.unique(token_array))
    
    # 检查是否超出uint16范围
    exceeds_uint16 = max_id > 65535
    
    # 检查是否超出词汇表范围
    exceeds_vocab = max_id >= vocab_size
    
    return {
        'min_id': min_id,
        'max_id': max_id,
        'unique_ids': unique_ids,
        'total_tokens': len(token_array),
        'exceeds_uint16': exceeds_uint16,
        'exceeds_vocab': exceeds_vocab,
        'vocab_size': vocab_size,
        'coverage': unique_ids / vocab_size * 100
    }


def save_tokenized_data(token_array: np.ndarray, output_path: str) -> None:
    """
    保存tokenized数据到文件
    
    Args:
        token_array: token ID数组
        output_path: 输出文件路径
    """
    print(f"保存tokenized数据到: {output_path}")
    
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为numpy格式
    np.save(output_path, token_array)
    
    # 也保存为二进制格式（更紧凑）
    binary_path = output_path.replace('.npy', '.bin')
    token_array.tofile(binary_path)
    
    print(f"保存完成:")
    print(f"  NumPy格式: {output_path} ({Path(output_path).stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  二进制格式: {binary_path} ({Path(binary_path).stat().st_size / 1024 / 1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description='将数据集编码为token ID序列')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    parser.add_argument('--vocab_path', type=str, required=True, help='词汇表文件路径')
    parser.add_argument('--merges_path', type=str, required=True, help='合并规则文件路径')
    parser.add_argument('--special_token', type=str, default='<|endoftext|>', help='特殊token')
    parser.add_argument('--output_path', type=str, required=True, help='输出文件路径')
    parser.add_argument('--max_docs', type=int, default=None, help='最大文档数量')
    parser.add_argument('--batch_size', type=int, default=100, help='批处理大小')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("实验(d): 数据集编码为token ID序列")
    print("=" * 60)
    
    # 加载tokenizer
    print(f"\n1. 加载tokenizer...")
    start_time = time.time()
    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=[args.special_token]
    )
    load_time = time.time() - start_time
    print(f"   Tokenizer加载时间: {load_time:.2f} 秒")
    print(f"   词汇表大小: {len(tokenizer.vocab)}")
    
    # 加载文档
    print(f"\n2. 加载文档...")
    start_time = time.time()
    documents = load_documents(args.data_path, args.max_docs)
    load_time = time.time() - start_time
    print(f"   文档加载时间: {load_time:.2f} 秒")
    
    # 编码文档
    print(f"\n3. 编码文档...")
    start_time = time.time()
    token_array = encode_documents(tokenizer, documents, args.batch_size)
    encode_time = time.time() - start_time
    print(f"   编码时间: {encode_time:.2f} 秒")
    
    # 分析token范围
    print(f"\n4. 分析token范围...")
    analysis = analyze_token_range(token_array, len(tokenizer.vocab))
    
    print(f"   Token ID范围: {analysis['min_id']} - {analysis['max_id']}")
    print(f"   唯一token数量: {analysis['unique_ids']:,}")
    print(f"   总token数量: {analysis['total_tokens']:,}")
    print(f"   词汇表覆盖率: {analysis['coverage']:.2f}%")
    print(f"   超出uint16范围: {analysis['exceeds_uint16']}")
    print(f"   超出词汇表范围: {analysis['exceeds_vocab']}")
    
    # 保存数据
    print(f"\n5. 保存数据...")
    save_tokenized_data(token_array, args.output_path)
    
    # 总结
    print(f"\n" + "=" * 60)
    print(f"编码完成总结:")
    print(f"  输入文档数: {len(documents)}")
    print(f"  输出token数: {len(token_array):,}")
    print(f"  数据类型: uint16")
    print(f"  总处理时间: {load_time + encode_time:.2f} 秒")
    print(f"  平均速度: {len(token_array) / (load_time + encode_time):.0f} tokens/秒")
    print("=" * 60)


if __name__ == "__main__":
    main()
