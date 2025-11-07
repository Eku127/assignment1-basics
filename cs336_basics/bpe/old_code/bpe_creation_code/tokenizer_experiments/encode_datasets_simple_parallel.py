#!/usr/bin/env python3
"""
实验(d): 简单并行编码数据集为token ID序列
使用多进程并行处理，简化版本
uv run python cs336_basics/bpe_creation_code/tokenizer_experiments/encode_datasets_simple_parallel.py \                                              130 ↵ jjj@test
  --data_path data/owt_train.txt \               
  --vocab_path cs336_basics/bpe_creation_code/bpe_results/owt_vocab.json \        
  --merges_path cs336_basics/bpe_creation_code/bpe_results/owt_merges.txt \        
  --special_token "<|endoftext|>" \
  --output_path cs336_basics/bpe_creation_code/tokenizer_experiments/encoded_datasets/owt_train_parallel.npy \  
  --num_processes 4 \ 
  --batch_size 2000
"""

import argparse
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from cs336_basics.tokenizer_improved import Tokenizer


def encode_documents_worker(args):
    """
    工作进程函数：编码一批文档
    
    Args:
        args: (documents, vocab_path, merges_path, special_token, worker_id)
        
    Returns:
        (worker_id, token_ids, doc_count, error_count)
    """
    documents, vocab_path, merges_path, special_token, worker_id = args
    
    try:
        # 在每个进程中创建tokenizer
        tokenizer = Tokenizer.from_files(
            vocab_path,
            merges_path,
            special_tokens=[special_token]
        )
        
        all_token_ids = []
        doc_count = 0
        error_count = 0
        
        for doc in documents:
            try:
                # 使用encode方法
                token_ids = tokenizer.encode(doc)
                all_token_ids.extend(token_ids)
                doc_count += 1
            except Exception as e:
                error_count += 1
                continue
        
        return (worker_id, all_token_ids, doc_count, error_count)
        
    except Exception as e:
        print(f"工作进程 {worker_id} 失败: {e}")
        return (worker_id, [], 0, len(documents))


def load_documents_batches(data_path: str, max_docs: int = None, batch_size: int = 1000):
    """
    加载文档并分批
    
    Args:
        data_path: 数据文件路径
        max_docs: 最大文档数量
        batch_size: 批次大小
        
    Returns:
        文档批次列表
    """
    print(f"加载文档从: {data_path}")
    
    documents = []
    current_doc = ""
    doc_count = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取文件"):
            # 检查当前行是否包含<|endoftext|>
            if "<|endoftext|>" in line:
                # 分割行，处理<|endoftext|>前后的内容
                parts = line.split("<|endoftext|>")
                
                # 添加<|endoftext|>之前的内容到当前文档
                if parts[0].strip():
                    current_doc += parts[0]
                
                # 如果当前文档不为空，保存它
                if current_doc.strip():
                    documents.append(current_doc.strip())
                    current_doc = ""
                    doc_count += 1
                    if max_docs and doc_count >= max_docs:
                        break
                
                # 处理<|endoftext|>之后的内容作为新文档的开始
                if len(parts) > 1 and parts[1].strip():
                    current_doc = parts[1]
            else:
                # 普通行，直接添加到当前文档
                current_doc += line
    
    # 处理最后一个文档
    if current_doc.strip():
        documents.append(current_doc.strip())
    
    # 分批
    batches = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batches.append(batch)
    
    print(f"加载了 {len(documents)} 个文档，分为 {len(batches)} 个批次")
    return batches


def encode_dataset_simple_parallel(
    data_path: str,
    vocab_path: str,
    merges_path: str,
    special_token: str,
    output_path: str,
    max_docs: int = None,
    num_processes: int = None,
    batch_size: int = 1000
) -> dict:
    """
    简单并行编码数据集
    
    Args:
        data_path: 数据文件路径
        vocab_path: 词汇表文件路径
        merges_path: 合并规则文件路径
        special_token: 特殊token
        output_path: 输出文件路径
        max_docs: 最大文档数量
        num_processes: 进程数量
        batch_size: 批次大小
        
    Returns:
        编码统计信息
    """
    print("=" * 60)
    print("实验(d): 简单并行编码数据集")
    print("=" * 60)
    
    if num_processes is None:
        num_processes = min(cpu_count(), 4)  # 限制进程数
    
    print(f"使用 {num_processes} 个进程，批次大小: {batch_size}")
    
    # 1. 加载文档批次
    print(f"\n1. 加载文档批次...")
    start_time = time.time()
    doc_batches = load_documents_batches(data_path, max_docs, batch_size)
    load_time = time.time() - start_time
    print(f"   文档加载时间: {load_time:.2f} 秒")
    
    # 2. 准备并行处理参数
    print(f"\n2. 准备并行处理...")
    worker_args = []
    for i, batch in enumerate(doc_batches):
        worker_args.append((batch, vocab_path, merges_path, special_token, i))
    
    # 3. 并行编码
    print(f"\n3. 并行编码文档...")
    start_time = time.time()
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(encode_documents_worker, worker_args),
            total=len(worker_args),
            desc="编码进度",
            unit="批次"
        ))
    
    encode_time = time.time() - start_time
    
    # 4. 合并结果
    print(f"\n4. 合并结果...")
    all_token_ids = []
    total_docs = 0
    total_errors = 0
    
    # 按worker_id排序结果
    results.sort(key=lambda x: x[0])
    
    for worker_id, token_ids, doc_count, error_count in results:
        all_token_ids.extend(token_ids)
        total_docs += doc_count
        total_errors += error_count
    
    print(f"   编码时间: {encode_time:.2f} 秒")
    print(f"   处理文档数: {total_docs}")
    print(f"   错误文档数: {total_errors}")
    print(f"   总token数: {len(all_token_ids):,}")
    
    # 5. 转换为numpy数组
    print(f"\n5. 转换为numpy数组...")
    token_array = np.array(all_token_ids, dtype=np.uint16)
    
    # 6. 分析结果
    print(f"\n6. 分析结果...")
    min_id = np.min(token_array)
    max_id = np.max(token_array)
    unique_ids = len(np.unique(token_array))
    
    print(f"   Token ID范围: {min_id} - {max_id}")
    print(f"   唯一token数量: {unique_ids:,}")
    print(f"   总token数量: {len(token_array):,}")
    print(f"   超出uint16范围: {max_id > 65535}")
    
    # 7. 保存数据
    print(f"\n7. 保存数据...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为numpy格式
    np.save(output_path, token_array)
    
    # 也保存为二进制格式
    binary_path = output_path.replace('.npy', '.bin')
    token_array.tofile(binary_path)
    
    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024
    binary_size_mb = Path(binary_path).stat().st_size / 1024 / 1024
    
    print(f"   保存完成:")
    print(f"     NumPy格式: {output_path} ({file_size_mb:.2f} MB)")
    print(f"     二进制格式: {binary_path} ({binary_size_mb:.2f} MB)")
    
    # 8. 返回统计信息
    return {
        'doc_count': total_docs,
        'error_count': total_errors,
        'total_tokens': len(token_array),
        'unique_tokens': unique_ids,
        'min_id': min_id,
        'max_id': max_id,
        'exceeds_uint16': max_id > 65535,
        'file_size_mb': file_size_mb,
        'processing_time': encode_time,
        'tokens_per_second': len(token_array) / encode_time,
        'batches_processed': len(doc_batches),
        'num_processes': num_processes
    }


def main():
    parser = argparse.ArgumentParser(description='简单并行编码数据集为token ID序列')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    parser.add_argument('--vocab_path', type=str, required=True, help='词汇表文件路径')
    parser.add_argument('--merges_path', type=str, required=True, help='合并规则文件路径')
    parser.add_argument('--special_token', type=str, default='<|endoftext|>', help='特殊token')
    parser.add_argument('--output_path', type=str, required=True, help='输出文件路径')
    parser.add_argument('--max_docs', type=int, default=None, help='最大文档数量')
    parser.add_argument('--num_processes', type=int, default=None, help='进程数量')
    parser.add_argument('--batch_size', type=int, default=1000, help='批次大小')
    
    args = parser.parse_args()
    
    # 执行编码
    stats = encode_dataset_simple_parallel(
        args.data_path,
        args.vocab_path,
        args.merges_path,
        args.special_token,
        args.output_path,
        args.max_docs,
        args.num_processes,
        args.batch_size
    )
    
    # 打印总结
    print(f"\n" + "=" * 60)
    print(f"简单并行编码完成总结:")
    print(f"  处理文档数: {stats['doc_count']}")
    print(f"  错误文档数: {stats['error_count']}")
    print(f"  总token数: {stats['total_tokens']:,}")
    print(f"  唯一token数: {stats['unique_tokens']:,}")
    print(f"  数据类型: uint16")
    print(f"  处理时间: {stats['processing_time']:.2f} 秒")
    print(f"  处理速度: {stats['tokens_per_second']:.0f} tokens/秒")
    print(f"  文件大小: {stats['file_size_mb']:.2f} MB")
    print(f"  使用进程数: {stats['num_processes']}")
    print(f"  处理批次: {stats['batches_processed']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
