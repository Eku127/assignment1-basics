#!/usr/bin/env python3
"""
Encode OpenWebText dataset using trained BPE tokenizer.

Usage:
    uv run python encode_owt.py
    
Prerequisites:
    - Train tokenizer first: uv run python train_owt_tokenizer.py
    
Output:
    data/encoded/owt_train.npy
    data/encoded/owt_val.npy
"""

from cs336_basics.bpe import Tokenizer
import numpy as np
import os
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path


def encode_documents_worker(args):
    """
    Worker function: encode a batch of documents.
    
    Args:
        args: (documents, vocab_path, merges_path, special_token, worker_id)
        
    Returns:
        (worker_id, token_ids, doc_count, error_count)
    """
    documents, vocab_path, merges_path, special_token, worker_id = args
    
    try:
        # Create tokenizer in each worker process
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
                token_ids = tokenizer.encode(doc)
                all_token_ids.extend(token_ids)
                doc_count += 1
            except Exception:
                error_count += 1
                continue
        
        return (worker_id, all_token_ids, doc_count, error_count)
        
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        return (worker_id, [], 0, len(documents))


def load_documents_batches(data_path: str, batch_size: int = 2000):
    """
    Load documents and split into batches.
    
    Args:
        data_path: Path to data file
        batch_size: Batch size for processing (larger for OWT)
        
    Returns:
        List of document batches
    """
    print(f"  📖 Reading file: {data_path}")
    
    documents = []
    current_doc = ""
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="  Loading", unit=" lines"):
            # Check if line contains <|endoftext|>
            if "<|endoftext|>" in line:
                parts = line.split("<|endoftext|>")
                
                # Add content before <|endoftext|> to current document
                if parts[0].strip():
                    current_doc += parts[0]
                
                # Save current document if not empty
                if current_doc.strip():
                    documents.append(current_doc.strip())
                    current_doc = ""
                
                # Start new document with content after <|endoftext|>
                if len(parts) > 1 and parts[1].strip():
                    current_doc = parts[1]
            else:
                # Regular line, add to current document
                current_doc += line
    
    # Handle last document
    if current_doc.strip():
        documents.append(current_doc.strip())
    
    # Split into batches
    batches = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batches.append(batch)
    
    print(f"  ✅ Loaded {len(documents):,} documents → {len(batches)} batches")
    return batches


def encode_file_parallel(tokenizer_paths, input_path, output_path, num_processes=None, batch_size=2000):
    """
    Encode a text file using parallel processing and save as numpy array.
    
    Args:
        tokenizer_paths: (vocab_path, merges_path, special_token)
        input_path: Input text file path
        output_path: Output .npy file path
        num_processes: Number of processes (None = auto)
        batch_size: Batch size for processing
        
    Returns:
        Encoding statistics dict
    """
    vocab_path, merges_path, special_token = tokenizer_paths
    
    if num_processes is None:
        num_processes = min(cpu_count(), 16)  # Use more cores for OWT
    
    print(f"\n🚀 Encoding: {Path(input_path).name}")
    print(f"  Using {num_processes} processes, batch size: {batch_size}")
    
    # 1. Load document batches
    start_time = time.time()
    doc_batches = load_documents_batches(input_path, batch_size)
    load_time = time.time() - start_time
    
    # 2. Prepare worker arguments
    worker_args = []
    for i, batch in enumerate(doc_batches):
        worker_args.append((batch, vocab_path, merges_path, special_token, i))
    
    # 3. Parallel encoding
    print(f"  🔄 Encoding documents...")
    encode_start = time.time()
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(encode_documents_worker, worker_args),
            total=len(worker_args),
            desc="  Progress",
            unit=" batch"
        ))
    
    encode_time = time.time() - encode_start
    
    # 4. Merge results
    print(f"  🔗 Merging results...")
    all_token_ids = []
    total_docs = 0
    total_errors = 0
    
    # Sort results by worker_id to maintain order
    results.sort(key=lambda x: x[0])
    
    for worker_id, token_ids, doc_count, error_count in results:
        all_token_ids.extend(token_ids)
        total_docs += doc_count
        total_errors += error_count
    
    # 5. Convert to numpy array
    token_array = np.array(all_token_ids, dtype=np.uint16)
    
    # 6. Analyze results
    min_id = np.min(token_array)
    max_id = np.max(token_array)
    unique_ids = len(np.unique(token_array))
    
    # 7. Save data
    print(f"  💾 Saving...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, token_array)
    
    # Also save binary format
    binary_path = output_path.replace('.npy', '.bin')
    token_array.tofile(binary_path)
    
    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024
    binary_size_mb = Path(binary_path).stat().st_size / 1024 / 1024
    
    total_time = load_time + encode_time
    
    # Print summary
    print(f"\n  ✅ Encoding Complete!")
    print(f"     Documents: {total_docs:,} ({total_errors} errors)")
    print(f"     Tokens: {len(token_array):,}")
    print(f"     Token ID range: {min_id} - {max_id}")
    print(f"     Unique tokens: {unique_ids:,}")
    print(f"     File size: {file_size_mb:.2f} MB (numpy), {binary_size_mb:.2f} MB (binary)")
    print(f"     Time: {total_time:.1f}s ({len(token_array)/total_time/1e6:.1f}M tokens/sec)")
    
    return {
        'doc_count': total_docs,
        'error_count': total_errors,
        'total_tokens': len(token_array),
        'unique_tokens': unique_ids,
        'min_id': int(min_id),
        'max_id': int(max_id),
        'file_size_mb': file_size_mb,
        'processing_time': total_time,
        'tokens_per_second': len(token_array) / total_time
    }

def main():
    print("=" * 60)
    print("Encoding OpenWebText Dataset (Parallel)")
    print("=" * 60)
    
    # Get paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    vocab_path = os.path.join(project_root, "data/tokenizers/owt_vocab.json")
    merges_path = os.path.join(project_root, "data/tokenizers/owt_merges.txt")
    special_token = "<|endoftext|>"
    
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print(f"\n❌ Error: Tokenizer not found!")
        print(f"   Expected files:")
        print(f"     - data/tokenizers/owt_vocab.json")
        print(f"     - data/tokenizers/owt_merges.txt")
        print(f"\n   Please train the tokenizer first:")
        print(f"     uv run python cs336_basics/bpe/applications/train_owt_tokenizer.py")
        return
    
    # Verify tokenizer
    print(f"\n📂 Verifying tokenizer...")
    print(f"  Vocabulary: {vocab_path}")
    print(f"  Merges: {merges_path}")
    
    # Quick load to get vocab size
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=[special_token]
    )
    vocab_size = len(tokenizer.vocab)
    print(f"  ✅ Tokenizer ready (vocab_size={vocab_size})")
    
    # Create output directory
    encoded_dir = os.path.join(project_root, "data/encoded")
    os.makedirs(encoded_dir, exist_ok=True)
    
    # Tokenizer paths for workers
    tokenizer_paths = (vocab_path, merges_path, special_token)
    
    # Encode training set
    train_stats = encode_file_parallel(
        tokenizer_paths,
        os.path.join(project_root, "data/owt_train.txt"),
        os.path.join(encoded_dir, "owt_train.npy"),
        num_processes=None,  # Auto-detect (will use up to 16)
        batch_size=2000  # Larger batch for OWT
    )
    
    # Encode validation set
    val_stats = encode_file_parallel(
        tokenizer_paths,
        os.path.join(project_root, "data/owt_valid.txt"),
        os.path.join(encoded_dir, "owt_valid.npy"),
        num_processes=None,  # Auto-detect
        batch_size=2000
    )
    
    # Print final summary
    print("\n" + "=" * 60)
    print("✅ All Encoding Complete!")
    print("=" * 60)
    print(f"\n📊 Training Set:")
    print(f"  Documents: {train_stats['doc_count']:,}")
    print(f"  Tokens: {train_stats['total_tokens']:,}")
    print(f"  Speed: {train_stats['tokens_per_second']/1e6:.1f}M tokens/sec")
    print(f"  File: owt_train.npy ({train_stats['file_size_mb']:.2f} MB)")
    
    print(f"\n📊 Validation Set:")
    print(f"  Documents: {val_stats['doc_count']:,}")
    print(f"  Tokens: {val_stats['total_tokens']:,}")
    print(f"  Speed: {val_stats['tokens_per_second']/1e6:.1f}M tokens/sec")
    print(f"  File: owt_valid.npy ({val_stats['file_size_mb']:.2f} MB)")
    
    print(f"\n📈 Overall:")
    print(f"  Total tokens: {train_stats['total_tokens'] + val_stats['total_tokens']:,}")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Token coverage: {max(train_stats['unique_tokens'], val_stats['unique_tokens']):,} / {vocab_size:,}")
    
    print(f"\n📁 Output files:")
    print(f"  - data/encoded/owt_train.npy + .bin")
    print(f"  - data/encoded/owt_valid.npy + .bin")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Load in training code:")
    print(f'     train_data = np.load("data/encoded/owt_train.npy", mmap_mode="r")')
    print(f"  2. Use memory-mapped mode for large files to save RAM")
    print(f"  3. Binary files (.bin) can be used with np.memmap for faster loading")
    print("=" * 60)

if __name__ == "__main__":
    main()

