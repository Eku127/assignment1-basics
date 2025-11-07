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

def encode_file(tokenizer, input_path, output_path):
    """Encode a text file and save as numpy array."""
    print(f"  Reading: {input_path}")
    
    tokens = []
    start_time = time.time()
    
    with open(input_path, "r", encoding="utf-8") as f:
        for i, token_id in enumerate(tokenizer.encode_iterable(f)):
            tokens.append(token_id)
            if (i + 1) % 10_000_000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed / 1e6
                print(f"    Processed {i+1:,} tokens ({rate:.1f}M tokens/sec)")
    
    # Save as numpy array
    token_array = np.array(tokens, dtype=np.uint16)
    np.save(output_path, token_array)
    
    elapsed = time.time() - start_time
    rate = len(tokens) / elapsed / 1e6
    
    print(f"  ✅ Saved: {output_path}")
    print(f"     Tokens: {len(tokens):,}")
    print(f"     Size: {token_array.nbytes / 1e6:.1f} MB")
    print(f"     Time: {elapsed:.1f}s ({rate:.1f}M tokens/sec)")
    
    return len(tokens)

def main():
    print("=" * 60)
    print("Encoding OpenWebText Dataset")
    print("=" * 60)
    
    # Check if tokenizer exists
    vocab_path = "data/tokenizers/owt_vocab.json"
    merges_path = "data/tokenizers/owt_merges.txt"
    
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print(f"\n❌ Error: Tokenizer not found!")
        print(f"   Expected files:")
        print(f"     - {vocab_path}")
        print(f"     - {merges_path}")
        print(f"\n   Please train the tokenizer first:")
        print(f"     uv run python train_owt_tokenizer.py")
        return
    
    # Load tokenizer
    print(f"\n📂 Loading tokenizer...")
    print(f"  Vocabulary: {vocab_path}")
    print(f"  Merges: {merges_path}")
    
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"]
    )
    
    print(f"  ✅ Loaded tokenizer (vocab_size={len(tokenizer.vocab)})")
    
    # Create output directory
    os.makedirs("data/encoded", exist_ok=True)
    
    # Encode training set
    print(f"\n🚀 Encoding training set (this may take 1-2 hours)...")
    train_tokens = encode_file(
        tokenizer,
        "/data/OpenWebText_train.txt",
        "data/encoded/owt_train.npy"
    )
    
    # Encode validation set
    print(f"\n🚀 Encoding validation set...")
    val_tokens = encode_file(
        tokenizer,
        "/data/OpenWebText_val.txt",
        "data/encoded/owt_val.npy"
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Encoding Complete!")
    print("=" * 60)
    print(f"  Training tokens: {train_tokens:,}")
    print(f"  Validation tokens: {val_tokens:,}")
    print(f"  Vocabulary size: {len(tokenizer.vocab)}")
    print(f"\nOutput files:")
    print(f"  - data/encoded/owt_train.npy")
    print(f"  - data/encoded/owt_val.npy")
    print("\nNext steps:")
    print("  1. Load data in training code:")
    print('     train_data = np.load("data/encoded/owt_train.npy", mmap_mode="r")')
    print("  2. Use memory-mapped mode for large files to save RAM")
    print("=" * 60)

if __name__ == "__main__":
    main()

