#!/usr/bin/env python3
"""
Train a BPE tokenizer on OpenWebText dataset.

Usage:
    uv run python train_owt_tokenizer.py
    
Output:
    data/tokenizers/owt_vocab.json
    data/tokenizers/owt_merges.txt
"""

from cs336_basics.bpe import train_bpe
import json
import os
import time

def main():
    print("=" * 60)
    print("Training OpenWebText BPE Tokenizer")
    print("=" * 60)
    
    # Configuration
    input_path = "/data/OpenWebText_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    
    print(f"\nConfiguration:")
    print(f"  Input: {input_path}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Special tokens: {special_tokens}")
    print(f"\n⚠️  Note: This may take 20-40 minutes depending on CPU cores")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"\n❌ Error: Input file not found: {input_path}")
        print(f"   Please make sure the OpenWebText dataset is available.")
        return
    
    # Train tokenizer
    print(f"\n🚀 Training tokenizer...")
    start_time = time.time()
    
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    elapsed = time.time() - start_time
    print(f"✅ Training completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    # Create output directory
    os.makedirs("data/tokenizers", exist_ok=True)
    
    # Save vocabulary
    vocab_path = "data/tokenizers/owt_vocab.json"
    print(f"\n💾 Saving vocabulary to {vocab_path}...")
    with open(vocab_path, "w") as f:
        json_vocab = {str(k): list(v) for k, v in vocab.items()}
        json.dump(json_vocab, f, indent=2)
    
    # Save merges
    merges_path = "data/tokenizers/owt_merges.txt"
    print(f"💾 Saving merges to {merges_path}...")
    with open(merges_path, "w") as f:
        for a, b in merges:
            f.write(f"{a!r} {b!r}\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Tokenizer Training Complete!")
    print("=" * 60)
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Number of merges: {len(merges)}")
    print(f"  Training time: {elapsed/60:.1f} minutes")
    print(f"\nOutput files:")
    print(f"  - {vocab_path}")
    print(f"  - {merges_path}")
    print(f"\nFirst 5 merges:")
    for i, (a, b) in enumerate(merges[:5]):
        print(f"  {i}: {a!r} + {b!r} → {a+b!r}")
    print("\nNext steps:")
    print("  1. Run: uv run python encode_owt.py")
    print("  2. Use the tokenizer in your training code")
    print("=" * 60)

if __name__ == "__main__":
    main()

