'''

uv run python cs336_basics/bpe/tokenizer_experiments/compute_compression.py \
  --data_path ./data/TinyStoriesV2-GPT4-train.txt \
  --vocab_path cs336_basics/bpe/bpe_results/tinystories_vocab.json \
  --merges_path cs336_basics/bpe/bpe_results/tinystories_merges.txt \
  --special_token "<|endoftext|>" \
  --num_docs 10 \
  --random_sample \
  --seed 42
  '''


import argparse
import os
import random
from typing import Iterable

from cs336_basics.tokenizer_improved import Tokenizer


def read_documents(path: str, delimiter: str, max_docs: int | None, random_sample: bool, seed: int) -> list[str]:
    """Read text file and split by delimiter token into documents.

    Returns at most max_docs documents if specified; otherwise returns all.
    If random_sample is True, sample without replacement using the given seed.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    docs = text.split(delimiter)
    # Remove possible empty docs
    docs = [d for d in docs if d]
    if max_docs is None or max_docs >= len(docs):
        selected = docs
    else:
        if random_sample:
            rng = random.Random(seed)
            indices = list(range(len(docs)))
            rng.shuffle(indices)
            selected = [docs[i] for i in indices[:max_docs]]
        else:
            selected = docs[:max_docs]
    # Re-attach delimiter at doc boundaries so that byte counts match original when desired
    # For compression ratio over content, we count just doc content (without delimiter token itself).
    return selected


def iter_documents(docs: list[str]) -> Iterable[str]:
    for d in docs:
        yield d


def compute_compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> tuple[int, int, float, int]:
    """Compute total bytes, total tokens, and bytes/token."""
    # Total bytes counted as UTF-8 bytes of concatenated docs
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)

    # Total tokens using streaming encoding for memory efficiency
    total_tokens = 0
    failed_docs = 0
    
    for doc in docs:
        try:
            # Try to encode this document
            doc_tokens = list(tokenizer.encode_iterable([doc]))
            total_tokens += len(doc_tokens)
        except ValueError as e:
            print(f"Warning: Failed to encode document: {e}")
            failed_docs += 1
            continue

    ratio = float(total_bytes) / total_tokens if total_tokens > 0 else float("inf")
    return total_bytes, total_tokens, ratio, failed_docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute bytes-per-token compression ratio for a dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input text file (e.g., TinyStories train)")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab json")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to merges txt")
    parser.add_argument("--special_token", type=str, default="<|endoftext|>", help="Special token delimiter present in data")
    parser.add_argument("--num_docs", type=int, default=10, help="Number of documents to sample")
    parser.add_argument("--random_sample", action="store_true", help="Sample documents randomly instead of taking the first N")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=[args.special_token])

    # Read docs
    docs = read_documents(
        path=args.data_path,
        delimiter=args.special_token,
        max_docs=args.num_docs,
        random_sample=args.random_sample,
        seed=args.seed,
    )

    total_bytes, total_tokens, ratio, failed_docs = compute_compression_ratio(tokenizer, docs)

    print("Documents:", len(docs))
    print("Failed documents:", failed_docs)
    print("Successful documents:", len(docs) - failed_docs)
    print("Total bytes:", total_bytes)
    print("Total tokens:", total_tokens)
    print("Bytes per token:", ratio)


if __name__ == "__main__":
    main()


