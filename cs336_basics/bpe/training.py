"""
BPE Tokenizer Training

This module implements the BPE training algorithm following the assignment specification:
- Byte-level BPE with UTF-8 encoding
- Regex-based pre-tokenization (GPT-2 style)
- No merges across special token boundaries
- Tie-breaking by lexicographically greater pair when counts tie
"""

import os
from collections import Counter, defaultdict
from typing import Union

from cs336_basics.bpe.utils import (
    split_on_special_tokens,
    pretokenize,
    build_initial_vocab,
)


def train_bpe(
    input_path: Union[str, bytes, "os.PathLike[str]"],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the given corpus.
    
    Algorithm:
    1. Initialize vocabulary with 256 byte tokens + special tokens
    2. Split corpus on special tokens (no merges across boundaries)
    3. Pre-tokenize using GPT-2 regex pattern
    4. Iteratively merge the most frequent pair of bytes
    5. Tie-break by lexicographically greater pair when counts tie
    
    Args:
        input_path: Path to text file with training data
        vocab_size: Maximum final vocabulary size (including bytes and special tokens)
        special_tokens: List of special token strings to add to vocabulary
    
    Returns:
        Tuple of (vocab, merges):
            - vocab: dict[int, bytes] mapping token IDs to byte sequences
            - merges: list[tuple[bytes, bytes]] ordered list of BPE merges
    
    Example:
        >>> vocab, merges = train_bpe("corpus.txt", vocab_size=1000, special_tokens=["<|endoftext|>"])
        >>> len(vocab)
        1000
        >>> len(merges)
        743  # 1000 - 256 - 1 (special token)
    """
    # 1. Read text and split on special tokens
    import time
    print(f"📖 Reading corpus from {input_path}...")
    start_read = time.time()
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    read_time = time.time() - start_read
    print(f"✅ Read {len(text):,} characters in {read_time:.1f}s ({len(text)/1e6/read_time:.1f} MB/s)")
    
    # Remove/split on special tokens so no merges cross them
    print(f"✂️  Splitting on special tokens: {special_tokens}...")
    start_split = time.time()
    segments = split_on_special_tokens(text, special_tokens or [])
    split_time = time.time() - start_split
    print(f"✅ Split into {len(segments):,} segments in {split_time:.1f}s")
    
    # Free memory
    del text
    
    # 2. Pre-tokenize segments and count frequency
    print(f"🔤 Pre-tokenizing with GPT-2 regex (this may take a while for large corpora)...")
    start_pretok = time.time()
    freq = pretokenize(segments)
    pretok_time = time.time() - start_pretok
    print(f"✅ Pre-tokenization complete in {pretok_time:.1f}s ({pretok_time/60:.1f} minutes)")
    print(f"   Found {len(freq):,} unique byte sequences")
    
    # Free memory
    del segments
    
    # 3. Initialize vocabulary: 256 bytes + special tokens
    print(f"📚 Initializing vocabulary...")
    id_to_bytes, bytes_to_id = build_initial_vocab(special_tokens or [])
    next_id = max(id_to_bytes) + 1 if id_to_bytes else 0
    print(f"✅ Initial vocabulary: {len(id_to_bytes)} tokens (256 bytes + {len(special_tokens or [])} special tokens)")
    
    # 4. Store corpus as list of (list[int], count)
    print(f"🔢 Building corpus ID representation...")
    start_corpus = time.time()
    corpus_ids: list[list[int]] = []
    corpus_counts: list[int] = []
    
    for byte_tuple, count in freq.items():
        ids = [bytes_to_id[bytes([b])] for b in byte_tuple]
        corpus_ids.append(ids)
        corpus_counts.append(count)
    
    corpus_time = time.time() - start_corpus
    print(f"✅ Corpus built in {corpus_time:.1f}s: {len(corpus_ids):,} unique sequences")
    
    merges: list[tuple[bytes, bytes]] = []
    initial_vocab_size = len(id_to_bytes)
    max_merges = max(0, vocab_size - initial_vocab_size)
    
    # Helper function to count pairs in a sequence
    def pairs_for_ids(ids: list[int]) -> Counter[tuple[int, int]]:
        """Count all adjacent pairs in an ID sequence."""
        ctr: Counter[tuple[int, int]] = Counter()
        if len(ids) < 2:
            return ctr
        for a, b in zip(ids, ids[1:]):
            ctr[(a, b)] += 1
        return ctr
    
    # Build per-word pair counters and global counts and reverse index
    print(f"🔗 Building pair statistics...")
    start_pairs = time.time()
    word_pair_counters: list[Counter[tuple[int, int]]] = []
    total_pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    
    for i, ids in enumerate(corpus_ids):
        ctr = pairs_for_ids(ids)
        word_pair_counters.append(ctr)
        c = corpus_counts[i]
        
        for pair, k in ctr.items():
            total_pair_counts[pair] += k * c
            pair_to_words[pair].add(i)
    
    pairs_time = time.time() - start_pairs
    print(f"✅ Pair statistics built in {pairs_time:.1f}s: {len(total_pair_counts):,} unique pairs")
    
    def merge_pair(a: int, b: int, new_token: int) -> None:
        """
        Merge pair (a, b) into new_token across all affected words.
        
        Incrementally updates:
        - corpus_ids: Replace (a, b) with new_token
        - word_pair_counters: Recompute pairs for affected words
        - total_pair_counts: Update global pair counts
        - pair_to_words: Update reverse index
        """
        target = (a, b)
        affected_words = list(pair_to_words.get(target, set()))
        
        for i in affected_words:
            ids = corpus_ids[i]
            if len(ids) < 2:
                continue
            
            # Remove old pair counts for this word from global
            old_pairs = word_pair_counters[i]
            count_multiplier = corpus_counts[i]
            
            for pair, k in old_pairs.items():
                total_pair_counts[pair] -= k * count_multiplier
                if total_pair_counts[pair] <= 0:
                    total_pair_counts.pop(pair, None)
            
            # Actually perform merge on this word
            j = 0
            out: list[int] = []
            
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == a and ids[j + 1] == b:
                    out.append(new_token)
                    j += 2
                else:
                    out.append(ids[j])
                    j += 1
            
            corpus_ids[i] = out
            
            # Recompute pairs and update indexes
            new_ctr = pairs_for_ids(out)
            word_pair_counters[i] = new_ctr
            
            # Remove i from pairs that disappeared
            for pair in old_pairs.keys():
                if pair not in new_ctr:
                    s = pair_to_words.get(pair)
                    if s is not None:
                        s.discard(i)
                        if not s:
                            pair_to_words.pop(pair, None)
            
            # Add i to new/updated pairs and update global counts
            for pair, k in new_ctr.items():
                total_pair_counts[pair] = total_pair_counts.get(pair, 0) + k * count_multiplier
                s = pair_to_words.get(pair)
                if s is None:
                    pair_to_words[pair] = {i}
                else:
                    s.add(i)
    
    # 5. Main merge loop
    import sys
    
    print(f"\n🔄 Starting BPE training: {max_merges} merges needed")
    print(f"   Initial vocab size: {initial_vocab_size}")
    print(f"   Target vocab size: {vocab_size}")
    print(f"   Progress updates every 1000 merges\n")
    
    start_time = time.time()
    last_update_time = start_time
    
    for merge_idx in range(max_merges):
        if not total_pair_counts:
            print(f"\n⚠️  No more pairs to merge at iteration {merge_idx}")
            break
        
        # Progress reporting every 1000 merges
        if merge_idx > 0 and merge_idx % 1000 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            elapsed_since_update = current_time - last_update_time
            rate = 1000.0 / elapsed_since_update if elapsed_since_update > 0 else 0
            progress_pct = (merge_idx / max_merges) * 100
            current_vocab_size = initial_vocab_size + merge_idx
            
            # Estimate time remaining
            if merge_idx > 0:
                avg_time_per_merge = elapsed / merge_idx
                remaining_merges = max_merges - merge_idx
                eta_seconds = avg_time_per_merge * remaining_merges
                eta_minutes = eta_seconds / 60
                eta_str = f"{int(eta_minutes)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = "calculating..."
            
            print(f"   [{merge_idx:5d}/{max_merges}] {progress_pct:5.1f}% | "
                  f"Vocab: {current_vocab_size:5d}/{vocab_size} | "
                  f"Rate: {rate:5.1f} merges/s | "
                  f"ETA: {eta_str}")
            sys.stdout.flush()
            last_update_time = current_time
        
        # Select (count, pair) maximum; tie-break lexicographically by bytes
        def pair_key(p: tuple[int, int]):
            a, b = p
            return (total_pair_counts[p], id_to_bytes[a], id_to_bytes[b])
        
        best_pair = max(total_pair_counts.keys(), key=pair_key)
        a, b = best_pair
        
        # Create new token representing bytes concatenation
        bytes_a = id_to_bytes[a]
        bytes_b = id_to_bytes[b]
        new_bytes = bytes_a + bytes_b
        new_id = next_id
        next_id += 1
        
        id_to_bytes[new_id] = new_bytes
        
        # Update only affected words and global counts
        merge_pair(a, b, new_id)
        merges.append((bytes_a, bytes_b))
        
        if len(id_to_bytes) >= vocab_size:
            break
    
    # Final progress update
    total_time = time.time() - start_time
    final_vocab_size = len(id_to_bytes)
    print(f"\n✅ BPE training completed!")
    print(f"   Total merges: {len(merges)}")
    print(f"   Final vocab size: {final_vocab_size}")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    if len(merges) > 0:
        print(f"   Average: {total_time/len(merges):.3f}s per merge\n")
    
    # Trim vocab to requested size (it will already be exact unless early break)
    if len(id_to_bytes) > vocab_size:
        # Keep the lowest vocab_size ids
        items = sorted(id_to_bytes.items(), key=lambda x: x[0])[:vocab_size]
        id_to_bytes = {i: b for i, (orig_id, b) in enumerate(items)}
    
    return id_to_bytes, merges

