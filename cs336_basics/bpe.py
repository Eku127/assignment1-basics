from __future__ import annotations

import re
from collections import defaultdict, Counter
from typing import Iterable

import regex as regex_mod  # faster regex engine with \p classes


# GPT-2 style pre-tokenizer pattern from the handout
GPT2_PRETOKENIZER_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    # Escape tokens and join with | for alternation
    escaped = [re.escape(tok) for tok in special_tokens]
    pattern = re.compile("(" + "|".join(escaped) + ")")
    # Split and drop the special tokens themselves per assignment guidance
    parts = pattern.split(text)
    # Keep only non-special segments
    return [seg for seg in parts if seg and seg not in special_tokens]


def _pretokenize(text_iter: Iterable[str]) -> dict[tuple[int, ...], int]:
    """Return frequency dict of pre-tokens represented as UTF-8 byte tuples."""
    pat = regex_mod.compile(GPT2_PRETOKENIZER_PATTERN)
    freq: dict[tuple[int, ...], int] = defaultdict(int)
    for chunk in text_iter:
        for m in pat.finditer(chunk):
            token = m.group(0)
            bs = token.encode("utf-8")
            freq[tuple(bs)] += 1
    return freq


def _build_initial_vocab(special_tokens: list[str]) -> tuple[dict[int, bytes], dict[bytes, int]]:
    id_to_bytes: dict[int, bytes] = {}
    bytes_to_id: dict[bytes, int] = {}
    # 256 byte vocabulary first
    for b in range(256):
        bb = bytes([b])
        id_to_bytes[b] = bb
        bytes_to_id[bb] = b
    next_id = 256
    # Append special tokens (as raw bytes of their literal string)
    for tok in special_tokens:
        b = tok.encode("utf-8")
        id_to_bytes[next_id] = b
        bytes_to_id[b] = next_id
        next_id += 1
    return id_to_bytes, bytes_to_id


def train_bpe(
    input_path: str | bytes | "os.PathLike[str]",
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE on the given corpus path. Follows assignment spec:
    - Regex pre-tokenization (no merges across pre-token boundaries)
    - Remove/split on special tokens before pre-tokenization
    - Tie-break by lexicographically greater pair when counts tie
    """

    # 1. 读取文本并分割特殊token
    # 2. 预分词（使用GPT-2风格的正则表达式）
    # 3. 初始化词汇表（256个字节 + 特殊token）
    # 4. 迭代合并最频繁的字节对
    # 5. 返回词汇表和合并规则
    
    # Read whole file (small fixtures in tests)
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Remove/split on special tokens so no merges cross them
    segments = _split_on_special_tokens(text, special_tokens or [])

    # Pre-tokenize segments and count frequency
    freq = _pretokenize(segments)

    # Represent pre-tokens as lists of token ids (start with byte ids 0..255)
    id_to_bytes, bytes_to_id = _build_initial_vocab(special_tokens or [])
    next_id = max(id_to_bytes) + 1 if id_to_bytes else 0

    # Store corpus as list of (list[int], count)
    corpus_ids: list[list[int]] = []
    corpus_counts: list[int] = []
    for byte_tuple, count in freq.items():
        ids = [bytes_to_id[bytes([b])] for b in byte_tuple]
        corpus_ids.append(ids)
        corpus_counts.append(count)

    merges: list[tuple[bytes, bytes]] = []
    initial_vocab_size = len(id_to_bytes)
    max_merges = max(0, vocab_size - initial_vocab_size)

    def pairs_for_ids(ids: list[int]) -> Counter[tuple[int, int]]:
        ctr: Counter[tuple[int, int]] = Counter()
        if len(ids) < 2:
            return ctr
        for a, b in zip(ids, ids[1:]):
            ctr[(a, b)] += 1
        return ctr

    # Build per-word pair counters and global counts and reverse index
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

    def merge_pair(a: int, b: int, new_token: int) -> None:
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

    # Main merge loop
    for _ in range(max_merges):
        if not total_pair_counts:
            break
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

    # Trim vocab to requested size (it will already be exact unless early break)
    if len(id_to_bytes) > vocab_size:
        # Keep the lowest vocab_size ids (ensures ids set is {0..vocab_size-1} when initial build followed by sequential ids)
        # Remap ids densely 0..vocab_size-1 while preserving byte values and order
        items = sorted(id_to_bytes.items(), key=lambda x: x[0])[:vocab_size]
        id_to_bytes = {i: b for i, (orig_id, b) in enumerate(items)}

    return id_to_bytes, merges


