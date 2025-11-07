"""
BPE Tokenizer Utility Functions

This module contains helper functions for BPE tokenization, including:
- Pre-tokenization using regex patterns
- Special token handling
- Vocabulary initialization
- Optimized parallel pre-tokenization
"""

import re
from collections import defaultdict
from typing import Iterable
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import regex as regex_mod  # Faster regex engine with \p classes


# GPT-2 style pre-tokenizer pattern from the handout
GPT2_PRETOKENIZER_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split text on special tokens, removing the special tokens themselves.
    
    This ensures no BPE merges occur across special token boundaries.
    
    Args:
        text: Input text to split
        special_tokens: List of special token strings
    
    Returns:
        List of text segments (without special tokens)
    
    Example:
        >>> text = "Hello<|endoftext|>World"
        >>> split_on_special_tokens(text, ["<|endoftext|>"])
        ['Hello', 'World']
    """
    if not special_tokens:
        return [text]
    
    # Escape tokens and join with | for alternation
    escaped = [re.escape(tok) for tok in special_tokens]
    pattern = re.compile("(" + "|".join(escaped) + ")")
    
    # Split and drop the special tokens themselves per assignment guidance
    parts = pattern.split(text)
    
    # Keep only non-special segments
    return [seg for seg in parts if seg and seg not in special_tokens]


def _pretokenize_chunk_batch(chunks: list[str]) -> dict[tuple[int, ...], int]:
    """
    Batch pre-tokenize multiple document chunks to reduce inter-process communication.
    
    This is a worker function for parallel pre-tokenization.
    
    Args:
        chunks: List of text chunks to process
    
    Returns:
        Frequency dictionary of byte tuples
    """
    pat = regex_mod.compile(GPT2_PRETOKENIZER_PATTERN)
    freq: dict[tuple[int, ...], int] = defaultdict(int)
    
    for chunk in chunks:
        for m in pat.finditer(chunk):
            token = m.group(0)
            bs = token.encode("utf-8")
            freq[tuple(bs)] += 1
    
    return dict(freq)


def pretokenize(
    text_iter: Iterable[str],
    use_multiprocessing: bool = True,
    num_processes: int = None,
    show_progress: bool = True
) -> dict[tuple[int, ...], int]:
    """
    Pre-tokenize text using GPT-2 style regex pattern with optional parallelization.
    
    Automatically chooses between single-threaded and multi-process based on data size.
    For large corpora (>10k segments), uses multiprocessing for significant speedup.
    
    Args:
        text_iter: Iterable of text chunks to pre-tokenize
        use_multiprocessing: Whether to use multiprocessing (default: True)
        num_processes: Number of processes (None = auto-detect based on CPU count)
        show_progress: Whether to show tqdm progress bar (default: True)
    
    Returns:
        Dictionary mapping byte tuples to their frequency counts
    
    Example:
        >>> text = ["hello world", "foo bar"]
        >>> freq = pretokenize(text)
        >>> # freq will contain byte tuples for "hello", " world", "foo", " bar"
    """
    # Convert to list if needed for length check
    segments = list(text_iter) if not isinstance(text_iter, list) else text_iter
    
    # Use single-threaded for small inputs or if multiprocessing is disabled
    if not use_multiprocessing or len(segments) < 10000:
        return _pretokenize_single_thread(segments, show_progress=show_progress)
    
    # Use multiprocessing for large inputs
    return _pretokenize_parallel(segments, num_processes=num_processes, show_progress=show_progress)


def _pretokenize_single_thread(
    segments: list[str],
    show_progress: bool = True
) -> dict[tuple[int, ...], int]:
    """
    Single-threaded pre-tokenization with optional progress bar.
    
    Args:
        segments: List of text segments
        show_progress: Whether to show progress bar
    
    Returns:
        Frequency dictionary of byte tuples
    """
    pat = regex_mod.compile(GPT2_PRETOKENIZER_PATTERN)
    freq: dict[tuple[int, ...], int] = defaultdict(int)
    
    iterator = tqdm(segments, desc="Pre-tokenizing", unit="segment") if show_progress else segments
    
    for chunk in iterator:
        for m in pat.finditer(chunk):
            token = m.group(0)
            bs = token.encode("utf-8")
            freq[tuple(bs)] += 1
    
    return dict(freq)


def _pretokenize_parallel(
    segments: list[str],
    num_processes: int = None,
    show_progress: bool = True
) -> dict[tuple[int, ...], int]:
    """
    Parallel pre-tokenization using multiprocessing.
    
    Divides segments into batches and processes them in parallel to reduce
    inter-process communication overhead.
    
    Args:
        segments: List of text segments
        num_processes: Number of processes (None = auto-detect)
        show_progress: Whether to show progress bar
    
    Returns:
        Frequency dictionary of byte tuples
    """
    if num_processes is None:
        num_processes = min(cpu_count(), len(segments))
    
    # Divide segments into batches to reduce inter-process communication
    # Use 4x the number of processes to ensure good load balancing
    batch_size = max(10, len(segments) // (num_processes * 4))
    batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
    
    if show_progress:
        print(f"   Using {num_processes} processes for {len(batches)} batches...")
    
    # Parallel processing with progress bar
    with Pool(processes=num_processes) as pool:
        if show_progress:
            batch_freqs = list(tqdm(
                pool.imap(_pretokenize_chunk_batch, batches),
                total=len(batches),
                desc="Pre-tokenizing",
                unit="batch"
            ))
        else:
            batch_freqs = pool.map(_pretokenize_chunk_batch, batches)
    
    # Merge all results
    total_freq: dict[tuple[int, ...], int] = defaultdict(int)
    for batch_freq in batch_freqs:
        for token_tuple, count in batch_freq.items():
            total_freq[token_tuple] += count
    
    return dict(total_freq)


def pretokenize_string(text: str) -> list[str]:
    """
    Pre-tokenize a single string using GPT-2 style regex pattern.
    
    Args:
        text: Input text string
    
    Returns:
        List of pre-tokenized strings
    
    Example:
        >>> pretokenize_string("hello world")
        ['hello', ' world']
    """
    pat = regex_mod.compile(GPT2_PRETOKENIZER_PATTERN)
    return [m.group(0) for m in pat.finditer(text)]


def build_initial_vocab(special_tokens: list[str]) -> tuple[dict[int, bytes], dict[bytes, int]]:
    """
    Build initial vocabulary with 256 byte tokens and special tokens.
    
    Args:
        special_tokens: List of special token strings
    
    Returns:
        Tuple of (id_to_bytes, bytes_to_id) mappings
    
    Example:
        >>> id_to_bytes, bytes_to_id = build_initial_vocab(["<|endoftext|>"])
        >>> len(id_to_bytes)  # 256 bytes + 1 special token
        257
    """
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

