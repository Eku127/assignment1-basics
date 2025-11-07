"""
BPE Tokenizer Utility Functions

This module contains helper functions for BPE tokenization, including:
- Pre-tokenization using regex patterns
- Special token handling
- Vocabulary initialization
"""

import re
from collections import defaultdict
from typing import Iterable

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


def pretokenize(text_iter: Iterable[str]) -> dict[tuple[int, ...], int]:
    """
    Pre-tokenize text using GPT-2 style regex pattern.
    
    Returns frequency dict of pre-tokens represented as UTF-8 byte tuples.
    
    Args:
        text_iter: Iterable of text chunks to pre-tokenize
    
    Returns:
        Dictionary mapping byte tuples to their frequency counts
    
    Example:
        >>> text = ["hello world"]
        >>> freq = pretokenize(text)
        >>> # freq will contain byte tuples for "hello" and " world"
    """
    pat = regex_mod.compile(GPT2_PRETOKENIZER_PATTERN)
    freq: dict[tuple[int, ...], int] = defaultdict(int)
    
    for chunk in text_iter:
        for m in pat.finditer(chunk):
            token = m.group(0)
            bs = token.encode("utf-8")
            freq[tuple(bs)] += 1
    
    return freq


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

