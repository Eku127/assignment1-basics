"""
BPE (Byte-Pair Encoding) Tokenizer Module

This module implements a byte-level BPE tokenizer following the CS336 Assignment 1 specification.

Components:
- Training: train_bpe() function for learning BPE merges
- Tokenizer: Tokenizer class for encoding/decoding text
- Utils: Helper functions for pre-tokenization and special token handling

Main exports:
- train_bpe: Train a BPE tokenizer on a corpus
- Tokenizer: BPE tokenizer class
- GPT2_PRETOKENIZER_PATTERN: Regex pattern for pre-tokenization
"""

from cs336_basics.bpe.training import train_bpe
from cs336_basics.bpe.tokenizer import Tokenizer
from cs336_basics.bpe.utils import GPT2_PRETOKENIZER_PATTERN

__all__ = [
    'train_bpe',
    'Tokenizer',
    'GPT2_PRETOKENIZER_PATTERN',
]

