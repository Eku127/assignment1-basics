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

# Import Tokenizer and GPT2_PRETOKENIZER_PATTERN immediately (needed for generation)
from cs336_basics.bpe.tokenizer import Tokenizer
from cs336_basics.bpe.utils import GPT2_PRETOKENIZER_PATTERN

# Lazy import train_bpe to avoid importing training dependencies when only using Tokenizer
# This allows generate.py to work without requiring regex module if only Tokenizer is used
def _lazy_import_train_bpe():
    """Lazy import of train_bpe to avoid importing training dependencies unnecessarily."""
    from cs336_basics.bpe.training import train_bpe
    return train_bpe

# Create a lazy loader for train_bpe
class _LazyTrainBPE:
    """Lazy loader for train_bpe function."""
    def __call__(self, *args, **kwargs):
        train_bpe = _lazy_import_train_bpe()
        return train_bpe(*args, **kwargs)
    
    def __getattr__(self, name):
        # Allow access to attributes of the actual function if needed
        train_bpe = _lazy_import_train_bpe()
        return getattr(train_bpe, name)

# Export train_bpe as a lazy loader
train_bpe = _LazyTrainBPE()

__all__ = [
    'train_bpe',
    'Tokenizer',
    'GPT2_PRETOKENIZER_PATTERN',
]

