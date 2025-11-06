"""
Cross-entropy loss implementation for language modeling.

This module implements numerically stable cross-entropy loss computation
for Transformer language models.
"""

import torch
import torch.nn as nn


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss for language modeling.
    
    Implements:
        ℓ = -log p(x_{i+1} | x_{1:i})
        where p(x_{i+1} | x_{1:i}) = softmax(logits)[x_{i+1}]
    
    Uses numerically stable computation:
    - Subtracts max for stability
    - Cancels log and exp where possible
    
    Args:
        logits: Predicted logits from model, shape (..., vocab_size)
                Typically (batch_size, seq_len, vocab_size)
        targets: Target token IDs, shape (...)
                 Typically (batch_size, seq_len)
                 Should be integer type (long)
    
    Returns:
        loss: Scalar tensor, average cross-entropy loss across all positions
    
    Example:
        >>> logits = model(input_ids)  # (batch, seq_len, vocab_size)
        >>> targets = target_ids       # (batch, seq_len)
        >>> loss = cross_entropy(logits, targets)
        >>> loss.backward()
    
    Implementation Notes:
    - For numerical stability:
        1. Subtract max(logits) before exp
        2. Use log-sum-exp trick: log(sum(exp(x))) = max + log(sum(exp(x - max)))
        3. Cancel log and exp: log(exp(x)) = x
    - Returns average loss across all batch and sequence dimensions
    """
    # TODO: Implement numerically stable cross-entropy loss
    raise NotImplementedError("Cross-entropy loss not yet implemented")


def perplexity(losses: torch.Tensor) -> float:
    """
    Compute perplexity from cross-entropy losses.
    
    perplexity = exp((1/m) * Σ ℓ_i)
    
    Args:
        losses: Cross-entropy losses for each position, shape (m,)
    
    Returns:
        perplexity: Perplexity value (scalar)
    """
    return torch.exp(losses.mean()).item()

