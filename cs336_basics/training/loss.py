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

    # Get the shape info
    # logits: (..., vocab_size), targets: (...)
    vocab_size = logits.size(-1)
    
    # Flatten batch dimensions if needed
    # Reshape to (N, vocab_size) and (N,) where N is the product of all batch dims
    original_shape = logits.shape[:-1]  # All dimensions except vocab_size
    logits_flat = logits.view(-1, vocab_size)  # (N, vocab_size)
    targets_flat = targets.view(-1)  # (N,)
    
    # Numerically stable cross-entropy computation
    # Cross-entropy: -log(softmax(logits)[target])
    #              = -log(exp(logits[target]) / sum(exp(logits)))
    #              = -logits[target] + log(sum(exp(logits)))
    
    # Step 1: Subtract max for numerical stability (doesn't change softmax)
    # find the max value in the logits_flat tensor
    max_logits = torch.max(logits_flat, dim=-1, keepdim=True)[0]  # (N, 1)
    # delete the max value
    logits_shifted = logits_flat - max_logits  # (N, vocab_size)

    # Step 2: Compute log(sum(exp(logits))) using the shifted logits
    # log(sum(exp(logits))) = max + log(sum(exp(logits - max)))
    # log(Σ exp(x_j)) = log(Σ exp(x_j - max) · exp(max))
    #            = log(exp(max) · Σ exp(x_j - max))
    #            = log(exp(max)) + log(Σ exp(x_j - max))
    #            = max + log(Σ exp(x_j - max))
    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.sum(torch.exp(logits_shifted), dim=-1))  # (N,) + (N,) -> (N,)

    # Step 3: Get the logits for the target classes
    # Use advanced indexing to gather the correct logits
    # 组合索引然后一次取出所有目标类别的logits
    target_logits = logits_flat[torch.arange(logits_flat.size(0), device=logits_flat.device), targets_flat]  # (N,)
    
    # Step 4: Compute cross-entropy: -target_logits + log_sum_exp

    loss_per_example = -target_logits + log_sum_exp  # (N,)

    # Step 5: Return average loss across all examples
    return loss_per_example.mean()

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

