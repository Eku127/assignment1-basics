"""
Attention mechanisms implementation for CS336 Assignment 1.

This module implements softmax and scaled dot-product attention as described in
"Attention Is All You Need" by Vaswani et al., 2017.
"""

import torch
import torch.nn as nn
import math


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax operation to the specified dimension of the input tensor.
    
    Softmax formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    
    For numerical stability, we subtract the maximum value in the target dimension
    from all elements before applying exp, since softmax is invariant to adding
    a constant to all inputs.
    
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension along which to apply softmax
        
    Returns:
        Output tensor of same shape as input, with softmax applied to specified dimension
    """
    # Step 1: Calculate maximum values for numerical stability
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    
    # Step 2: Subtract max values (softmax is invariant to adding constants)
    x_stable = x - max_vals
    
    # Step 3: Apply exponential
    exp_x = torch.exp(x_stable)
    
    # Step 4: Calculate sum along the specified dimension
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    
    # Step 5: Normalize by sum
    softmax_x = exp_x / sum_exp
    
    return softmax_x


def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Args:
        Q: Query tensor of shape (..., seq_len_q, d_k)
        K: Key tensor of shape (..., seq_len_k, d_k)  
        V: Value tensor of shape (..., seq_len_v, d_v)
        mask: Optional boolean mask of shape (seq_len_q, seq_len_k)
              True means "attend to", False means "mask out"
    
    Returns:
        Output tensor of shape (..., seq_len_q, d_v)
    """
    # TODO: Implement scaled dot-product attention
    # Step 1: Compute attention scores QK^T
    # Hint: Use torch.einsum for efficient matrix multiplication
    
    # Step 2: Scale by sqrt(d_k)
    # Hint: d_k is the last dimension of Q and K
    
    # Step 3: Apply mask if provided
    # Hint: Set masked positions to -inf before softmax
    
    # Step 4: Apply softmax
    # Hint: Use the softmax function you implemented above
    
    # Step 5: Apply attention weights to values
    # Hint: Use torch.einsum for efficient multiplication
    
    raise NotImplementedError("Please implement scaled dot-product attention")


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Create a causal mask for autoregressive language modeling.
    
    Causal mask prevents attention to future positions:
    - True: can attend (current and past positions)
    - False: cannot attend (future positions)
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        Boolean mask of shape (seq_len, seq_len)
    """
    # TODO: Implement causal mask creation
    # Hint: Create upper triangular matrix with False values
    # Hint: Lower triangular part should be True (can attend to past/current)
    # Hint: Diagonal should be True (can attend to current position)
    
    raise NotImplementedError("Please implement causal mask creation")


def create_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Create padding mask from attention mask.
    
    Args:
        attention_mask: Boolean tensor of shape (..., seq_len)
                       True means valid token, False means padding
    
    Returns:
        Boolean mask of shape (..., seq_len, seq_len)
        True means "attend to", False means "mask out"
    """
    # TODO: Implement padding mask creation
    # Hint: Expand attention_mask to create pairwise mask
    # Hint: Use broadcasting to create (..., seq_len, seq_len) shape
    
    raise NotImplementedError("Please implement padding mask creation")
