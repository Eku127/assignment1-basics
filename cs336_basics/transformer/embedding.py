"""
Embedding module implementation for CS336 Assignment 1.

This module implements a custom Embedding layer that maps token IDs to dense vectors.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math


class Embedding(nn.Module):
    """
    A custom embedding layer that maps token IDs to dense vectors.
    
    This is equivalent to PyTorch's nn.Embedding but implemented from scratch.
    
    Args:
        num_embeddings: Size of the vocabulary (number of unique tokens)
        embedding_dim: Dimension of the embedding vectors (d_model)
        device: Device to store the parameters on
        dtype: Data type of the parameters
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embedding matrix with shape (num_embeddings, embedding_dim)
        # Following the assignment specifications:
        # N(μ=0, σ²=1) truncated at [-3, 3]
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal distribution
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        # Following assignment specifications: N(μ=0, σ²=1) truncated at [-3, 3]
        init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embedding vectors for the given token IDs.
        
        Args:
            token_ids: Input tensor of shape (..., ) containing token IDs
                      (integers from 0 to num_embeddings-1)
            
        Returns:
            Output tensor of shape (..., embedding_dim) containing embedding vectors
        """
        # Implement the embedding lookup
        # Hint: Use torch.index_select or advanced indexing
        # The input token_ids has shape (..., ) and we want output of shape (..., embedding_dim)
        # We need to select rows from self.weight based on token_ids
        
        # Method 1: Using torch.index_select
        # You need to flatten token_ids, select embeddings, then reshape back
        
        # Method 2: Using advanced indexing
        # You can directly use token_ids as indices: self.weight[token_ids]

        return self.weight[token_ids]
        
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}'
