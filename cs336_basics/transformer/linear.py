"""
Linear module implementation for CS336 Assignment 1.

This module implements a custom Linear layer that performs a linear transformation:
y = Wx

where W is a learnable weight matrix of shape (out_features, in_features).
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math


class Linear(nn.Module):
    """
    A linear transformation module that performs y = Wx.
    
    This is equivalent to PyTorch's nn.Linear but without bias terms,
    following modern LLM practices.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample  
        device: Device to store the parameters on
        dtype: Data type of the parameters
    """
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight matrix W with shape (out_features, in_features)
        # This follows row-major memory ordering convention
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal distribution
        # Following the assignment specifications:
        # N(μ=0, σ²=2/(in_features + out_features)) truncated at [-3σ, 3σ]
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        # Calculate standard deviation
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        
        # Initialize with truncated normal distribution
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation to input.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Use einsum for clarity and to handle arbitrary batch dimensions
        # x has shape (..., in_features), weight has shape (out_features, in_features)
        # We want output of shape (..., out_features)

        return torch.einsum('... i, o i->...o', x, self.weight)
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=False'
