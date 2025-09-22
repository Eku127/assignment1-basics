"""
SwiGLU Feed-Forward Network implementation for CS336 Assignment 1.

This module implements the SwiGLU activation function which combines:
- SiLU (Swish) activation: SiLU(x) = x * sigmoid(x)
- Gated Linear Unit (GLU): GLU(x, W1, W2) = sigmoid(W1 x) ⊙ W2 x
- SwiGLU: FFN(x) = W2 (SiLU(W1 x) ⊙ W3 x)
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network module.
    
    This implements the SwiGLU activation function used in modern LLMs like LLaMA 3.
    It combines SiLU activation with Gated Linear Units for better performance.
    
    Args:
        d_model: Input dimension of the model
        d_ff: Hidden dimension of the feed-forward network
        device: Device to store the parameters on
        dtype: Data type of the parameters
    """
    
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # TODO: Initialize the three weight matrices
        # W1: (d_ff, d_model) - first linear transformation
        # W2: (d_model, d_ff) - output projection
        # W3: (d_ff, d_model) - gating projection
        
        # Hint: Use the same initialization as Linear module
        # std = sqrt(2.0 / (in_features + out_features))

        # Ensure dtype is floating point for gradient computation
        if dtype is not None and not torch.is_floating_point(torch.tensor(0, dtype=dtype)):
            dtype = torch.float32
        
        # First linear within the gat
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype or torch.float32))
        # The output weight
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype or torch.float32))
        # The content way weight
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype or torch.float32))

        # weight initialization
        self._init_weights(self.W1, d_model, d_ff)
        self._init_weights(self.W2, d_ff, d_model)
        self._init_weights(self.W3, d_model, d_ff)
        
    
    def _init_weights(self, weight_tensor, in_features, out_features):
        """Initialize weights using truncated normal distribution."""
        std = math.sqrt(2.0 / (in_features + out_features))
        init.trunc_normal_(weight_tensor, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU (Swish) activation function.
        
        SiLU(x) = x * sigmoid(x)
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with SiLU activation applied
        """
        # TODO: Implement SiLU activation
        # Hint: Use torch.sigmoid for numerical stability
        # return x * torch.sigmoid(x)
        
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward network.
        
        FFN(x) = W2 (SiLU(W1 x) ⊙ W3 x)
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
            
        Implementation steps:
        1. Apply W1 transformation: h1 = W1 x
        2. Apply W3 transformation: h3 = W3 x  
        3. Apply SiLU to h1: silu_h1 = SiLU(h1)
        4. Element-wise multiply: gated = silu_h1 ⊙ h3
        5. Apply W2 transformation: output = W2 gated
        """
        # Implement SwiGLU forward pass
        # Step 1: Apply W1 transformation
        h1 = torch.einsum('...d,fd->...f', x, self.W1)        
        
        # Step 2: Apply W3 transformation  
        h3 = torch.einsum('...d,fd->...f', x, self.W3)

        # Step 3: Apply SiLU to h1
        silu_h1 = self.silu(h1)

        # Step 4: Element-wise multiply (gating)
        gated = silu_h1 * h3

        # Step 5: Apply W2 transformation
        output = torch.einsum('...f,df->...d', gated, self.W2)

        return output
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return f'd_model={self.d_model}, d_ff={self.d_ff}'
