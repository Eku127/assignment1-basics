"""
RMSNorm (Root Mean Square Layer Normalization) implementation for CS336 Assignment 1.

This module implements RMSNorm as described in the assignment:
RMSNorm(ai) = (ai / RMS(a)) * gi

where RMS(a) = sqrt(1/d_model * sum(ai^2) + eps)
"""

import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization module.
    
    This implements the RMSNorm normalization as used in modern LLMs like LLaMA.
    It normalizes activations by their root mean square instead of mean and variance.
    
    Args:
        d_model: Hidden dimension of the model
        eps: Epsilon value for numerical stability (default: 1e-5)
        device: Device to store the parameters on
        dtype: Data type of the parameters
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        # Initialize the gain parameter
        # The gain parameter 'g' has shape (d_model,) and should be initialized to 1
        # Use nn.Parameter to make it learnable
        # Hint: torch.ones(d_model, device=device, dtype=dtype)
        # Ensure dtype is floating point for gradient computation
        if dtype is not None and not torch.is_floating_point(torch.tensor(0, dtype=dtype)):
            dtype = torch.float32

        self.gain = nn.Parameter(
            torch.ones(self.d_model, device=device, dtype=dtype or torch.float32)
        )

        # for gain we do not need to init the weights
        # self._init_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            Output tensor of the same shape as input
            
        Implementation steps:
        1. Upcast input to torch.float32 for numerical stability
        2. Calculate RMS(a) = sqrt(1/d_model * sum(ai^2) + eps)
        3. Normalize: ai / RMS(a)
        4. Apply gain: (ai / RMS(a)) * gi
        5. Downcast back to original dtype
        """
        # Implement RMSNorm forward pass
        # Step 1: Upcast to float32 for numerical stability
        # Step 2: Calculate RMS(a) = sqrt(1/d_model * sum(ai^2) + eps)
        # Hint: Use torch.mean(x**2, dim=-1, keepdim=True) to get mean of squared values
        # Then take sqrt and add eps
        # Step 3: Normalize by RMS
        # Step 4: Apply gain parameter
        # Step 5: Downcast to original dtype


        # get and save the original input type
        in_dtype = x.dtype

        # get RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        # get norm 
        normalized = x / rms

        # apply gain
        rmsnorm = normalized * self.gain

        rmsnorm = torch.einsum('... d, d-> ... d', normalized, self.gain)

        return rmsnorm.to(in_dtype)

        
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return f'd_model={self.d_model}, eps={self.eps}'