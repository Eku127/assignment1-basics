"""
Transformer Block implementation for CS336 Assignment 1.

This module implements the pre-norm Transformer block as described in the assignment.
Each block contains a multi-head self-attention sublayer and a position-wise feed-forward
sublayer, with RMSNorm applied before each sublayer and residual connections.
"""

import torch
import torch.nn as nn
from .multihead_attention import MultiHeadSelfAttention
from .positionwise_feedforward import SwiGLU, FFN_SiLU
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer Block.
    
    Implements a single Transformer block with the pre-norm architecture:
    1. RMSNorm -> Multi-Head Self-Attention -> Residual connection
    2. RMSNorm -> Position-wise Feed-Forward -> Residual connection
    
    Args:
        d_model: Dimensionality of the model (input/output dimension)
        num_heads: Number of attention heads
        d_ff: Dimensionality of the feed-forward inner layer
        use_rope: Whether to use Rotary Positional Embedding
        max_seq_len: Maximum sequence length for RoPE (if used)
        theta: RoPE theta parameter (if used)
        eps: Epsilon for RMSNorm numerical stability
        use_swiglu: Whether to use SwiGLU (True) or FFN_SiLU (False) for feed-forward
        device: Device to store parameters on
        dtype: Data type of parameters
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = True,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        eps: float = 1e-5,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rope = use_rope
        
        # Initialize the components of the Transformer block
        # Hint: You need 4 components:
        # 1. RMSNorm for attention sublayer
        # 2. Multi-Head Self-Attention layer
        # 3. RMSNorm for feed-forward sublayer  
        # 4. Position-wise feed-forward layer (SwiGLU)
        
        # Initialize RMSNorm layers
        # Hint: Use RMSNorm with d_model dimension and eps
        self.norm1 = RMSNorm(d_model=d_model, eps=eps)
        self.norm2 = RMSNorm(d_model=d_model, eps=eps)
        
        # Initialize Multi-Head Self-Attention
        # Hint: Use MultiHeadSelfAttention with appropriate parameters
        # Hint: Pass use_rope, max_seq_len, theta if using RoPE
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
            )
        
        # Initialize Position-wise Feed-Forward Network
        # Use SwiGLU or FFN_SiLU based on use_swiglu flag
        self.feed_forward = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        # self.feed_forward = FFN_SiLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = True
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Apply Transformer block forward pass with optional KV cache support.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional position indices for RoPE
            past_key_values: Optional tuple of (past_K, past_V) cache tensors for this layer
            use_cache: Whether to return updated cache (default: True)
            
        Returns:
            Tuple of:
            - output: Output tensor of shape (..., seq_len, d_model)
            - new_past_key_values: Updated cache tuple (K, V) or None
        """
        # Implement the pre-norm Transformer block forward pass
        # 
        # Pre-norm architecture:
        # 1. First sublayer: y = x + MultiHeadSelfAttention(RMSNorm(x))
        # 2. Second sublayer: z = y + FFN(RMSNorm(y))
        
        # Step 1 - Attention sublayer
        # Apply norm1, then attention with cache, then add residual connection
        residual1 = x
        normed1 = self.norm1(x)
        attention_out, new_past_key_values = self.attention(
            normed1, 
            token_positions=token_positions,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        y = residual1 + attention_out
        
        # Step 2 - Feed-forward sublayer  
        # Apply norm2, then feed_forward, then add residual connection
        # Note: FFN does not use cache, so no cache handling needed here
        normed2 = self.norm2(y)
        ff_out = self.feed_forward(normed2)
        z = y + ff_out

        return z, new_past_key_values
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return f'd_model={self.d_model}, num_heads={self.num_heads}, d_ff={self.d_ff}, use_rope={self.use_rope}'
