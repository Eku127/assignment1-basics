"""
Multi-Head Self-Attention implementation for CS336 Assignment 1.

This module implements causal multi-head self-attention as described in
"Attention Is All You Need" by Vaswani et al., 2017.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math
from .attention import scaled_dot_product_attention, create_causal_mask
from .rope import RotaryPositionalEmbedding
from .linear import Linear
from einops import rearrange


class MultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention module.
    
    Implements the multi-head self-attention mechanism with causal masking
    to prevent attention to future tokens. Supports both with and without RoPE.
    
    Args:
        d_model: Dimensionality of the input and output
        num_heads: Number of attention heads
        d_k: Key/Query dimension per head (default: d_model // num_heads)
        d_v: Value dimension per head (default: d_model // num_heads)
        use_rope: Whether to use Rotary Positional Embedding
        max_seq_len: Maximum sequence length for RoPE (if used)
        theta: RoPE theta parameter (if used)
        device: Device to store parameters on
        dtype: Data type of parameters
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.d_v = d_v if d_v is not None else d_model // num_heads
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device
            )

        # Validate dimensions
        if self.d_k * num_heads != d_model:
            raise ValueError(f"d_k * num_heads ({self.d_k * num_heads}) must equal d_model ({d_model})")
        if self.d_v * num_heads != d_model:
            raise ValueError(f"d_v * num_heads ({self.d_v * num_heads}) must equal d_model ({d_model})")
        
        # Initialize projection layers using Linear modules
        # Each projection: d_model -> d_model
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask to prevent attention to future tokens.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Boolean mask of shape (seq_len, seq_len)
            True means "can attend", False means "mask out"
        """
        # Implement causal mask creation
        # Hint: Use torch.triu or broadcasted comparison
        # Hint: Lower triangular part should be True (can attend to past/current)
        # Hint: Upper triangular part should be False (cannot attend to future)
        # Hint: Return boolean tensor of shape (seq_len, seq_len)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        return ~mask
        
    
    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project input to Query, Key, and Value matrices.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            
        Returns:
            Tuple of (Q, K, V) tensors of shape (..., seq_len, d_model)
        """
        # Use Linear layers for QKV projection
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        return Q, K, V
    
    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape tensor for multi-head attention.
        
        Args:
            x: Tensor of shape (..., seq_len, d_model)
            
        Returns:
            Tensor of shape (..., num_heads, seq_len, d_k) or (..., num_heads, seq_len, d_v)
        """
        # TODO: Implement reshaping for multi-head attention
        # Hint: Reshape to separate heads
        # Hint: Use view() to separate heads, then transpose to move head dimension
        # Hint: x_reshaped = x.view(..., seq_len, num_heads, d_k).transpose(-3, -2)
        # Hint: Result shape should be (..., num_heads, seq_len, d_k)

        # Achieved by view and transpose
        # # get seq_len
        # seq_len = x.shape[-2]
        # # split for heads
        # x_reshaped = x.view(..., seq_len, self.num_heads, self.d_k)
        # # swap the seq_len to the back
        # x_reshaped = x_reshaped.transpose(-3, -2)

        x_reshaped = rearrange(x, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', 
                              num_heads=self.num_heads, d_k=self.d_k)

        return x_reshaped
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine multi-head outputs back to single tensor.
        
        Args:
            x: Tensor of shape (..., num_heads, seq_len, d_v)
            
        Returns:
            Tensor of shape (..., seq_len, d_model)
        """
        # Implement head combination
        # Hint: Transpose heads back to last dimension
        # Hint: Use transpose(-3, -2) to move head dimension back
        # Hint: Use contiguous() to ensure memory layout is correct
        # Hint: Use view() to combine all heads into d_model dimension
        # Hint: Final shape should be (..., seq_len, d_model)

        x_combined = rearrange(x, '... num_heads seq_len d_v -> ... seq_len (num_heads d_v)', 
                              num_heads=self.num_heads, d_v=self.d_v)

        return x_combined
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional position indices for RoPE
            
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # Implement multi-head self-attention forward pass
        # Hint: Step 1: Project input to Q, K, V using _project_qkv()
        # x: (..., seq_len, d_model) --> Q, K, V: (..., seq_len, d_model)
        Q, K, V = self._project_qkv(x)

        # Hint: Step 2: Reshape for multi-head attention using _reshape_for_heads()
        # Q, K, V: (..., seq_len, d_model) --> Q, K, V: (..., num_heads, seq_len, d_k or d_v)
        Q = self._reshape_for_heads(Q)
        K = self._reshape_for_heads(K)
        V = self._reshape_for_heads(V)

        # Hint: Step 3: Apply RoPE if enabled (only to Q and K, not V)
        if self.use_rope and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Hint: Step 4: Create causal mask using _create_causal_mask()
        causal_mask = self._create_causal_mask(x.shape[-2], x.device)

        # Hint: Step 5: Apply scaled dot-product attention using scaled_dot_product_attention()

        # Q, K, V: (..., num_heads, seq_len, d_k or d_v) --> Att: (..., num_heads, seq_len, d_v)
        attention_output = scaled_dot_product_attention(Q, K, V, causal_mask)

        # Hint: Step 6: Combine heads using _combine_heads()
        # Att: (..., num_heads, seq_len, d_v) --> Att: (..., seq_len, d_model)
        attention_output = self._combine_heads(attention_output)

        # Step 7: Apply output projection using Linear layer
        output = self.o_proj(attention_output)

        return output
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return f'd_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k}, d_v={self.d_v}, use_rope={self.use_rope}'


class MultiHeadSelfAttentionWithRoPE(MultiHeadSelfAttention):
    """
    Multi-Head Self-Attention with RoPE support.
    
    This is a convenience class that automatically enables RoPE.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device=None,
        dtype=None
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        
        # RoPE module is already initialized in the parent class
        # No additional initialization needed
