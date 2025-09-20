"""
Rotary Positional Embedding (RoPE) implementation for CS336 Assignment 1.

This module implements RoPE as described in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
by Su et al., 2021. RoPE applies rotary transformations to query and key vectors to encode positional information.
"""

import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) module.
    
    RoPE applies rotary transformations to pairs of embedding elements to encode
    positional information. For a given position i and dimension pair (2k-1, 2k),
    the rotation angle is θ_i,k = Θ^(2k-1)/d where Θ is a constant.
    
    Args:
        theta: The base frequency parameter Θ for RoPE
        d_k: Dimension of query and key vectors (must be even)
        max_seq_len: Maximum sequence length to pre-compute rotations for
        device: Device to store the buffers on
    """
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Validate that d_k is even (needed for pairing dimensions)
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")
        
        # TODO: Pre-compute cos and sin values for all positions and dimension pairs
        # Hint: Create buffers using self.register_buffer(persistent=False)
        # Hint: You'll need cos and sin tensors of shape (max_seq_len, d_k // 2)
        # Hint: For each position i and dimension pair k, compute:
        #       cos_vals[i, k] = cos(θ_i,k)
        #       sin_vals[i, k] = sin(θ_i,k)
        

        # construct sin and cos buffer
        # cos_buffer = torch.zeros(max_seq_len, d_k // 2, device=device)
        # sin_buffer = torch.zeros(max_seq_len, d_k // 2, device=device)

        # for i in range(max_seq_len):
        #     for k in range(d_k // 2):
        #         angle = i * (theta ** (-2 * k / d_k))  # Formula: i * theta^(-2k/d)
        #         cos_buffer[i, k] = math.cos(angle)
        #         sin_buffer[i, k] = math.sin(angle)

        # Create position indices (0 to max_seq_len-1)
        positions = torch.arange(max_seq_len, device=device).float()
        # Create dimension indices (0 to d_k//2-1) 
        dims = torch.arange(d_k // 2, device=device).float()
        
        # Compute angles using broadcasting: (max_seq_len, d_k//2)
        # Formula: i * theta^(-2k/d) for position i and dimension k
        angles = positions.unsqueeze(1) * (theta ** (-2 * dims / d_k))
        
        # Compute cos and sin values
        cos_buffer = torch.cos(angles)
        sin_buffer = torch.sin(angles)

        # 注册缓冲区
        self.register_buffer('cos_buffer', cos_buffer, persistent=False)
        self.register_buffer('sin_buffer', sin_buffer, persistent=False)

    
    def _compute_rotation_angles(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotation angles for given positions.
        
        Args:
            positions: Tensor of shape (..., seq_len) containing position indices
            
        Returns:
            Tuple of (cos_vals, sin_vals) tensors of shape (..., seq_len, d_k//2)
        """
        # 确保positions在正确的设备上
        positions = positions.to(self.cos_buffer.device)
        
        # 限制位置范围在有效区间内
        positions = torch.clamp(positions, 0, self.max_seq_len - 1)
        
        # 使用高级索引从缓冲区选择cos和sin值
        cos_vals = self.cos_buffer[positions]
        sin_vals = self.sin_buffer[positions]
        
        return cos_vals, sin_vals

        
    
    def _apply_rotary_embedding(self, x: torch.Tensor, cos_vals: torch.Tensor, sin_vals: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            cos_vals: Cosine values of shape (..., seq_len, d_k//2)
            sin_vals: Sine values of shape (..., seq_len, d_k//2)
            
        Returns:
            Output tensor of same shape as input
        """
        # TODO: Implement rotary embedding application
        # Hint: Reshape x to separate even and odd dimensions
        # Hint: Apply rotation formula for each pair (x_2k-1, x_2k):
        #       x'_2k-1 = x_2k-1 * cos(θ) - x_2k * sin(θ)
        #       x'_2k = x_2k-1 * sin(θ) + x_2k * cos(θ)
        # Hint: Reshape back to original shape
        # 分离偶数和奇数维度
        # (..., seq_len, d_k) --> (..., seq_len, d_k//2)
        x_even = x[..., ::2]    # 偶数索引: 0, 2, 4, ...
        x_odd = x[..., 1::2]    # 奇数索引: 1, 3, 5, ...
        
        # 应用旋转公式
        # cos_vals and sin_vals are (..., seq_len, d_k//2)
        x_even_rotated = x_even * cos_vals - x_odd * sin_vals
        x_odd_rotated = x_even * sin_vals + x_odd * cos_vals
        
        # 重新组合 - 交错排列恢复原始顺序
        x_rotated = torch.zeros_like(x)
        x_rotated[..., ::2] = x_even_rotated   # 偶数位置放偶数维度
        x_rotated[..., 1::2] = x_odd_rotated   # 奇数位置放奇数维度
                
        return x_rotated
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices of shape (..., seq_len)
            
        Returns:
            Output tensor of same shape as input
        """
        # TODO: Implement RoPE forward pass
        # Step 1: Compute rotation angles for given positions
        cos_vals, sin_vals = self._compute_rotation_angles(token_positions)
        
        # Step 2: Apply rotary embedding
        output = self._apply_rotary_embedding(x, cos_vals, sin_vals)

        return output        
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return f'theta={self.theta}, d_k={self.d_k}, max_seq_len={self.max_seq_len}'
