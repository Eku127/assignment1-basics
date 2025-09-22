"""
Transformer Block Demo for CS336 Assignment 1.

This demo illustrates the mathematical principles and implementation of the
pre-norm Transformer block architecture.
"""

import torch
import torch.nn as nn
from cs336_basics.transformer import (
    TransformerBlock, RMSNorm, MultiHeadSelfAttention, SwiGLU
)


def demonstrate_transformer_block():
    """Demonstrate the Transformer block architecture and components."""
    print("=" * 60)
    print("CS336 Assignment 1 - Transformer Block Demo")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    d_ff = 16
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Feed-forward dimension (d_ff): {d_ff}")
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    print(f"Input tensor:\n{x}")
    
    print("\n" + "=" * 60)
    print("Pre-norm Transformer Block Architecture")
    print("=" * 60)
    
    print("\nPre-norm architecture follows this pattern:")
    print("1. First sublayer: y = x + MultiHeadSelfAttention(RMSNorm(x))")
    print("2. Second sublayer: z = y + FFN(RMSNorm(y))")
    print("\nThis provides a clean residual stream from input to output.")
    
    # Demonstrate individual components
    print("\n" + "-" * 40)
    print("Component 1: RMSNorm")
    print("-" * 40)
    
    norm = RMSNorm(d_model)
    x_normed = norm(x)
    print(f"After RMSNorm: shape = {x_normed.shape}")
    print(f"Original mean: {x.mean(dim=-1)}")
    print(f"Normalized RMS: {torch.sqrt(torch.mean(x_normed**2, dim=-1))}")
    
    print("\n" + "-" * 40)
    print("Component 2: Multi-Head Self-Attention")
    print("-" * 40)
    
    attention = MultiHeadSelfAttention(d_model, num_heads)
    attn_out = attention(x_normed)
    print(f"Attention output shape: {attn_out.shape}")
    print(f"Attention preserves sequence and model dimensions")
    
    print("\n" + "-" * 40)
    print("Component 3: SwiGLU Feed-Forward")
    print("-" * 40)
    
    ffn = SwiGLU(d_model, d_ff)
    ffn_out = ffn(x_normed)
    print(f"FFN output shape: {ffn_out.shape}")
    print(f"FFN processes each position independently")
    
    print("\n" + "=" * 60)
    print("Complete Transformer Block Forward Pass")
    print("=" * 60)
    
    # Note: This is framework code - user needs to implement the actual forward pass
    print("\nFramework created - you need to implement:")
    print("1. Initialize components in __init__:")
    print("   - self.norm1 = RMSNorm(d_model)")
    print("   - self.attention = MultiHeadSelfAttention(...)")
    print("   - self.norm2 = RMSNorm(d_model)")
    print("   - self.feed_forward = SwiGLU(...)")
    print("\n2. Implement forward pass:")
    print("   - Apply first sublayer with residual connection")
    print("   - Apply second sublayer with residual connection")
    
    print("\n" + "=" * 60)
    print("Key Implementation Notes")
    print("=" * 60)
    
    print("\n1. Residual Connections:")
    print("   - Essential for gradient flow in deep networks")
    print("   - Allow information to bypass transformations")
    print("   - Enable training of very deep models")
    
    print("\n2. Pre-norm vs Post-norm:")
    print("   - Pre-norm: Apply normalization before sublayer")
    print("   - Provides cleaner residual stream")
    print("   - Better training stability")
    
    print("\n3. RoPE Integration:")
    print("   - Applied within the attention mechanism")
    print("   - Provides relative positional information")
    print("   - No learnable parameters")
    
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    
    print("\n1. Implement the TransformerBlock.__init__ method")
    print("2. Implement the TransformerBlock.forward method")
    print("3. Test with: uv run pytest -k test_transformer_block")
    print("4. Stack multiple blocks to create a full Transformer LM")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demonstrate_transformer_block()
