#!/usr/bin/env python3
"""
Demo script to test the Linear module implementation.
"""

import torch
from linear import Linear


def test_linear_demo():
    """Test the Linear module with a simple example."""
    print("Testing Linear module...")
    
    # Create a simple linear layer
    d_in = 4
    d_out = 3
    linear = Linear(d_in, d_out)
    
    print(f"Created Linear layer: {d_in} -> {d_out}")
    print(f"Weight shape: {linear.weight.shape}")
    print(f"Weight:\n{linear.weight}")
    
    # Create test input
    batch_size = 2
    seq_len = 5
    x = torch.randn(batch_size, seq_len, d_in)
    print(f"\nInput shape: {x.shape}")
    print(f"Input:\n{x}")
    
    # Forward pass
    output = linear(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output:\n{output}")
    
    # Verify the computation manually
    print("\nVerifying computation...")
    expected = torch.einsum('bsi,oi->bso', x, linear.weight)
    print(f"Expected output:\n{expected}")
    print(f"Output matches expected: {torch.allclose(output, expected)}")
    
    # Test with different batch dimensions
    print("\nTesting with different batch dimensions...")
    x_4d = torch.randn(2, 3, 4, d_in)  # 4D input
    output_4d = linear(x_4d)
    print(f"4D input shape: {x_4d.shape}")
    print(f"4D output shape: {output_4d.shape}")
    
    print("\nLinear module test completed successfully!")


if __name__ == "__main__":
    test_linear_demo()
