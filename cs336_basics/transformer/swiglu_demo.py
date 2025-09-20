#!/usr/bin/env python3
"""
Demo script to understand SwiGLU implementation.
"""

import torch
import torch.nn as nn
import math


def demo_silu_activation():
    """演示SiLU激活函数"""
    print("=== SiLU激活函数演示 ===\n")
    
    # 1. SiLU公式
    print("1. SiLU公式: SiLU(x) = x * sigmoid(x)")
    print()
    
    # 2. 手动实现SiLU
    def manual_silu(x):
        return x * torch.sigmoid(x)
    
    # 3. 测试SiLU
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    silu_result = manual_silu(x)
    
    print(f"输入: {x}")
    print(f"SiLU输出: {silu_result}")
    print()
    
    # 4. 与ReLU对比
    relu_result = torch.relu(x)
    print(f"ReLU输出: {relu_result}")
    print(f"SiLU vs ReLU: {torch.allclose(silu_result, relu_result)}")
    print()


def demo_glu_concept():
    """演示GLU概念"""
    print("=== GLU (Gated Linear Unit) 概念演示 ===\n")
    
    # 1. GLU公式
    print("1. GLU公式: GLU(x, W1, W2) = sigmoid(W1 x) ⊙ W2 x")
    print("   其中 ⊙ 表示逐元素相乘")
    print()
    
    # 2. 简单例子
    batch_size, d_model = 2, 4
    d_ff = 6
    
    x = torch.randn(batch_size, d_model)
    W1 = torch.randn(d_ff, d_model)
    W2 = torch.randn(d_ff, d_model)
    
    print(f"输入 x 形状: {x.shape}")
    print(f"权重 W1 形状: {W1.shape}")
    print(f"权重 W2 形状: {W2.shape}")
    print()
    
    # 3. 计算GLU
    h1 = torch.einsum('bd,fd->bf', x, W1)  # W1 x
    h2 = torch.einsum('bd,fd->bf', x, W2)  # W2 x
    gated = torch.sigmoid(h1) * h2  # sigmoid(W1 x) ⊙ W2 x
    
    print(f"h1 = W1 x 形状: {h1.shape}")
    print(f"h2 = W2 x 形状: {h2.shape}")
    print(f"gated 形状: {gated.shape}")
    print()


def demo_swiglu_math():
    """演示SwiGLU数学原理"""
    print("=== SwiGLU数学原理演示 ===\n")
    
    # 1. SwiGLU公式
    print("1. SwiGLU公式: FFN(x) = W2 (SiLU(W1 x) ⊙ W3 x)")
    print("   其中:")
    print("   - W1: (d_ff, d_model) - 第一个线性变换")
    print("   - W2: (d_model, d_ff) - 输出投影")
    print("   - W3: (d_ff, d_model) - 门控投影")
    print()
    
    # 2. 手动实现SwiGLU
    def manual_swiglu(x, W1, W2, W3):
        # Step 1: W1 x
        h1 = torch.einsum('...d,fd->...f', x, W1)
        
        # Step 2: W3 x
        h3 = torch.einsum('...d,fd->...f', x, W3)
        
        # Step 3: SiLU(W1 x)
        silu_h1 = h1 * torch.sigmoid(h1)
        
        # Step 4: SiLU(W1 x) ⊙ W3 x
        gated = silu_h1 * h3
        
        # Step 5: W2 (gated)
        output = torch.einsum('...f,df->...d', gated, W2)
        
        return output
    
    # 3. 测试SwiGLU
    batch_size, seq_len, d_model = 2, 3, 4
    d_ff = 6
    
    x = torch.randn(batch_size, seq_len, d_model)
    W1 = torch.randn(d_ff, d_model)
    W2 = torch.randn(d_model, d_ff)
    W3 = torch.randn(d_ff, d_model)
    
    print(f"输入 x 形状: {x.shape}")
    print(f"权重 W1 形状: {W1.shape}")
    print(f"权重 W2 形状: {W2.shape}")
    print(f"权重 W3 形状: {W3.shape}")
    print()
    
    # 4. 计算SwiGLU
    output = manual_swiglu(x, W1, W2, W3)
    print(f"SwiGLU输出形状: {output.shape}")
    print(f"SwiGLU输出:\n{output}")
    print()


def demo_swiglu_implementation_hints():
    """提供实现提示"""
    print("=== SwiGLU实现提示 ===\n")
    
    print("__init__方法需要实现:")
    print("1. 初始化三个权重矩阵:")
    print("   self.W1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))")
    print("   self.W2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))")
    print("   self.W3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))")
    print()
    print("2. 初始化权重:")
    print("   self._init_weights(self.W1, d_model, d_ff)")
    print("   self._init_weights(self.W2, d_ff, d_model)")
    print("   self._init_weights(self.W3, d_model, d_ff)")
    print()
    
    print("silu方法需要实现:")
    print("   return x * torch.sigmoid(x)")
    print()
    
    print("forward方法需要实现:")
    print("1. 应用W1变换:")
    print("   h1 = torch.einsum('...d,fd->...f', x, self.W1)")
    print()
    print("2. 应用W3变换:")
    print("   h3 = torch.einsum('...d,fd->...f', x, self.W3)")
    print()
    print("3. 应用SiLU:")
    print("   silu_h1 = self.silu(h1)")
    print()
    print("4. 逐元素相乘:")
    print("   gated = silu_h1 * h3")
    print()
    print("5. 应用W2变换:")
    print("   output = torch.einsum('...f,df->...d', gated, self.W2)")
    print()


def demo_swiglu_vs_ffn():
    """对比SwiGLU和传统FFN"""
    print("=== SwiGLU vs 传统FFN对比 ===\n")
    
    # 传统FFN
    def traditional_ffn(x, W1, W2):
        h = torch.einsum('...d,fd->...f', x, W1)  # W1 x
        h = torch.relu(h)  # ReLU activation
        output = torch.einsum('...f,df->...d', h, W2)  # W2 h
        return output
    
    # SwiGLU
    def swiglu_ffn(x, W1, W2, W3):
        h1 = torch.einsum('...d,fd->...f', x, W1)  # W1 x
        h3 = torch.einsum('...d,fd->...f', x, W3)  # W3 x
        silu_h1 = h1 * torch.sigmoid(h1)  # SiLU(W1 x)
        gated = silu_h1 * h3  # SiLU(W1 x) ⊙ W3 x
        output = torch.einsum('...f,df->...d', gated, W2)  # W2 gated
        return output
    
    # 测试
    batch_size, d_model = 2, 4
    d_ff = 6
    
    x = torch.randn(batch_size, d_model)
    W1 = torch.randn(d_ff, d_model)
    W2 = torch.randn(d_model, d_ff)
    W3 = torch.randn(d_ff, d_model)
    
    # 传统FFN
    traditional_output = traditional_ffn(x, W1, W2)
    
    # SwiGLU
    swiglu_output = swiglu_ffn(x, W1, W2, W3)
    
    print(f"输入形状: {x.shape}")
    print(f"传统FFN输出形状: {traditional_output.shape}")
    print(f"SwiGLU输出形状: {swiglu_output.shape}")
    print(f"两种方法输出不同: {not torch.allclose(traditional_output, swiglu_output)}")
    print()


if __name__ == "__main__":
    demo_silu_activation()
    demo_glu_concept()
    demo_swiglu_math()
    demo_swiglu_implementation_hints()
    demo_swiglu_vs_ffn()
