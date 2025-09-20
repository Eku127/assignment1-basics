#!/usr/bin/env python3
"""
Demo script to understand RoPE (Rotary Positional Embedding) implementation.
"""

import torch
import torch.nn as nn
import math


def demo_rope_math():
    """演示RoPE的数学原理"""
    print("=== RoPE数学原理演示 ===\n")
    
    # 1. RoPE公式
    print("1. RoPE公式:")
    print("   对于位置i和维度对(2k-1, 2k):")
    print("   θ_i,k = Θ^(2k-1)/d_k")
    print("   旋转矩阵R_k^i = [[cos(θ_i,k), -sin(θ_i,k)],")
    print("                     [sin(θ_i,k),  cos(θ_i,k)]]")
    print()
    
    # 2. 简单例子
    d_k = 4  # 必须是偶数
    theta = 10000.0
    max_seq_len = 10
    
    print(f"参数设置:")
    print(f"  d_k = {d_k}")
    print(f"  theta = {theta}")
    print(f"  max_seq_len = {max_seq_len}")
    print()
    
    # 3. 计算旋转角度
    print("2. 计算旋转角度:")
    for i in range(3):  # 前3个位置
        print(f"   位置 {i}:")
        for k in range(1, d_k // 2 + 1):  # 维度对
            angle = theta ** ((2 * k - 1) / d_k)
            cos_val = math.cos(angle)
            sin_val = math.sin(angle)
            print(f"     维度对 {2*k-1},{2*k}: θ = {angle:.4f}, cos = {cos_val:.4f}, sin = {sin_val:.4f}")
        print()


def demo_rotation_matrix():
    """演示旋转矩阵的应用"""
    print("=== 旋转矩阵应用演示 ===\n")
    
    # 1. 2D旋转示例
    print("1. 2D旋转示例:")
    angle = math.pi / 4  # 45度
    cos_val = math.cos(angle)
    sin_val = math.sin(angle)
    
    # 旋转矩阵
    R = torch.tensor([[cos_val, -sin_val],
                      [sin_val, cos_val]])
    
    # 输入向量
    x = torch.tensor([1.0, 0.0])  # 指向x轴正方向的向量
    
    print(f"旋转角度: {angle:.4f} 弧度 ({math.degrees(angle):.1f}度)")
    print(f"旋转矩阵 R:\n{R}")
    print(f"输入向量 x: {x}")
    
    # 应用旋转
    x_rotated = R @ x
    print(f"旋转后向量: {x_rotated}")
    print()
    
    # 2. 多个向量的旋转
    print("2. 多个向量的旋转:")
    positions = torch.tensor([0, 1, 2])
    x_batch = torch.tensor([[1.0, 0.0],
                           [0.0, 1.0], 
                           [1.0, 1.0]])
    
    print(f"位置: {positions}")
    print(f"输入向量:\n{x_batch}")
    
    # 为每个位置计算不同的旋转角度
    angles = positions.float() * math.pi / 4
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    
    print(f"各位置角度: {angles}")
    print(f"cos值: {cos_vals}")
    print(f"sin值: {sin_vals}")
    
    # 应用旋转
    x_rotated_batch = torch.zeros_like(x_batch)
    for i in range(len(positions)):
        R_i = torch.tensor([[cos_vals[i], -sin_vals[i]],
                           [sin_vals[i], cos_vals[i]]])
        x_rotated_batch[i] = R_i @ x_batch[i]
    
    print(f"旋转后向量:\n{x_rotated_batch}")
    print()


def demo_rope_implementation_hints():
    """提供RoPE实现提示"""
    print("=== RoPE实现提示 ===\n")
    
    print("__init__方法需要实现:")
    print("1. 预计算cos和sin值:")
    print("   - 创建形状为(max_seq_len, d_k//2)的缓冲区")
    print("   - 对每个位置i和维度对k计算θ_i,k = theta^(2k-1)/d_k")
    print("   - 使用self.register_buffer(persistent=False)创建缓冲区")
    print()
    
    print("2. 缓冲区创建示例:")
    print("   cos_buffer = torch.zeros(max_seq_len, d_k // 2)")
    print("   sin_buffer = torch.zeros(max_seq_len, d_k // 2)")
    print("   for i in range(max_seq_len):")
    print("       for k in range(d_k // 2):")
    print("           angle = theta ** ((2 * k - 1) / d_k)")
    print("           cos_buffer[i, k] = math.cos(angle)")
    print("           sin_buffer[i, k] = math.sin(angle)")
    print()
    
    print("_compute_rotation_angles方法需要实现:")
    print("1. 使用位置索引从预计算缓冲区中选择值:")
    print("   - 使用高级索引: cos_vals = self.cos_buffer[positions]")
    print("   - 处理任意批次维度")
    print()
    
    print("_apply_rotary_embedding方法需要实现:")
    print("1. 重塑输入以分离偶数和奇数维度:")
    print("   - x_even = x[..., ::2]  # 偶数维度")
    print("   - x_odd = x[..., 1::2]  # 奇数维度")
    print()
    print("2. 应用旋转公式:")
    print("   - x'_even = x_even * cos - x_odd * sin")
    print("   - x'_odd = x_even * sin + x_odd * cos")
    print()
    print("3. 重塑回原始形状:")
    print("   - 交错排列偶数和奇数维度")
    print()


def demo_rope_vs_absolute_positional():
    """对比RoPE和绝对位置编码"""
    print("=== RoPE vs 绝对位置编码对比 ===\n")
    
    print("绝对位置编码:")
    print("- 为每个位置学习固定的嵌入向量")
    print("- 位置信息是绝对的，不依赖于相对关系")
    print("- 需要学习参数")
    print()
    
    print("RoPE优势:")
    print("- 编码相对位置信息，更符合注意力机制")
    print("- 无需学习参数，基于数学公式")
    print("- 可以处理任意长度的序列（理论上）")
    print("- 在长序列上表现更好")
    print()


def demo_rope_attention_interaction():
    """演示RoPE与注意力的交互"""
    print("=== RoPE与注意力的交互 ===\n")
    
    print("在注意力机制中:")
    print("1. 对Query和Key都应用RoPE")
    print("2. 计算注意力分数时，位置信息被编码在向量中")
    print("3. 相对位置关系通过旋转角度体现")
    print()
    
    print("数学表示:")
    print("Q' = RoPE(Q, pos_q)  # Query旋转")
    print("K' = RoPE(K, pos_k)  # Key旋转")
    print("Attention(Q', K') = softmax(Q'K'^T/√d_k)")
    print()
    
    print("相对位置编码:")
    print("- 两个位置i和j的相对位置差为|i-j|")
    print("- 通过旋转角度的差异体现相对位置关系")
    print()


if __name__ == "__main__":
    demo_rope_math()
    demo_rotation_matrix()
    demo_rope_implementation_hints()
    demo_rope_vs_absolute_positional()
    demo_rope_attention_interaction()
