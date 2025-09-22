#!/usr/bin/env python3
"""
Demo script to understand softmax implementation.
"""

import torch
import torch.nn.functional as F
import math


def demo_softmax_basic():
    """演示基本softmax操作"""
    print("=== 基本Softmax操作演示 ===\n")
    
    # 1. 基本softmax公式
    print("1. Softmax公式: softmax(x_i) = exp(x_i) / sum(exp(x_j))")
    print()
    
    # 2. 简单例子
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"输入: {x}")
    
    # 手动计算softmax
    exp_x = torch.exp(x)
    sum_exp = torch.sum(exp_x)
    softmax_manual = exp_x / sum_exp
    
    print(f"exp(x): {exp_x}")
    print(f"sum(exp(x)): {sum_exp}")
    print(f"手动softmax: {softmax_manual}")
    print(f"PyTorch softmax: {F.softmax(x, dim=0)}")
    print(f"是否相等: {torch.allclose(softmax_manual, F.softmax(x, dim=0))}")
    print()


def demo_softmax_numerical_stability():
    """演示数值稳定性问题"""
    print("=== 数值稳定性问题演示 ===\n")
    
    # 1. 大数值问题
    print("1. 大数值问题:")
    x_large = torch.tensor([100.0, 101.0, 102.0])
    print(f"输入: {x_large}")
    
    # 直接计算会溢出
    try:
        exp_large = torch.exp(x_large)
        print(f"exp(x): {exp_large}")
        print("结果: 数值溢出!")
    except:
        print("结果: 数值溢出!")
    
    # 2. 数值稳定性技巧
    print("\n2. 数值稳定性技巧:")
    print("技巧: 减去最大值，softmax不变")
    print("softmax(x_i) = softmax(x_i - max(x))")
    
    # 减去最大值
    x_stable = x_large - torch.max(x_large)
    print(f"x - max(x): {x_stable}")
    
    exp_stable = torch.exp(x_stable)
    sum_exp_stable = torch.sum(exp_stable)
    softmax_stable = exp_stable / sum_exp_stable
    
    print(f"exp(x - max): {exp_stable}")
    print(f"sum(exp(x - max)): {sum_exp_stable}")
    print(f"稳定softmax: {softmax_stable}")
    print(f"PyTorch softmax: {F.softmax(x_large, dim=0)}")
    print(f"是否相等: {torch.allclose(softmax_stable, F.softmax(x_large, dim=0))}")
    print()


def demo_softmax_dimensions():
    """演示不同维度的softmax"""
    print("=== 不同维度的Softmax演示 ===\n")
    
    # 2D张量
    x_2d = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    
    print("2D张量:")
    print(f"输入形状: {x_2d.shape}")
    print(f"输入:\n{x_2d}")
    print()
    
    # 沿最后一个维度softmax
    print("沿最后一个维度 (dim=-1):")
    softmax_last = F.softmax(x_2d, dim=-1)
    print(f"结果:\n{softmax_last}")
    print(f"每行和: {torch.sum(softmax_last, dim=-1)}")
    print()
    
    # 沿第一个维度softmax
    print("沿第一个维度 (dim=0):")
    softmax_first = F.softmax(x_2d, dim=0)
    print(f"结果:\n{softmax_first}")
    print(f"每列和: {torch.sum(softmax_first, dim=0)}")
    print()


def demo_softmax_implementation_hints():
    """提供softmax实现提示"""
    print("=== Softmax实现提示 ===\n")
    
    print("实现步骤:")
    print("1. 计算最大值:")
    print("   max_vals = torch.max(x, dim=dim, keepdim=True)[0]")
    print()
    print("2. 减去最大值 (数值稳定性):")
    print("   x_stable = x - max_vals")
    print()
    print("3. 计算指数:")
    print("   exp_x = torch.exp(x_stable)")
    print()
    print("4. 计算和:")
    print("   sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)")
    print()
    print("5. 归一化:")
    print("   softmax_x = exp_x / sum_exp")
    print()
    
    print("边界情况处理:")
    print("- 如果所有值都是-inf，应该返回均匀分布")
    print("- 使用keepdim=True保持维度形状")
    print("- 确保输出形状与输入相同")
    print()


def demo_softmax_vs_other_normalizations():
    """对比softmax与其他归一化方法"""
    print("=== Softmax vs 其他归一化方法 ===\n")
    
    x = torch.tensor([1.0, 2.0, 3.0])
    
    print(f"输入: {x}")
    print()
    
    # Softmax
    softmax_x = F.softmax(x, dim=0)
    print(f"Softmax: {softmax_x}")
    print(f"和: {torch.sum(softmax_x)}")
    print()
    
    # L1归一化
    l1_norm = x / torch.sum(torch.abs(x))
    print(f"L1归一化: {l1_norm}")
    print(f"和: {torch.sum(l1_norm)}")
    print()
    
    # L2归一化
    l2_norm = x / torch.norm(x)
    print(f"L2归一化: {l2_norm}")
    print(f"和: {torch.sum(l2_norm)}")
    print()
    
    print("Softmax特点:")
    print("- 输出在[0,1]范围内")
    print("- 所有元素和为1")
    print("- 保持相对大小关系")
    print("- 指数函数放大差异")


def demo_attention_scores():
    """演示注意力分数计算"""
    print("=== 注意力分数计算演示 ===\n")
    
    # 模拟Q, K, V
    seq_len, d_k = 3, 4
    Q = torch.randn(seq_len, d_k)
    K = torch.randn(seq_len, d_k)
    V = torch.randn(seq_len, d_k)
    
    print(f"Q形状: {Q.shape}")
    print(f"K形状: {K.shape}")
    print(f"V形状: {V.shape}")
    print()
    
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1))
    print(f"注意力分数 QK^T:\n{scores}")
    print()
    
    # 缩放
    scaled_scores = scores / math.sqrt(d_k)
    print(f"缩放后分数 QK^T/√d_k:\n{scaled_scores}")
    print()
    
    # 应用softmax
    attention_weights = F.softmax(scaled_scores, dim=-1)
    print(f"注意力权重 (softmax):\n{attention_weights}")
    print(f"每行和: {torch.sum(attention_weights, dim=-1)}")
    print()
    
    # 应用权重到值
    output = torch.matmul(attention_weights, V)
    print(f"最终输出:\n{output}")


if __name__ == "__main__":
    demo_softmax_basic()
    demo_softmax_numerical_stability()
    demo_softmax_dimensions()
    demo_softmax_implementation_hints()
    demo_softmax_vs_other_normalizations()
    demo_attention_scores()
