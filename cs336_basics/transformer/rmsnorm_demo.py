#!/usr/bin/env python3
"""
Demo script to understand RMSNorm implementation.
"""

import torch
import torch.nn as nn
import math


def demo_rmsnorm_math():
    """演示RMSNorm的数学原理"""
    print("=== RMSNorm数学原理演示 ===\n")
    
    # 1. 基本概念
    print("1. RMSNorm公式:")
    print("   RMSNorm(ai) = (ai / RMS(a)) * gi")
    print("   其中 RMS(a) = sqrt(1/d_model * sum(ai^2) + eps)")
    print()
    
    # 2. 手动计算示例
    print("2. 手动计算示例:")
    batch_size, seq_len, d_model = 2, 3, 4
    eps = 1e-5
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    print(f"输入数据:\n{x}")
    print()
    
    # 计算RMS
    squared = x ** 2  # 每个元素的平方
    mean_squared = torch.mean(squared, dim=-1, keepdim=True)  # 在最后一个维度上求平均
    rms = torch.sqrt(mean_squared + eps)  # 计算RMS
    print(f"平方后:\n{squared}")
    print(f"平均平方:\n{mean_squared}")
    print(f"RMS:\n{rms}")
    print()
    
    # 归一化
    normalized = x / rms
    print(f"归一化后:\n{normalized}")
    print()
    
    # 应用gain参数
    gain = torch.ones(d_model)  # gain参数初始化为1
    result = normalized * gain
    print(f"应用gain后:\n{result}")
    print()


def demo_rmsnorm_vs_layernorm():
    """对比RMSNorm和LayerNorm"""
    print("=== RMSNorm vs LayerNorm 对比 ===\n")
    
    batch_size, seq_len, d_model = 2, 3, 4
    x = torch.randn(batch_size, seq_len, d_model)
    
    # LayerNorm (PyTorch内置)
    layernorm = nn.LayerNorm(d_model)
    layernorm_result = layernorm(x)
    
    # RMSNorm (手动实现)
    def manual_rmsnorm(x, eps=1e-5):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # 计算RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        
        # 归一化
        normalized = x / rms
        
        # 应用gain (这里简化为1)
        result = normalized
        
        return result.to(in_dtype)
    
    rmsnorm_result = manual_rmsnorm(x)
    
    print(f"输入形状: {x.shape}")
    print(f"LayerNorm结果:\n{layernorm_result}")
    print(f"RMSNorm结果:\n{rmsnorm_result}")
    print(f"两种方法结果不同: {not torch.allclose(layernorm_result, rmsnorm_result)}")
    print()


def demo_rmsnorm_implementation_hints():
    """提供实现提示"""
    print("=== RMSNorm实现提示 ===\n")
    
    print("__init__方法需要实现:")
    print("1. 初始化gain参数:")
    print("   self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))")
    print()
    
    print("forward方法需要实现:")
    print("1. 保存原始数据类型:")
    print("   in_dtype = x.dtype")
    print("   x = x.to(torch.float32)")
    print()
    
    print("2. 计算RMS:")
    print("   rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)")
    print()
    
    print("3. 归一化:")
    print("   normalized = x / rms")
    print()
    
    print("4. 应用gain:")
    print("   result = normalized * self.gain")
    print()
    
    print("5. 转换回原始类型:")
    print("   return result.to(in_dtype)")
    print()
    
    print("完整实现:")
    print("```python")
    print("def forward(self, x: torch.Tensor) -> torch.Tensor:")
    print("    in_dtype = x.dtype")
    print("    x = x.to(torch.float32)")
    print("    ")
    print("    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)")
    print("    normalized = x / rms")
    print("    result = normalized * self.gain")
    print("    ")
    print("    return result.to(in_dtype)")
    print("```")


def demo_rmsnorm_properties():
    """演示RMSNorm的特性"""
    print("=== RMSNorm特性演示 ===\n")
    
    # 创建测试数据
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # 形状: (1, 1, 4)
    print(f"输入: {x}")
    
    # 手动计算RMS
    d_model = 4
    eps = 1e-5
    squared = x ** 2
    mean_squared = torch.mean(squared, dim=-1, keepdim=True)
    rms = torch.sqrt(mean_squared + eps)
    
    print(f"平方: {squared}")
    print(f"平均平方: {mean_squared}")
    print(f"RMS: {rms}")
    
    # 归一化
    normalized = x / rms
    print(f"归一化后: {normalized}")
    
    # 验证归一化后的RMS
    normalized_rms = torch.sqrt(torch.mean(normalized**2, dim=-1, keepdim=True) + eps)
    print(f"归一化后的RMS: {normalized_rms}")
    print(f"归一化后RMS接近1: {torch.allclose(normalized_rms, torch.ones_like(normalized_rms), atol=1e-4)}")


if __name__ == "__main__":
    demo_rmsnorm_math()
    demo_rmsnorm_vs_layernorm()
    demo_rmsnorm_implementation_hints()
    demo_rmsnorm_properties()
