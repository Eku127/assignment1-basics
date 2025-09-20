#!/usr/bin/env python3
"""
演示 nn.Parameter 和普通张量的区别
"""

import torch
import torch.nn as nn


class LinearWithParameter(nn.Module):
    """使用 nn.Parameter 的版本"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
    
    def forward(self, x):
        return torch.einsum('...i,oi->...o', x, self.weight)


class LinearWithoutParameter(nn.Module):
    """不使用 nn.Parameter 的版本"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.randn(out_features, in_features)  # 普通张量
    
    def forward(self, x):
        return torch.einsum('...i,oi->...o', x, self.weight)


def demo_parameter_differences():
    print("=== nn.Parameter vs 普通张量对比 ===\n")
    
    # 创建两个模型
    model_with_param = LinearWithParameter(4, 3)
    model_without_param = LinearWithoutParameter(4, 3)
    
    print("1. 参数注册情况:")
    print(f"使用 nn.Parameter 的模型参数数量: {len(list(model_with_param.parameters()))}")
    print(f"不使用 nn.Parameter 的模型参数数量: {len(list(model_without_param.parameters()))}")
    
    print("\n2. 参数名称:")
    for name, param in model_with_param.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print("\n3. 梯度计算测试:")
    x = torch.randn(2, 4)
    target = torch.randn(2, 3)
    
    # 测试使用 nn.Parameter 的模型
    output1 = model_with_param(x)
    loss1 = torch.nn.functional.mse_loss(output1, target)
    loss1.backward()
    print(f"使用 nn.Parameter 的模型梯度: {model_with_param.weight.grad is not None}")
    
    # 测试不使用 nn.Parameter 的模型
    output2 = model_without_param(x)
    loss2 = torch.nn.functional.mse_loss(output2, target)
    loss2.backward()
    print(f"不使用 nn.Parameter 的模型梯度: {model_without_param.weight.grad is not None}")
    
    print("\n4. 优化器测试:")
    try:
        optimizer1 = torch.optim.Adam(model_with_param.parameters())
        print("使用 nn.Parameter 的模型可以创建优化器: ✓")
    except Exception as e:
        print(f"使用 nn.Parameter 的模型创建优化器失败: {e}")
    
    try:
        optimizer2 = torch.optim.Adam(model_without_param.parameters())
        print("不使用 nn.Parameter 的模型可以创建优化器: ✓")
    except Exception as e:
        print(f"不使用 nn.Parameter 的模型创建优化器失败: {e}")
    
    print("\n5. 模型保存测试:")
    state_dict1 = model_with_param.state_dict()
    state_dict2 = model_without_param.state_dict()
    print(f"使用 nn.Parameter 的模型 state_dict 键: {list(state_dict1.keys())}")
    print(f"不使用 nn.Parameter 的模型 state_dict 键: {list(state_dict2.keys())}")


if __name__ == "__main__":
    demo_parameter_differences()
