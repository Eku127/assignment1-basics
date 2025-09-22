#!/usr/bin/env python3
"""
Demo script to understand Multi-Head Self-Attention implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def demo_attention_basics():
    """演示注意力机制基础"""
    print("=== 注意力机制基础演示 ===\n")
    
    # 1. 基本注意力公式
    print("1. 注意力公式:")
    print("   Attention(Q, K, V) = softmax(QK^T/√d_k)V")
    print()
    
    # 2. 简单例子
    batch_size, seq_len, d_k = 2, 3, 4
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    print(f"Q 形状: {Q.shape}")
    print(f"K 形状: {K.shape}")
    print(f"V 形状: {V.shape}")
    print()
    
    # 3. 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1))
    print(f"注意力分数 QK^T 形状: {scores.shape}")
    print(f"注意力分数:\n{scores}")
    print()
    
    # 4. 缩放
    scaled_scores = scores / math.sqrt(d_k)
    print(f"缩放后分数 形状: {scaled_scores.shape}")
    print()
    
    # 5. 应用softmax
    attention_weights = F.softmax(scaled_scores, dim=-1)
    print(f"注意力权重 形状: {attention_weights.shape}")
    print(f"每行和: {torch.sum(attention_weights, dim=-1)}")
    print()
    
    # 6. 应用权重到值
    output = torch.matmul(attention_weights, V)
    print(f"最终输出 形状: {output.shape}")
    print()


def demo_multi_head_concept():
    """演示多头注意力概念"""
    print("=== 多头注意力概念演示 ===\n")
    
    # 1. 多头注意力公式
    print("1. 多头注意力公式:")
    print("   MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O")
    print("   其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)")
    print()
    
    # 2. 参数设置
    d_model = 8
    num_heads = 2
    d_k = d_v = d_model // num_heads  # 4
    
    print(f"参数设置:")
    print(f"  d_model = {d_model}")
    print(f"  num_heads = {num_heads}")
    print(f"  d_k = d_v = {d_k}")
    print()
    
    # 3. 模拟多头处理
    batch_size, seq_len = 1, 3
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"输入 x 形状: {x.shape}")
    print()
    
    # 4. 模拟每个头的处理
    print("2. 每个头的处理:")
    for head in range(num_heads):
        start_idx = head * d_k
        end_idx = (head + 1) * d_k
        
        # 模拟每个头的Q, K, V
        Q_head = x[..., start_idx:end_idx]  # 模拟投影后的Q
        K_head = x[..., start_idx:end_idx]  # 模拟投影后的K
        V_head = x[..., start_idx:end_idx]  # 模拟投影后的V
        
        print(f"  头 {head + 1}:")
        print(f"    Q_head 形状: {Q_head.shape}")
        print(f"    K_head 形状: {K_head.shape}")
        print(f"    V_head 形状: {V_head.shape}")
        
        # 计算注意力
        scores = torch.matmul(Q_head, K_head.transpose(-2, -1))
        attention_weights = F.softmax(scores / math.sqrt(d_k), dim=-1)
        head_output = torch.matmul(attention_weights, V_head)
        
        print(f"    头输出形状: {head_output.shape}")
        print()
    
    # 5. 合并所有头
    print("3. 合并所有头:")
    print("   - 将所有头的输出拼接")
    print("   - 应用输出投影 W^O")
    print("   - 得到最终输出")
    print()


def demo_causal_masking():
    """演示因果掩码"""
    print("=== 因果掩码演示 ===\n")
    
    # 1. 因果掩码概念
    print("1. 因果掩码概念:")
    print("   - 防止模型看到未来token")
    print("   - 位置i只能关注位置j ≤ i")
    print("   - 用于自回归语言建模")
    print()
    
    # 2. 创建因果掩码
    seq_len = 4
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    print(f"序列长度: {seq_len}")
    print(f"因果掩码形状: {causal_mask.shape}")
    print(f"因果掩码:\n{causal_mask}")
    print()
    
    # 3. 解释掩码含义
    print("2. 掩码含义:")
    print("   True = 可以注意")
    print("   False = 屏蔽")
    print()
    for i in range(seq_len):
        can_attend = torch.where(causal_mask[i] == False)[0].tolist()
        print(f"   位置 {i} 可以注意位置: {can_attend}")
    print()
    
    # 4. 应用掩码到注意力分数
    print("3. 应用掩码到注意力分数:")
    scores = torch.randn(seq_len, seq_len)
    print(f"原始分数:\n{scores}")
    
    # 将掩码位置设为-inf
    masked_scores = scores.masked_fill(causal_mask, float('-inf'))
    print(f"掩码后分数:\n{masked_scores}")
    
    # 应用softmax
    attention_weights = F.softmax(masked_scores, dim=-1)
    print(f"注意力权重:\n{attention_weights}")
    print(f"每行和: {torch.sum(attention_weights, dim=-1)}")
    print()


def demo_implementation_hints():
    """提供实现提示"""
    print("=== Multi-Head Attention实现提示 ===\n")
    
    print("__init__方法需要实现:")
    print("1. 初始化投影权重矩阵:")
    print("   self.WQ = nn.Parameter(torch.empty(d_model, d_model))")
    print("   self.WK = nn.Parameter(torch.empty(d_model, d_model))")
    print("   self.WV = nn.Parameter(torch.empty(d_model, d_model))")
    print("   self.WO = nn.Parameter(torch.empty(d_model, d_model))")
    print()
    
    print("2. 权重初始化:")
    print("   init.xavier_uniform_(self.WQ)")
    print("   init.xavier_uniform_(self.WK)")
    print("   init.xavier_uniform_(self.WV)")
    print("   init.xavier_uniform_(self.WO)")
    print()
    
    print("forward方法需要实现:")
    print("1. QKV投影:")
    print("   Q = torch.einsum('...sd,dd->...sd', x, self.WQ)")
    print("   K = torch.einsum('...sd,dd->...sd', x, self.WK)")
    print("   V = torch.einsum('...sd,dd->...sd', x, self.WV)")
    print()
    
    print("2. 重塑为多头:")
    print("   Q = Q.view(..., seq_len, num_heads, d_k).transpose(-3, -2)")
    print("   K = K.view(..., seq_len, num_heads, d_k).transpose(-3, -2)")
    print("   V = V.view(..., seq_len, num_heads, d_v).transpose(-3, -2)")
    print()
    
    print("3. 创建因果掩码:")
    print("   mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()")
    print()
    
    print("4. 应用注意力:")
    print("   attention_output = scaled_dot_product_attention(Q, K, V, mask)")
    print()
    
    print("5. 合并头:")
    print("   attention_output = attention_output.transpose(-3, -2).contiguous()")
    print("   attention_output = attention_output.view(..., seq_len, d_model)")
    print()
    
    print("6. 输出投影:")
    print("   output = torch.einsum('...sd,dd->...sd', attention_output, self.WO)")
    print()


def demo_rope_integration():
    """演示RoPE集成"""
    print("=== RoPE集成演示 ===\n")
    
    print("1. RoPE在Multi-Head Attention中的应用:")
    print("   - 只对Q和K应用RoPE，不对V应用")
    print("   - 将头维度作为批次维度处理")
    print("   - 每个头应用相同的旋转")
    print()
    
    print("2. 实现步骤:")
    print("   Step 1: 重塑Q, K为 (..., num_heads, seq_len, d_k)")
    print("   Step 2: 将头维度视为批次维度")
    print("   Step 3: 应用RoPE到Q和K")
    print("   Step 4: 继续正常的注意力计算")
    print()
    
    print("3. 代码示例:")
    print("   # 重塑Q, K")
    print("   Q_reshaped = Q.view(..., num_heads, seq_len, d_k)")
    print("   K_reshaped = K.view(..., num_heads, seq_len, d_k)")
    print()
    print("   # 应用RoPE (将头维度作为批次)")
    print("   Q_rope = rope(Q_reshaped, positions)")
    print("   K_rope = rope(K_reshaped, positions)")
    print()
    print("   # 继续注意力计算")
    print("   attention_output = scaled_dot_product_attention(Q_rope, K_rope, V, mask)")
    print()


def demo_efficiency_tips():
    """演示效率优化提示"""
    print("=== 效率优化提示 ===\n")
    
    print("1. 批量矩阵乘法:")
    print("   - 使用torch.bmm或torch.einsum进行批量操作")
    print("   - 避免循环处理每个头")
    print()
    
    print("2. 内存效率:")
    print("   - 使用in-place操作减少内存使用")
    print("   - 合理使用contiguous()")
    print()
    
    print("3. 数值稳定性:")
    print("   - 使用稳定的softmax实现")
    print("   - 注意掩码的应用顺序")
    print()
    
    print("4. 形状管理:")
    print("   - 仔细管理张量形状变化")
    print("   - 使用view和transpose的组合")
    print("   - 确保维度匹配")
    print()


if __name__ == "__main__":
    demo_attention_basics()
    demo_multi_head_concept()
    demo_causal_masking()
    demo_implementation_hints()
    demo_rope_integration()
    demo_efficiency_tips()
