#!/usr/bin/env python3
"""
Demo script to understand the Embedding module implementation.
"""

import torch
import torch.nn as nn


def demo_embedding_concepts():
    """演示Embedding模块的核心概念"""
    print("=== Embedding模块核心概念演示 ===\n")
    
    # 1. 基本概念
    print("1. 基本概念:")
    print("   - Embedding将离散的token ID映射为连续的向量")
    print("   - 每个token ID对应一个d_model维的向量")
    print("   - 权重矩阵形状: (vocab_size, d_model)")
    print()
    
    # 2. 手动实现embedding lookup
    print("2. 手动实现embedding lookup:")
    vocab_size = 5
    d_model = 3
    
    # 创建embedding矩阵
    embedding_matrix = torch.randn(vocab_size, d_model)
    print(f"Embedding矩阵形状: {embedding_matrix.shape}")
    print(f"Embedding矩阵:\n{embedding_matrix}")
    print()
    
    # 创建token IDs
    token_ids = torch.tensor([0, 2, 1, 4])  # 形状: (4,)
    print(f"Token IDs: {token_ids}")
    print(f"Token IDs形状: {token_ids.shape}")
    print()
    
    # 方法1: 使用高级索引
    print("方法1: 使用高级索引 self.weight[token_ids]")
    embeddings1 = embedding_matrix[token_ids]
    print(f"结果形状: {embeddings1.shape}")
    print(f"结果:\n{embeddings1}")
    print()
    
    # 方法2: 使用torch.index_select
    print("方法2: 使用torch.index_select")
    embeddings2 = torch.index_select(embedding_matrix, 0, token_ids)
    print(f"结果形状: {embeddings2.shape}")
    print(f"结果:\n{embeddings2}")
    print(f"两种方法结果相同: {torch.allclose(embeddings1, embeddings2)}")
    print()
    
    # 3. 批处理示例
    print("3. 批处理示例:")
    batch_token_ids = torch.tensor([[0, 1, 2], [3, 4, 0]])  # 形状: (2, 3)
    print(f"批处理Token IDs形状: {batch_token_ids.shape}")
    print(f"批处理Token IDs:\n{batch_token_ids}")
    
    batch_embeddings = embedding_matrix[batch_token_ids]
    print(f"批处理结果形状: {batch_embeddings.shape}")
    print(f"批处理结果:\n{batch_embeddings}")
    print()
    
    # 4. 与PyTorch内置Embedding对比
    print("4. 与PyTorch内置Embedding对比:")
    pytorch_embedding = nn.Embedding(vocab_size, d_model)
    pytorch_embedding.weight.data = embedding_matrix
    
    pytorch_result = pytorch_embedding(token_ids)
    print(f"PyTorch结果形状: {pytorch_result.shape}")
    print(f"PyTorch结果:\n{pytorch_result}")
    print(f"与手动实现相同: {torch.allclose(embeddings1, pytorch_result)}")
    print()
    
    # 5. 实现提示
    print("5. 实现提示:")
    print("   在forward方法中，你需要:")
    print("   - 输入: token_ids 形状 (..., )")
    print("   - 输出: embeddings 形状 (..., d_model)")
    print("   - 使用: self.weight[token_ids] 或 torch.index_select")
    print("   - 注意: 输入可以有任意数量的批处理维度")


def demo_embedding_implementation():
    """演示如何实现Embedding模块"""
    print("\n=== Embedding模块实现示例 ===\n")
    
    print("def forward(self, token_ids: torch.Tensor) -> torch.Tensor:")
    print("    \"\"\"")
    print("    Lookup embedding vectors for the given token IDs.")
    print("    \"\"\"")
    print("    # 方法1: 使用高级索引 (推荐)")
    print("    return self.weight[token_ids]")
    print()
    print("    # 方法2: 使用torch.index_select")
    print("    # return torch.index_select(self.weight, 0, token_ids)")
    print()
    print("    # 方法3: 使用torch.gather (不推荐，更复杂)")
    print("    # return torch.gather(self.weight, 0, token_ids.unsqueeze(-1).expand(-1, self.embedding_dim))")


if __name__ == "__main__":
    demo_embedding_concepts()
    demo_embedding_implementation()
