"""
BPE (Byte-Pair Encoding) 模块

包含BPE训练的各种实现：
- 基础实现
- 优化实现  
- 多进程实现
- 训练脚本
"""

from .core import train_bpe
from .optimized import OptimizedBPETrainer, train_bpe_optimized
from .trainer import BPETrainer, train_bpe_with_progress

__all__ = [
    'train_bpe',
    'OptimizedBPETrainer', 
    'train_bpe_optimized',
    'BPETrainer',
    'train_bpe_with_progress'
]
