"""
Training infrastructure for Transformer Language Models

This module provides all the necessary components for training a Transformer LM:
- Loss functions (cross-entropy)
- Optimizers (SGD, AdamW)
- Learning rate schedulers (cosine annealing with warmup)
- Gradient clipping
- Data loading utilities
- Checkpointing utilities
- Training loop infrastructure
- Text generation/decoding utilities
"""

from cs336_basics.training.loss import cross_entropy
from cs336_basics.training.optimizer import SGD, AdamW
from cs336_basics.training.lr_scheduler import get_lr_cosine_schedule
from cs336_basics.training.gradient_clipping import clip_gradients
from cs336_basics.training.data_loader import get_batch
from cs336_basics.training.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.training.decode import generate_text

__all__ = [
    # Loss
    "cross_entropy",
    # Optimizers
    "SGD",
    "AdamW",
    # Learning rate scheduling
    "get_lr_cosine_schedule",
    # Gradient clipping
    "clip_gradients",
    # Data loading
    "get_batch",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    # Text generation
    "generate_text",
]

