"""
Learning rate schedulers for training Transformer language models.

This module implements cosine annealing schedule with linear warmup.
"""

import math


def get_lr_cosine_schedule(
    t: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine annealing learning rate schedule with linear warmup.
    
    Used in LLaMA [Touvron et al., 2023] and other modern LLMs.
    
    Schedule:
        - Warmup (t < T_w): Linear increase from 0 to α_max
        - Cosine (T_w ≤ t ≤ T_c): Cosine decay from α_max to α_min
        - Post-anneal (t > T_c): Constant α_min
    
    Mathematical definition:
                  ⎧ (t / T_w) * α_max                                    if t < T_w
        α_t =     ⎨ α_min + 0.5 * (1 + cos((t-T_w)/(T_c-T_w)*π)) * (α_max - α_min)  if T_w ≤ t ≤ T_c
                  ⎩ α_min                                                if t > T_c
    
    Args:
        t: Current iteration number (starting from 0)
        max_lr: Maximum learning rate (α_max), reached after warmup
        min_lr: Minimum learning rate (α_min), floor value
        warmup_iters: Number of warmup iterations (T_w)
        cosine_cycle_iters: Total iterations for cosine cycle (T_c)
    
    Returns:
        lr: Learning rate for iteration t
    
    Example:
        >>> # Typical settings: 5k warmup, 100k total
        >>> for step in range(100000):
        >>>     lr = get_lr_cosine_schedule(
        >>>         t=step,
        >>>         max_lr=3e-4,
        >>>         min_lr=3e-5,
        >>>         warmup_iters=5000,
        >>>         cosine_cycle_iters=100000
        >>>     )
        >>>     # Update optimizer
        >>>     for param_group in optimizer.param_groups:
        >>>         param_group['lr'] = lr
    
    Visualization:
        α
        │    ╱╲
        │   ╱  ╲___
        │  ╱       ╲___
        │ ╱            ╲___
        │╱                 ────────
        └──────────────────────────> t
         warmup  cosine    constant
    """
    # TODO: Implement cosine annealing schedule with warmup
    # Hints:
    # 1. Warmup phase: linear increase from 0 to max_lr
    # 2. Cosine phase: smooth decay using cosine function
    # 3. Post-anneal: constant min_lr
    
    raise NotImplementedError("Learning rate scheduler not yet implemented")

