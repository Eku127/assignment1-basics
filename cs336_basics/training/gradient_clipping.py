"""
Gradient clipping utilities for stable training.

Prevents exploding gradients by clipping the global gradient norm.
"""

import torch
from typing import Iterable


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> float:
    """
    Clip gradients by global norm.
    
    Computes the L2 norm of all gradients combined and scales them down
    if the norm exceeds max_norm.
    
    Algorithm:
        ∥g∥_2 = √(Σ_p ∥g_p∥²)  for all parameters p
        
        If ∥g∥_2 > M:
            g ← g * M / (∥g∥_2 + ε)
    
    Args:
        parameters: Iterable of parameters (typically model.parameters())
        max_norm: Maximum allowed gradient norm (M)
        eps: Small value for numerical stability (default: 1e-6)
    
    Returns:
        total_norm: The global gradient norm before clipping
    
    Example:
        >>> # In training loop
        >>> loss.backward()
        >>> total_norm = clip_gradients(model.parameters(), max_norm=1.0)
        >>> optimizer.step()
        >>> 
        >>> # Check if clipping occurred
        >>> if total_norm > 1.0:
        >>>     print(f"Clipped gradients: {total_norm:.2f} -> 1.0")
    
    Note:
        - Modifies gradients in-place
        - Applied after backward() but before optimizer.step()
        - Helps prevent training instabilities
    """
    # TODO: Implement gradient clipping
    # Hints:
    # 1. Collect all gradients that are not None
    # 2. Compute global L2 norm: √(Σ ∥g_p∥²)
    # 3. If norm > max_norm, scale all gradients by max_norm / (norm + eps)
    # 4. Return the original norm (before clipping)

    # clloect gradient
    gradients = [p.grad for p in parameters if p.grad is not None]
    # compute global L2 norm: √(Σ ∥g_p∥²)
    global_norm = torch.sqrt(sum(grad.norm()**2 for grad in gradients))
    # if global norm > max norm, then  we scale all gradients by max norm / (global norm + eps)
    if global_norm > max_norm:
        for grad in gradients:
            grad.data *= max_norm / (global_norm + eps)
    # return the original global norm (before clipping)
    return global_norm