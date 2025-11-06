"""
Optimizers for training Transformer language models.

This module implements:
- SGD with learning rate decay
- AdamW with decoupled weight decay
"""

import torch
import math
from typing import Optional, Callable, Iterable


class SGD(torch.optim.Optimizer):
    """
    Stochastic Gradient Descent with learning rate decay.
    
    Implements:
        θ_{t+1} = θ_t - (α / √(t+1)) * ∇L(θ_t; B_t)
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (α) (default: 1e-3)
    
    Example:
        >>> optimizer = SGD(model.parameters(), lr=1e-3)
        >>> for step in range(num_steps):
        >>>     optimizer.zero_grad()
        >>>     loss = compute_loss()
        >>>     loss.backward()
        >>>     optimizer.step()
    """
    
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional callable that reevaluates the model and returns loss
        
        Returns:
            loss: Loss value if closure is provided, else None
        """
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]  # Get learning rate
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]  # Get state associated with parameter
                t = state.get("t", 0)  # Get iteration number (default 0)
                grad = p.grad.data  # Get gradient
                
                # Update: θ = θ - (lr / √(t+1)) * grad
                p.data -= lr / math.sqrt(t + 1) * grad
                
                # Increment iteration number
                state["t"] = t + 1
        
        return loss


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer with decoupled weight decay.
    
    Implements Algorithm 1 from Loshchilov & Hutter (2019):
        m ← β_1 * m + (1 - β_1) * g              # First moment
        v ← β_2 * v + (1 - β_2) * g²             # Second moment
        α_t ← α * √(1 - β_2^t) / (1 - β_1^t)     # Bias correction
        θ ← θ - α_t * m / (√v + ε)               # Parameter update
        θ ← θ - α * λ * θ                        # Weight decay
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (α) (default: 1e-3)
        betas: Coefficients for moment estimates (β_1, β_2) (default: (0.9, 0.999))
        eps: Term for numerical stability (ε) (default: 1e-8)
        weight_decay: Weight decay coefficient (λ) (default: 0.0)
    
    Example:
        >>> # Standard settings
        >>> optimizer = AdamW(model.parameters(), lr=3e-4, 
        ...                   betas=(0.9, 0.999), weight_decay=0.1)
        >>> 
        >>> # LLaMA/GPT-3 settings
        >>> optimizer = AdamW(model.parameters(), lr=3e-4,
        ...                   betas=(0.9, 0.95), weight_decay=0.1)
    
    Note:
        - Iteration counter t starts at 1 (not 0)
        - Requires 3× parameter memory (params + m + v)
        - Weight decay is decoupled from gradient update
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional callable that reevaluates the model and returns loss
        
        Returns:
            loss: Loss value if closure is provided, else None
        """
        loss = None if closure is None else closure()
        
        # TODO: Implement AdamW update step
        # Hints:
        # 1. Iterate over parameter groups and parameters
        # 2. For each parameter, get or initialize state (m, v, t)
        # 3. Update moments: m, v
        # 4. Compute bias-corrected learning rate: α_t
        # 5. Update parameter with Adam step
        # 6. Apply weight decay
        # 7. Increment iteration counter t
        
        raise NotImplementedError("AdamW optimizer not yet implemented")
        
        return loss

