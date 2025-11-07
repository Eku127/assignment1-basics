"""
Checkpointing utilities for saving and restoring training state.

Enables resumable training by saving model, optimizer, and training state.
"""

import torch
import os
from typing import BinaryIO, Union
from pathlib import Path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO],
) -> None:
    """
    Save a training checkpoint.
    
    Saves all state needed to resume training:
    - Model weights
    - Optimizer state (including moment estimates for AdamW)
    - Current iteration number
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer to save (includes state like AdamW moments)
        iteration: Current training step/iteration
        out: Destination path (str/Path) or file-like object
    
    Example:
        >>> # Save checkpoint
        >>> save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     iteration=10000,
        ...     out='checkpoint_10000.pt'
        ... )
        >>> 
        >>> # Save to file-like object
        >>> with open('checkpoint.pt', 'wb') as f:
        ...     save_checkpoint(model, optimizer, iteration, f)
    
    Checkpoint Structure:
        {
            'model': model.state_dict(),        # Model weights
            'optimizer': optimizer.state_dict(),  # Optimizer state
            'iteration': iteration,              # Training step
        }
    
    Note:
        - Use model.state_dict() to get weights
        - Use optimizer.state_dict() to get optimizer state
        - torch.save() handles both paths and file objects
    """
    # TODO: Implement checkpoint saving
    # Hints:
    # 1. Create checkpoint dictionary with model, optimizer, iteration
    # 2. Use torch.save(checkpoint, out)

    # get the checkpoint dictionary with model, optimizer, iteration
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    # use torch.save(checkpoint, out) to save the checkpoint
    torch.save(checkpoint, out)

def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load a training checkpoint.
    
    Restores training state from a saved checkpoint:
    - Model weights
    - Optimizer state
    - Training iteration number
    
    Args:
        src: Source path (str/Path) or file-like object
        model: PyTorch model to restore weights into
        optimizer: Optimizer to restore state into
    
    Returns:
        iteration: The training step that was saved in the checkpoint
    
    Example:
        >>> # Load checkpoint and resume training
        >>> iteration = load_checkpoint(
        ...     src='checkpoint_10000.pt',
        ...     model=model,
        ...     optimizer=optimizer
        ... )
        >>> print(f"Resuming from iteration {iteration}")
        >>> 
        >>> # Continue training
        >>> for step in range(iteration, max_steps):
        ...     train_step()
    
    Note:
        - Use torch.load(src) to load checkpoint
        - Use model.load_state_dict() to restore weights
        - Use optimizer.load_state_dict() to restore optimizer state
        - Model and optimizer must have same structure as when saved
    """
    # TODO: Implement checkpoint loading
    # Hints:
    # 1. Load checkpoint with torch.load(src)
    # 2. Restore model with model.load_state_dict(checkpoint['model'])
    # 3. Restore optimizer with optimizer.load_state_dict(checkpoint['optimizer'])
    # 4. Return checkpoint['iteration']

    # load the checkpoint
    checkpoint = torch.load(src)
    # restore the model with model.load_state_dict(checkpoint['model'])
    model.load_state_dict(checkpoint['model'])
    # restore the optimizer with optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # return the iteration
    return checkpoint['iteration']