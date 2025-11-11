"""
Data loading utilities for training Transformer language models.

Efficiently samples batches from tokenized sequences.
"""

import torch
import numpy as np
from numpy.typing import NDArray


def get_batch(
    data: NDArray[np.int_],
    batch_size: int,
    context_length: int,
    device: str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of sequences from tokenized data.
    
    Given a long sequence of tokens x = (x_1, ..., x_n), samples B random
    subsequences of length m, along with their corresponding next tokens.
    
    Args:
        data: Token array, shape (n,) with token IDs
              Can be memory-mapped for large datasets (np.memmap)
        batch_size: Number of sequences to sample (B)
        context_length: Length of each sequence (m)
        device: PyTorch device string ('cpu', 'cuda', 'cuda:0', 'mps', etc.)
    
    Returns:
        inputs: Input sequences, shape (batch_size, context_length)
                Contains tokens x_i
        targets: Target sequences, shape (batch_size, context_length)
                 Contains next tokens x_{i+1}
    
    Example:
        >>> # Load data (memory-mapped for large datasets)
        >>> data = np.load('tokens.npy', mmap_mode='r')
        >>> 
        >>> # Sample a batch
        >>> inputs, targets = get_batch(
        ...     data=data,
        ...     batch_size=32,
        ...     context_length=512,
        ...     device='cuda:0'
        ... )
        >>> 
        >>> # Use in training
        >>> logits = model(inputs)
        >>> loss = cross_entropy(logits, targets)
    
    Batch Structure:
        Given x = [1, 2, 3, 4, 5, 6, 7, 8, ...]
        With batch_size=2, context_length=3:
        
        inputs:  [[2, 3, 4],    targets: [[3, 4, 5],
                  [5, 6, 7]]              [6, 7, 8]]
    
    Note:
        - Randomly samples starting positions
        - All sequences have same length (no padding needed)
        - Input and target are offset by 1 token
        - Efficient with memory-mapped arrays (lazy loading)
    
    Device Support:
        - 'cpu': Standard CPU
        - 'cuda' or 'cuda:0': NVIDIA GPU
    """
    # TODO: Implement batch sampling
    # Hints:
    # 1. Randomly sample B starting positions from [0, n - context_length)
    # 2. Extract sequences: inputs[i] = data[start:start+context_length]
    # 3. Extract targets: targets[i] = data[start+1:start+context_length+1]
    # 4. Convert to PyTorch tensors and move to device
    # 5. Return (inputs, targets)
    
    # randomly sample B starting positions from [0, n - context_length)
    start_positions = np.random.randint(0, len(data) - context_length, batch_size)

    # extract sequences: inputs[i] = data[start:start+context_length]
    inputs = np.array([data[i:i+context_length] for i in start_positions])
    # convert to PyTorch tensors and move to device
    # Use dtype=torch.long for token indices (required for embedding lookup)
    inputs = torch.from_numpy(inputs).long().to(device) # (batch_size, context_length)

    # extract targets: targets[i] = data[start+1:start+context_length+1]
    targets = np.array([data[i+1:i+context_length+1] for i in start_positions]) # (batch_size, context_length)
    # convert to PyTorch tensors and move to device
    # Use dtype=torch.long for token indices (required for embedding lookup)
    targets = torch.from_numpy(targets).long().to(device) # (batch_size, context_length)

    return inputs, targets
