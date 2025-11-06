"""
Text generation utilities for Transformer language models.

Implements various decoding strategies:
- Temperature sampling
- Top-p (nucleus) sampling
- Autoregressive generation
"""

import torch
import torch.nn.functional as F
from typing import Optional


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = 'cpu',
    eos_token_id: Optional[int] = None,
) -> str:
    """
    Generate text from a trained language model.
    
    Performs autoregressive generation:
    1. Encode prompt to token IDs
    2. For each step:
       - Run model to get logits for next token
       - Apply temperature scaling
       - Apply top-p filtering (optional)
       - Sample next token
       - Append to sequence
    3. Stop when EOS token or max_tokens reached
    4. Decode token IDs back to text
    
    Args:
        model: Trained TransformerLM
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Input text to continue from
        max_tokens: Maximum number of tokens to generate (default: 100)
        temperature: Sampling temperature (default: 1.0)
            - temperature → 0: Greedy (most likely tokens)
            - temperature = 1.0: Standard sampling
            - temperature > 1.0: More random/diverse
        top_p: Nucleus sampling threshold (default: 1.0, no filtering)
            - top_p = 1.0: Use full vocabulary
            - top_p = 0.9: Use top 90% probability mass
        device: Device to run on ('cpu', 'cuda', 'mps')
        eos_token_id: End-of-sequence token ID to stop generation
                      If None, uses tokenizer.eos_token_id
    
    Returns:
        text: Generated text string (includes prompt)
    
    Examples:
        >>> # Greedy decoding (deterministic)
        >>> text = generate_text(
        ...     model, tokenizer, "Once upon a time",
        ...     temperature=0.0, max_tokens=100
        ... )
        >>> 
        >>> # Diverse sampling
        >>> text = generate_text(
        ...     model, tokenizer, "Once upon a time",
        ...     temperature=0.8, top_p=0.9, max_tokens=100
        ... )
        >>> 
        >>> # Very creative (high temperature)
        >>> text = generate_text(
        ...     model, tokenizer, "Once upon a time",
        ...     temperature=1.5, max_tokens=100
        ... )
    """
    # TODO: Implement text generation
    # Hints:
    # 1. Encode prompt to token IDs
    # 2. Loop for max_tokens iterations:
    #    a. Get logits from model (last position)
    #    b. Apply temperature scaling
    #    c. Apply top-p filtering
    #    d. Sample next token
    #    e. Check for EOS token
    #    f. Append to sequence
    # 3. Decode token IDs to text
    
    raise NotImplementedError("Text generation not yet implemented")


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    
    Scaled softmax:
        softmax(v, τ)_i = exp(v_i / τ) / Σ exp(v_j / τ)
    
    Args:
        logits: Raw logits from model, shape (vocab_size,) or (..., vocab_size)
        temperature: Temperature parameter (τ)
            - temperature → 0: Peaked at maximum (greedy)
            - temperature = 1.0: Standard softmax
            - temperature > 1.0: Flatter distribution (more random)
    
    Returns:
        scaled_logits: Temperature-scaled logits, same shape as input
    
    Example:
        >>> logits = model(input_ids)[-1]  # Last position
        >>> scaled_logits = apply_temperature(logits, temperature=0.8)
        >>> probs = F.softmax(scaled_logits, dim=-1)
    """
    return logits / temperature


def top_p_filtering(
    logits: torch.Tensor,
    top_p: float,
    filter_value: float = -float('Inf'),
) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling filtering.
    
    Keeps only the smallest set of tokens whose cumulative probability
    exceeds threshold p, setting all other logits to filter_value.
    
    Algorithm:
        1. Compute probabilities via softmax
        2. Sort by probability (descending)
        3. Compute cumulative probabilities
        4. Find cutoff where cumsum >= p
        5. Mask out tokens below cutoff
    
    Args:
        logits: Raw logits, shape (vocab_size,) or (..., vocab_size)
        top_p: Probability threshold (0 < p ≤ 1.0)
            - top_p = 1.0: Keep all tokens (no filtering)
            - top_p = 0.9: Keep top 90% probability mass
        filter_value: Value to set filtered logits to (default: -inf)
    
    Returns:
        filtered_logits: Logits with low-probability tokens masked, same shape
    
    Example:
        >>> logits = model(input_ids)[-1]  # (vocab_size,)
        >>> filtered = top_p_filtering(logits, top_p=0.9)
        >>> probs = F.softmax(filtered, dim=-1)
        >>> next_token = torch.multinomial(probs, num_samples=1)
    
    Reference:
        Holtzman et al. (2020). "The Curious Case of Neural Text Degeneration"
    """
    # TODO: Implement top-p filtering
    # Hints:
    # 1. Sort logits by probability (descending)
    # 2. Compute cumulative probabilities
    # 3. Find tokens to keep (cumsum < top_p)
    # 4. Mask out filtered tokens
    
    raise NotImplementedError("Top-p filtering not yet implemented")

