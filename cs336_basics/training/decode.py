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
    model.eval()
    
    # Determine EOS token ID
    if eos_token_id is None:
        # Try to get from tokenizer
        if hasattr(tokenizer, 'eos_token_id'):
            eos_token_id = tokenizer.eos_token_id
        else:
            # Try to encode <|endoftext|> token
            try:
                eos_tokens = tokenizer.encode("<|endoftext|>")
                if len(eos_tokens) > 0:
                    eos_token_id = eos_tokens[0]
            except:
                eos_token_id = None
    
    # 1. Encode prompt to token IDs
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)
    
    # 2. Autoregressive generation loop with KV cache
    past_key_values = None  # Initialize KV cache
    with torch.no_grad():
        for step in range(max_tokens):
            # a. Prepare input: for first step, use full prompt; for subsequent steps, use only new token
            if step == 0:
                # First step: process the entire prompt
                # Handle context length limitation
                if input_ids.size(1) > model.context_length:
                    # Truncate to fit context length
                    model_input = input_ids[:, -model.context_length:]
                else:
                    model_input = input_ids
            else:
                # Subsequent steps: only the newly generated token
                model_input = input_ids[:, -1:]  # (batch_size, 1)
            
            # b. Forward pass with KV cache
            logits, past_key_values = model(
                model_input,
                past_key_values=past_key_values,
                use_cache=True
            )  # (batch_size, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # c. Apply temperature scaling
            next_token_logits = apply_temperature(next_token_logits, temperature)
            
            # d. Apply top-p filtering
            if top_p < 1.0:
                next_token_logits = top_p_filtering(next_token_logits, top_p)
            
            # e. Sample next token
            probs = F.softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            # f. Check for EOS token
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            
            # g. Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    # 3. Decode token IDs to text
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


def generate_text_no_cache(
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
    Generate text from a trained language model WITHOUT using KV cache.
    
    This version processes the entire sequence at each step, which is simpler
    but less efficient than the KV cache version. Useful for debugging or
    when you want to ensure identical behavior to training.
    
    Performs autoregressive generation:
    1. Encode prompt to token IDs
    2. For each step:
       - Run model on entire sequence so far (no cache)
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
        >>> text = generate_text_no_cache(
        ...     model, tokenizer, "Once upon a time",
        ...     temperature=0.0, max_tokens=100
        ... )
        >>> 
        >>> # Diverse sampling
        >>> text = generate_text_no_cache(
        ...     model, tokenizer, "Once upon a time",
        ...     temperature=0.8, top_p=0.9, max_tokens=100
        ... )
    """
    model.eval()
    
    # Determine EOS token ID
    if eos_token_id is None:
        # Try to get from tokenizer
        if hasattr(tokenizer, 'eos_token_id'):
            eos_token_id = tokenizer.eos_token_id
        else:
            # Try to encode <|endoftext|> token
            try:
                eos_tokens = tokenizer.encode("<|endoftext|>")
                if len(eos_tokens) > 0:
                    eos_token_id = eos_tokens[0]
            except:
                eos_token_id = None
    
    # 1. Encode prompt to token IDs
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)
    
    # 2. Autoregressive generation loop WITHOUT KV cache
    with torch.no_grad():
        for step in range(max_tokens):
            # Handle context length limitation
            if input_ids.size(1) > model.context_length:
                # Truncate to fit context length (sliding window)
                model_input = input_ids[:, -model.context_length:]
            else:
                model_input = input_ids
            
            # Forward pass WITHOUT KV cache (use_cache=False)
            logits = model(
                model_input,
                past_key_values=None,
                use_cache=False
            )  # (batch_size, seq_len, vocab_size)
            
            # Get logits for the last token (next token prediction)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature scaling
            next_token_logits = apply_temperature(next_token_logits, temperature)
            
            # Apply top-p filtering
            if top_p < 1.0:
                next_token_logits = top_p_filtering(next_token_logits, top_p)
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            # Check for EOS token
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    # 3. Decode token IDs to text
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


def generate_text_v2(
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
    Generate text from a trained language model using the model's built-in generate method.
    
    This is an alternative implementation that uses model.generate() instead of
    manually implementing the generation loop. It provides the same interface
    as generate_text() but delegates the generation logic to the model.
    
    Args:
        model: Trained TransformerLM (must have a generate() method)
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Input text to continue from
        max_tokens: Maximum number of tokens to generate (default: 100)
        temperature: Sampling temperature (default: 1.0)
        top_p: Nucleus sampling threshold (default: 1.0, no filtering)
        device: Device to run on ('cpu', 'cuda', 'mps')
        eos_token_id: End-of-sequence token ID to stop generation
                      If None, uses tokenizer.eos_token_id
    
    Returns:
        text: Generated text string (includes prompt)
    
    Examples:
        >>> # Greedy decoding
        >>> text = generate_text_v2(
        ...     model, tokenizer, "Once upon a time",
        ...     temperature=0.01, max_tokens=100
        ... )
        >>> 
        >>> # Diverse sampling
        >>> text = generate_text_v2(
        ...     model, tokenizer, "Once upon a time",
        ...     temperature=0.8, top_p=0.9, max_tokens=100
        ... )
    """
    model.eval()
    
    # Determine EOS token ID
    if eos_token_id is None:
        # Try to get from tokenizer
        if hasattr(tokenizer, 'eos_token_id'):
            eos_token_id = tokenizer.eos_token_id
        else:
            # Try to encode <|endoftext|> token
            try:
                eos_tokens = tokenizer.encode("<|endoftext|>")
                if len(eos_tokens) > 0:
                    eos_token_id = eos_tokens[0]
            except:
                eos_token_id = None
    
    # 1. Encode prompt to token IDs
    prompt_ids = tokenizer.encode(prompt)
    prompt_tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, prompt_len)
    
    # 2. Call model's generate method
    with torch.no_grad():
        generated_tokens = model.generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p if top_p < 1.0 else None,
            eos_token_id=eos_token_id
        )  # (batch_size, prompt_len + num_generated)
    
    # 3. Decode token IDs to text
    generated_ids = generated_tokens[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


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
    # Handle edge case: very small temperature (greedy decoding)
    if temperature < 1e-8:
        temperature = 1e-8
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
    # No filtering needed if top_p is 1.0
    if top_p >= 1.0:
        return logits
    
    # 1. Compute probabilities via softmax
    probs = F.softmax(logits, dim=-1)
    
    # 2. Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # 3. Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 4. Find tokens to remove (cumulative probability > top_p)
    # We want to keep tokens until cumsum >= top_p
    # But we need to include the token that pushes us over the threshold
    # So we mask tokens where cumsum > top_p AND it's not the first token to exceed
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Keep at least the first token (highest probability)
    # Shift the mask to the right to keep the first token that exceeds threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 5. Create a mask in the original (unsorted) order
    # Scatter the removal mask back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    
    # 6. Apply the mask
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = filter_value
    
    return filtered_logits

