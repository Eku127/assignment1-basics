"""
Transformer Language Model Demo for CS336 Assignment 1.

This demo illustrates the complete Transformer Language Model architecture
and how all components work together.
"""

import torch
import torch.nn as nn
from cs336_basics.transformer import (
    TransformerLM, Embedding, TransformerBlock, RMSNorm, Linear
)


def demonstrate_transformer_lm():
    """Demonstrate the complete Transformer Language Model architecture."""
    print("=" * 60)
    print("CS336 Assignment 1 - Transformer Language Model Demo")
    print("=" * 60)
    
    # Configuration
    vocab_size = 100
    context_length = 8
    d_model = 16
    num_layers = 2
    num_heads = 2
    d_ff = 32
    batch_size = 2
    seq_len = 6
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Context length: {context_length}")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Feed-forward dimension (d_ff): {d_ff}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Create input token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"\nInput token IDs shape: {token_ids.shape}")
    print(f"Token IDs:\n{token_ids}")
    
    print("\n" + "=" * 60)
    print("Transformer Language Model Architecture")
    print("=" * 60)
    
    print("\nThe complete Transformer LM consists of:")
    print("1. Token Embedding: token_ids → embeddings")
    print("2. Multiple Transformer Blocks: embeddings → hidden_states")
    print("3. Final RMSNorm: hidden_states → normalized_states")
    print("4. LM Head (Linear): normalized_states → logits")
    print("5. Softmax (during loss computation): logits → probabilities")
    
    print("\n" + "-" * 40)
    print("Component 1: Token Embedding")
    print("-" * 40)
    
    embedding = Embedding(vocab_size, d_model)
    embeddings = embedding(token_ids)
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Each token is mapped to a {d_model}-dimensional vector")
    
    print("\n" + "-" * 40)
    print("Component 2: Transformer Blocks")
    print("-" * 40)
    
    print(f"Stack of {num_layers} identical Transformer blocks")
    print("Each block applies:")
    print("  - Multi-head self-attention with causal masking")
    print("  - Position-wise feed-forward network")
    print("  - Residual connections and layer normalization")
    
    # Note: This would be implemented in the actual forward pass
    hidden_states = embeddings  # Placeholder
    print(f"Hidden states shape: {hidden_states.shape}")
    
    print("\n" + "-" * 40)
    print("Component 3: Final Normalization")
    print("-" * 40)
    
    final_norm = RMSNorm(d_model)
    normalized = final_norm(hidden_states)
    print(f"Normalized states shape: {normalized.shape}")
    print("Final normalization ensures proper scaling for output projection")
    
    print("\n" + "-" * 40)
    print("Component 4: Language Model Head")
    print("-" * 40)
    
    lm_head = Linear(d_model, vocab_size)
    logits = lm_head(normalized)
    print(f"Logits shape: {logits.shape}")
    print(f"Each position gets a distribution over {vocab_size} vocabulary items")
    
    print("\n" + "-" * 40)
    print("Component 5: Next Token Prediction")
    print("-" * 40)
    
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Each position has a probability distribution summing to 1.0")
    print(f"Probability sums: {probs.sum(dim=-1)}")
    
    print("\n" + "=" * 60)
    print("Autoregressive Language Modeling")
    print("=" * 60)
    
    print("\nFor training:")
    print("- Input: [w1, w2, w3, w4, w5]")
    print("- Targets: [w2, w3, w4, w5, w6]")
    print("- Predict next token at each position")
    print("- Use causal masking to prevent future information leakage")
    
    print("\nFor generation:")
    print("- Start with prompt tokens")
    print("- Predict next token using model")
    print("- Append predicted token to sequence")
    print("- Repeat until end token or max length")
    
    print("\n" + "=" * 60)
    print("Framework Implementation Guide")
    print("=" * 60)
    
    print("\nYou need to implement:")
    print("\n1. TransformerLM.__init__:")
    print("   - self.token_embedding = Embedding(vocab_size, d_model)")
    print("   - self.transformer_blocks = nn.ModuleList([...])")
    print("   - self.final_norm = RMSNorm(d_model)")
    print("   - self.lm_head = Linear(d_model, vocab_size)")
    
    print("\n2. TransformerLM.forward:")
    print("   - Embed token IDs to get embeddings")
    print("   - Pass through all Transformer blocks")
    print("   - Apply final normalization")
    print("   - Project to vocabulary logits")
    
    print("\n3. Optional: TransformerLM.generate:")
    print("   - Implement autoregressive text generation")
    print("   - Support temperature and top-p sampling")
    
    print("\n" + "=" * 60)
    print("Key Design Decisions")
    print("=" * 60)
    
    print("\n1. Pre-norm Architecture:")
    print("   - More stable training than post-norm")
    print("   - Clean residual stream from input to output")
    print("   - Requires final normalization before output projection")
    
    print("\n2. Causal Masking:")
    print("   - Prevents attention to future tokens")
    print("   - Essential for autoregressive language modeling")
    print("   - Implemented in scaled dot-product attention")
    
    print("\n3. RoPE (Rotary Positional Embedding):")
    print("   - Provides relative positional information")
    print("   - Applied to query and key vectors in attention")
    print("   - No learnable parameters, just trigonometric functions")
    
    print("\n4. SwiGLU Feed-Forward:")
    print("   - Modern alternative to ReLU-based FFN")
    print("   - Combines SiLU activation with gating mechanism")
    print("   - Better performance on language modeling tasks")
    
    print("\n" + "=" * 60)
    print("Testing and Validation")
    print("=" * 60)
    
    print("\n1. Unit Tests:")
    print("   - uv run pytest -k test_transformer_block")
    print("   - uv run pytest -k test_transformer_lm")
    
    print("\n2. Shape Verification:")
    print("   - Check tensor shapes at each step")
    print("   - Ensure dimensions match expected values")
    
    print("\n3. Gradient Flow:")
    print("   - Verify gradients flow through all parameters")
    print("   - Check for vanishing/exploding gradients")
    
    print("\n4. Overfitting Test:")
    print("   - Train on single batch to verify implementation")
    print("   - Should achieve near-zero loss if correct")
    
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    
    print("\n1. Complete TransformerBlock implementation")
    print("2. Implement TransformerLM framework")
    print("3. Run tests to verify correctness")
    print("4. Move on to training loop and optimization")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demonstrate_transformer_lm()
