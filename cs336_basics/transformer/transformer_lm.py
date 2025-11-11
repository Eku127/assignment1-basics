"""
Transformer Language Model implementation for CS336 Assignment 1.

This module implements the complete Transformer Language Model architecture,
combining token embeddings, multiple Transformer blocks, and output projection
to create a full language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear


class TransformerLM(nn.Module):
    """
    Complete Transformer Language Model.
    
    Implements the full Transformer LM architecture:
    1. Token Embedding
    2. Multiple Transformer Blocks
    3. Final RMSNorm (for pre-norm architecture)
    4. Output Linear Projection (LM Head)
    5. Softmax (applied during loss computation, not in forward)
    
    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum context length (sequence length)
        d_model: Dimensionality of the model
        num_layers: Number of Transformer blocks
        num_heads: Number of attention heads per block
        d_ff: Dimensionality of feed-forward inner layer
        use_rope: Whether to use Rotary Positional Embedding
        theta: RoPE theta parameter (if used)
        eps: Epsilon for RMSNorm numerical stability
        device: Device to store parameters on
        dtype: Data type of parameters
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = True,
        theta: float = 10000.0,
        eps: float = 1e-5,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rope = use_rope
        self.theta = theta
        self.eps = eps
        
        # Initialize the components of the Transformer LM
        # You need 4 main components:
        # 1. Token embedding layer
        # 2. Multiple Transformer blocks
        # 3. Final layer normalization (for pre-norm architecture)
        # 4. Output projection layer (LM head)
        
        # Initialize token embedding
        # Hint: Use Embedding class with vocab_size and d_model
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # Initialize Transformer blocks
        # Hint: Use nn.ModuleList to store multiple TransformerBlock instances
        # Hint: Each block should have same parameters: d_model, num_heads, d_ff, etc.
        # Hint: Pass context_length as max_seq_len for RoPE
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                use_rope=use_rope,
                max_seq_len=context_length,
                theta=theta,
                eps=eps,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        
        # Initialize final layer normalization (for pre-norm architecture)
        # Hint: Use RMSNorm with d_model dimension
        self.final_norm = RMSNorm(d_model, eps, device=device, dtype=dtype)
        
        # Initialize output projection (LM head)
        # Hint: Use Linear layer from d_model to vocab_size
        # Hint: This projects hidden states to vocabulary logits
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(
        self, 
        token_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        use_cache: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None] | None]:
        """
        Apply Transformer LM forward pass with optional KV cache support.
        
        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)
            past_key_values: Optional list of cache tuples, one per layer
                Each tuple contains (past_K, past_V) for that layer
            use_cache: Whether to return updated cache (default: False for training compatibility)
            
        Returns:
            If use_cache=False: Logits tensor of shape (batch_size, seq_len, vocab_size)
            If use_cache=True: Tuple of (logits, new_past_key_values)
            
        Note:
            Token positions for RoPE are automatically inferred from KV cache if available,
            otherwise start from 0. This ensures correct absolute position encoding during
            incremental generation with KV cache.
        """
        # Step 1 - Token embedding
        # (B, S) --> (B, S, d_model)
        embedding = self.token_embedding(token_ids)

        # Step 2 - Pass through Transformer blocks
        # For RoPE, automatically infer token_positions from KV cache or start from 0
        if self.use_rope:
            seq_len = token_ids.shape[-1]
            # Try to infer position from KV cache if available
            if past_key_values is not None:
                # Find first non-None cache to get past_seq_len
                # all layer is the same so we only find one is enough
                for layer_cache in past_key_values:
                    if layer_cache is not None:
                        past_K, _ = layer_cache
                        # past_K shape: (..., num_heads, past_seq_len, d_k)
                        past_seq_len = past_K.shape[-2]
                        # Current token positions start from past_seq_len
                        token_positions = torch.arange(
                            past_seq_len, past_seq_len + seq_len, 
                            device=token_ids.device
                        )
                        break
                else:
                    # No cache found, start from 0 (first generation step)
                    token_positions = torch.arange(seq_len, device=token_ids.device)
            else:
                # No cache provided, start from 0 (training or first generation)
                token_positions = torch.arange(seq_len, device=token_ids.device)
        else:
            token_positions = None
        
        # Initialize cache list if needed
        if past_key_values is None:
            past_key_values = [None] * self.num_layers
        
        # Pass through each block with cache
        new_past_key_values = []
        for i in range(self.num_layers):
            layer_cache = past_key_values[i] if past_key_values else None
            embedding, new_cache = self.transformer_blocks[i](
                embedding, 
                token_positions=token_positions,
                past_key_values=layer_cache,
                use_cache=use_cache
            )
            if use_cache:
                new_past_key_values.append(new_cache)

        # Step 3 - Final normalization
        normalized = self.final_norm(embedding)

        # Step 4 - Output projection
        logits = self.lm_head(normalized)

        # Return based on use_cache flag
        if use_cache:
            return logits, new_past_key_values
        else:
            return logits
    
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float | None = None,
        eos_token_id: int | None = None
    ) -> torch.Tensor:
        """
        Generate text using the language model.
        
        Args:
            prompt_tokens: Initial tokens of shape (batch_size, prompt_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling (default: 1.0)
            top_p: Top-p (nucleus) sampling threshold (default: None)
            eos_token_id: End-of-sequence token ID (default: None)
            
        Returns:
            Generated tokens of shape (batch_size, prompt_len + num_generated)
        """
        # 实现自回归文本生成
        # 设置模型为评估模式，禁用dropout等
        self.eval()
        
        with torch.no_grad():  # 生成时不需要梯度计算
            # 获取设备信息，确保所有张量在同一设备上
            device = prompt_tokens.device
            batch_size, prompt_len = prompt_tokens.shape
            
            # 初始化生成序列，从提示词开始
            # generated_tokens: (batch_size, prompt_len + max_new_tokens)
            generated_tokens = prompt_tokens.clone()
            
            # 初始化 KV cache
            past_key_values = None
            
            # 自回归生成循环：逐个生成新token
            for step in range(max_new_tokens):
                # 准备输入：第一步处理整个 prompt，后续步骤只处理新 token
                if step == 0:
                    # 第一步：处理整个 prompt
                    # 检查序列长度是否超过上下文窗口
                    if generated_tokens.shape[1] > self.context_length:
                        # 如果超过上下文长度，只保留最后context_length个token
                        input_tokens = generated_tokens[:, -self.context_length:]
                    else:
                        input_tokens = generated_tokens
                else:
                    # 后续步骤：只处理新生成的 token
                    input_tokens = generated_tokens[:, -1:]  # (batch_size, 1)
                
                # Step 1: 前向传播获取下一个token的logits（使用 KV cache）
                # logits: (batch_size, seq_len, vocab_size)
                logits, past_key_values = self.forward(
                    input_tokens,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # Step 2: 只关注最后一个位置的logits（下一个token的预测）
                # next_token_logits: (batch_size, vocab_size)
                next_token_logits = logits[:, -1, :]
                
                # Step 3: 应用温度缩放控制生成的随机性（与 generate_text 保持一致）
                # 温度越高，分布越平滑（更随机）；温度越低，分布越尖锐（更确定）
                # Handle edge case: very small temperature (greedy decoding)
                if temperature < 1e-8:
                    temperature = 1e-8
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Step 4: 应用top-p（nucleus）采样（与 generate_text 保持一致）
                if top_p is not None and 0.0 < top_p < 1.0:
                    next_token_logits = self._apply_top_p_filtering(next_token_logits, top_p)
                
                # Step 5: 将logits转换为概率分布并采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                
                # Step 6: 检查是否生成了结束token（与 generate_text 保持一致）
                if eos_token_id is not None:
                    # 对于单样本情况，使用 item() 检查（与 generate_text 一致）
                    if next_token.item() == eos_token_id:
                        break
                
                # Step 7: 将新生成的token追加到序列中
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            
            return generated_tokens
    
    def _apply_top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        应用top-p（nucleus）采样过滤（与 decode.py 中的 top_p_filtering 保持一致）。
        
        保留累积概率达到top_p的最高概率token，将其他token的logits设为负无穷。
        
        Args:
            logits: 形状为 (batch_size, vocab_size) 的logits张量
            top_p: Top-p阈值，范围 (0, 1)
            
        Returns:
            过滤后的logits张量，形状不变
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
        
        # 6. Apply the mask (set filtered logits to -inf)
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        return filtered_logits
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return (f'vocab_size={self.vocab_size}, context_length={self.context_length}, '
                f'd_model={self.d_model}, num_layers={self.num_layers}, '
                f'num_heads={self.num_heads}, d_ff={self.d_ff}, use_rope={self.use_rope}')
