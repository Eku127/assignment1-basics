"""
Transformer Language Model implementation for CS336 Assignment 1.

This module implements the complete Transformer Language Model architecture,
combining token embeddings, multiple Transformer blocks, and output projection
to create a full language model.
"""

import torch
import torch.nn as nn
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
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply Transformer LM forward pass.
        
        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        # TODO: Implement the full Transformer LM forward pass
        #
        # Steps:
        # 1. Token embedding: token_ids -> embeddings
        # 2. Pass through all Transformer blocks
        # 3. Apply final normalization
        # 4. Apply output projection to get logits
        
        # Hint: Step 1 - Token embedding
        # Hint: Use self.token_embedding to convert token_ids to embeddings

        # (B, S) --> (B, S, d_model)
        embedding = self.token_embedding(token_ids)

        
        # Hint: Step 2 - Pass through Transformer blocks
        # Hint: For RoPE, create token_positions as torch.arange(seq_len)
        # Hint: Loop through self.transformer_blocks or use a for loop
        if self.use_rope:
            seq_len = token_ids.shape[-1]
            token_positions = torch.arange(seq_len, device=token_ids.device)
        else:
            token_positions = None
        for i in range(self.num_layers):
            embedding = self.transformer_blocks[i](embedding, token_positions)

        
        # Hint: Step 3 - Final normalization
        # Hint: Apply self.final_norm to the output of the last block

        normalized = self.final_norm(embedding)
        # Hint: Step 4 - Output projection
        # Hint: Apply self.lm_head to get logits over vocabulary

        logits = self.lm_head(normalized)

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
            
            # 自回归生成循环：逐个生成新token
            for step in range(max_new_tokens):
                # 检查序列长度是否超过上下文窗口
                current_len = generated_tokens.shape[1]
                if current_len >= self.context_length:
                    # 如果超过上下文长度，只保留最后context_length个token
                    input_tokens = generated_tokens[:, -self.context_length:]
                else:
                    input_tokens = generated_tokens
                
                # Step 1: 前向传播获取下一个token的logits
                # logits: (batch_size, seq_len, vocab_size)
                logits = self.forward(input_tokens)
                
                # Step 2: 只关注最后一个位置的logits（下一个token的预测）
                # next_token_logits: (batch_size, vocab_size)
                next_token_logits = logits[:, -1, :]
                
                # Step 3: 应用温度缩放控制生成的随机性
                # 温度越高，分布越平滑（更随机）；温度越低，分布越尖锐（更确定）
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Step 4: 应用top-p（nucleus）采样
                if top_p is not None and 0.0 < top_p < 1.0:
                    next_token_logits = self._apply_top_p_filtering(next_token_logits, top_p)
                
                # Step 5: 将logits转换为概率分布并采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                
                # Step 6: 将新生成的token追加到序列中
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                
                # Step 7: 检查是否生成了结束token
                if eos_token_id is not None:
                    # 检查是否所有批次都生成了EOS token
                    if torch.all(next_token.squeeze(-1) == eos_token_id):
                        break
            
            return generated_tokens
    
    def _apply_top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        应用top-p（nucleus）采样过滤。
        
        保留累积概率达到top_p的最高概率token，将其他token的logits设为负无穷。
        
        Args:
            logits: 形状为 (batch_size, vocab_size) 的logits张量
            top_p: Top-p阈值，范围 (0, 1)
            
        Returns:
            过滤后的logits张量，形状不变
        """
        # 将logits转换为概率分布
        probs = torch.softmax(logits, dim=-1)
        
        # 按概率降序排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过top_p的位置
        # 保留第一个超过阈值的token（确保至少有一个token可选）
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False  # 始终保留概率最高的token
        
        # 将需要移除的token的概率设置为0
        sorted_probs[sorted_indices_to_remove] = 0.0
        
        # 将排序后的概率映射回原始顺序
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(1, sorted_indices, sorted_probs)
        
        # 重新归一化概率（避免数值不稳定）
        filtered_probs = filtered_probs / (filtered_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 将概率转换回logits（用于后续采样）
        # 使用log避免数值下溢，对于概率为0的位置设置为负无穷
        filtered_logits = torch.log(filtered_probs + 1e-8)
        filtered_logits[filtered_probs == 0] = float('-inf')
        
        return filtered_logits
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return (f'vocab_size={self.vocab_size}, context_length={self.context_length}, '
                f'd_model={self.d_model}, num_layers={self.num_layers}, '
                f'num_heads={self.num_heads}, d_ff={self.d_ff}, use_rope={self.use_rope}')
