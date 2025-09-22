# Transformer Implementation for CS336 Assignment 1

This directory contains the implementation of a Transformer language model from scratch for CS336 Assignment 1.

## Implemented Modules

### ✅ Linear Module (`linear.py`)

A custom linear transformation module that performs `y = Wx` where:
- `W` is a learnable weight matrix of shape `(out_features, in_features)`
- No bias term (following modern LLM practices)
- Uses truncated normal initialization: `N(μ=0, σ²=2/(in_features + out_features))` truncated at `[-3σ, 3σ]`
- Supports arbitrary batch dimensions using `torch.einsum`

**Key Features:**
- Inherits from `torch.nn.Module`
- Weight matrix stored as `nn.Parameter` with shape `(out_features, in_features)`
- Uses `torch.einsum` for efficient batched computation
- Proper initialization following assignment specifications

**Usage:**
```python
from cs336_basics.transformer.linear import Linear

# Create a linear layer: 512 -> 1024
linear = Linear(in_features=512, out_features=1024)

# Forward pass with batched input
x = torch.randn(batch_size, seq_len, 512)  # Input
y = linear(x)  # Output shape: (batch_size, seq_len, 1024)
```

### 🔨 Embedding Module (`embedding.py`) - **TODO: 需要实现**

A custom embedding layer that maps token IDs to dense vectors:
- Maps discrete token IDs to continuous `d_model`-dimensional vectors
- Weight matrix shape: `(vocab_size, d_model)`
- Uses truncated normal initialization: `N(μ=0, σ²=1)` truncated at `[-3, 3]`
- Supports arbitrary batch dimensions

**Key Features:**
- Inherits from `torch.nn.Module`
- Embedding matrix stored as `nn.Parameter` with shape `(vocab_size, d_model)`
- Uses advanced indexing for efficient lookup
- Proper initialization following assignment specifications

**Usage:**
```python
from cs336_basics.transformer.embedding import Embedding

# Create an embedding layer: vocab_size=10000, d_model=512
embedding = Embedding(num_embeddings=10000, embedding_dim=512)

# Forward pass with token IDs
token_ids = torch.tensor([[1, 5, 3], [2, 4, 6]])  # Shape: (batch_size, seq_len)
embeddings = embedding(token_ids)  # Shape: (batch_size, seq_len, 512)
```

**Implementation Hint:**
```python
def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    # Use advanced indexing: self.weight[token_ids]
    return self.weight[token_ids]
```

### ✅ RMSNorm Module (`rmsnorm.py`)

Root Mean Square Layer Normalization module:
- Normalizes activations using root mean square instead of mean and variance
- Formula: `RMSNorm(ai) = (ai / RMS(a)) * gi` where `RMS(a) = sqrt(1/d_model * sum(ai^2) + eps)`
- Uses learnable gain parameter for scaling
- More stable than LayerNorm for deep networks

**Key Features:**
- Inherits from `torch.nn.Module`
- Gain parameter stored as `nn.Parameter` with shape `(d_model,)`
- Upcasts to float32 for numerical stability
- Proper initialization (gain = 1)
- Supports arbitrary batch dimensions

**Usage:**
```python
from cs336_basics.transformer.rmsnorm import RMSNorm

# Create RMSNorm layer
rmsnorm = RMSNorm(d_model=512, eps=1e-5)

# Forward pass
x = torch.randn(batch_size, seq_len, 512)  # Input
normalized = rmsnorm(x)  # Output shape: (batch_size, seq_len, 512)
```

**Implementation Details:**
- Uses `torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)` for RMS calculation
- Applies gain parameter via element-wise multiplication or einsum
- Proper dtype handling for numerical stability

### ✅ Embedding Module (`embedding.py`) - **已完成实现**

Embedding layer for token-to-vector mapping:
- Maps discrete token IDs to dense continuous vectors
- Uses learnable weight matrix of shape (vocab_size, embedding_dim)
- Supports arbitrary batch dimensions and sequence lengths

**Key Features:**
- Inherits from `torch.nn.Module`
- Weight initialization: N(μ=0, σ²=1) truncated at [-3, 3]
- Uses advanced indexing for efficient lookup
- Handles arbitrary input shapes with broadcasting

**Usage:**
```python
from cs336_basics.transformer.embedding import Embedding

# Create embedding layer
embedding = Embedding(num_embeddings=50000, embedding_dim=512)

# Forward pass
token_ids = torch.tensor([1, 5, 10, 23])  # Token IDs
embeddings = embedding(token_ids)  # Output shape: (4, 512)
```

**Implementation Details:**
- Uses `self.weight[token_ids]` for efficient embedding lookup
- Supports arbitrary batch dimensions: (..., seq_len) → (..., seq_len, embedding_dim)
- Proper weight initialization using truncated normal distribution
- No bias term (following modern LLM practices)

### ✅ SwiGLU Module (`positionwise_feedforward.py`) - **已完成实现**

SwiGLU Feed-Forward Network module:
- Combines SiLU (Swish) activation with Gated Linear Units (GLU)
- Formula: `FFN(x) = W2 (SiLU(W1 x) ⊙ W3 x)`
- Uses three weight matrices: W1, W2, W3
- More effective than traditional ReLU-based FFN

**Key Features:**
- Inherits from `torch.nn.Module`
- Three weight matrices: W1(d_ff, d_model), W2(d_model, d_ff), W3(d_ff, d_model)
- SiLU activation: `SiLU(x) = x * sigmoid(x)`
- Gating mechanism for better performance
- Proper weight initialization using truncated normal distribution

**Usage:**
```python
from cs336_basics.transformer.positionwise_feedforward import SwiGLU

# Create SwiGLU layer
swiglu = SwiGLU(d_model=512, d_ff=2048)

# Forward pass
x = torch.randn(batch_size, seq_len, 512)  # Input
output = swiglu(x)  # Output shape: (batch_size, seq_len, 512)
```

**Implementation Details:**
- Uses `torch.einsum` for efficient matrix operations
- Implements proper SiLU activation function
- Applies gating mechanism with element-wise multiplication
- Handles arbitrary batch dimensions with `...` notation

### ✅ RoPE Module (`rope.py`) - **已完成实现**

Rotary Positional Embedding (RoPE) module:
- Implements relative positional encoding using rotary transformations
- Formula: `θ_{i,k} = i × Θ^(-2k/d)` for position i and dimension pair k
- Applies 2D rotations to pairs of embedding elements
- No learnable parameters - uses pre-computed cos/sin buffers

**Key Features:**
- Inherits from `torch.nn.Module`
- Pre-computes cos and sin values for all positions and dimension pairs
- Uses broadcasting for efficient angle computation
- Handles arbitrary batch dimensions and sequence lengths
- Applies rotation formula: `x'_even = x_even * cos - x_odd * sin`

**Usage:**
```python
from cs336_basics.transformer.rope import RotaryPositionalEmbedding

# Create RoPE module
rope = RotaryPositionalEmbedding(theta=10000.0, d_k=64, max_seq_len=512)

# Forward pass
x = torch.randn(batch_size, seq_len, 64)  # Input
positions = torch.arange(seq_len)  # Position indices
output = rope(x, positions)  # Output shape: (batch_size, seq_len, 64)
```

**Implementation Details:**
- Pre-computes rotation angles using broadcasting
- Separates even/odd dimensions for pairwise rotation
- Applies 2D rotation matrix to each dimension pair
- Recombines dimensions in original order
- Uses `register_buffer` for efficient cos/sin storage

### ✅ Scaled Dot-Product Attention (`attention.py`) - **已完成实现**

Implements the core attention mechanism from "Attention Is All You Need":
- Formula: `Attention(Q, K, V) = softmax(QK^T / √d_k)V`
- Includes numerically stable softmax implementation
- Supports causal masking for autoregressive generation
- Handles arbitrary batch dimensions

**Key Features:**
- Custom softmax function with numerical stability (subtracts max for stability)
- Efficient `torch.einsum` operations for matrix multiplication
- Causal mask support to prevent attention to future tokens
- Proper scaling by `√d_k` for gradient stability

**Usage:**
```python
from cs336_basics.transformer.attention import scaled_dot_product_attention

# Input tensors
Q = torch.randn(batch_size, num_heads, seq_len, d_k)  # Queries
K = torch.randn(batch_size, num_heads, seq_len, d_k)  # Keys  
V = torch.randn(batch_size, num_heads, seq_len, d_v)  # Values

# Optional causal mask
mask = create_causal_mask(seq_len)  # Lower triangular mask

# Apply attention
output = scaled_dot_product_attention(Q, K, V, mask)
# Output shape: (batch_size, num_heads, seq_len, d_v)
```

**Implementation Details:**
- Uses `torch.einsum('...qd, ...kd->...qk', Q, K)` for attention scores
- Applies numerical stability by subtracting max values before exp
- Supports both 3D and 4D input tensors for flexibility
- Efficiently handles causal masking with `masked_fill`

### ✅ Multi-Head Self-Attention (`multihead_attention.py`) - **已完成实现**

Complete multi-head self-attention implementation with optional RoPE support:
- Splits input into multiple attention heads for parallel processing
- Applies linear projections for Q, K, V transformations
- Combines multiple attention heads back to original dimension
- Supports both standard and RoPE-enhanced attention

**Key Features:**
- Uses custom `Linear` layers for all projections (Q, K, V, output)
- Efficient head splitting using `einops.rearrange`
- Causal masking for autoregressive language modeling
- Optional Rotary Positional Embedding integration
- Proper weight initialization following assignment specs

**Usage:**
```python
from cs336_basics.transformer.multihead_attention import MultiHeadSelfAttention

# Standard multi-head attention
mha = MultiHeadSelfAttention(
    d_model=512, 
    num_heads=8,
    use_rope=False
)

# With RoPE support
mha_rope = MultiHeadSelfAttention(
    d_model=512, 
    num_heads=8, 
    use_rope=True,
    max_seq_len=2048,
    theta=10000.0
)

# Forward pass
x = torch.randn(batch_size, seq_len, d_model)
output = mha(x)  # Shape: (batch_size, seq_len, d_model)

# With RoPE and position indices
token_positions = torch.arange(seq_len)
output_rope = mha_rope(x, token_positions)
```

**Implementation Details:**
- QKV projections: `d_model → d_model` using `Linear` layers
- Head reshaping: `(batch, seq, d_model) → (batch, heads, seq, d_k)`
- Attention computation using `scaled_dot_product_attention`
- Output projection: `d_model → d_model` for final transformation
- Automatic causal mask creation for each forward pass

## TODO: Modules to Implement

- [x] Embedding Module (`embedding.py`) - **✅ 已完成实现**
- [x] RMSNorm (`rmsnorm.py`) - **✅ 已完成实现**
- [x] SwiGLU (`positionwise_feedforward.py`) - **✅ 已完成实现**
- [x] Rotary Positional Embedding (`rope.py`) - **✅ 已完成实现**
- [x] Scaled Dot-Product Attention (`attention.py`) - **✅ 已完成实现**
- [x] Multi-Head Self-Attention (`multihead_attention.py`) - **✅ 已完成实现**
- [ ] Transformer Block (`transformer_block.py`)
- [ ] Full Transformer LM (`transformer_lm.py`)

## Demo Scripts

All demo scripts are located in the `demo/` directory:

- `demo/linear_demo.py` - Linear module demonstration
- `demo/embedding_demo.py` - Embedding module demonstration  
- `demo/rmsnorm_demo.py` - RMSNorm module demonstration
- `demo/swiglu_demo.py` - SwiGLU module demonstration
- `demo/rope_demo.py` - RoPE module demonstration
- `demo/softmax_demo.py` - Softmax function demonstration
- `demo/parameter_demo.py` - Parameter and initialization demonstration

### Run Demo Scripts

```bash
# Run individual demos
uv run python cs336_basics/transformer/demo/linear_demo.py
uv run python cs336_basics/transformer/demo/embedding_demo.py
uv run python cs336_basics/transformer/demo/rmsnorm_demo.py
uv run python cs336_basics/transformer/demo/swiglu_demo.py
uv run python cs336_basics/transformer/demo/rope_demo.py
uv run python cs336_basics/transformer/demo/softmax_demo.py
```

## Testing

### Run Individual Module Tests

Run the tests for the Linear module:
```bash
uv run pytest tests/test_model.py::test_linear -v
```

Run the tests for the Embedding module (when implemented):
```bash
uv run pytest tests/test_model.py::test_embedding -v
```

Run the tests for the RMSNorm module (when implemented):
```bash
uv run pytest tests/test_model.py::test_rmsnorm -v
```

Run the tests for the SwiGLU module (when implemented):
```bash
uv run pytest tests/test_model.py::test_swiglu -v
```

Run the tests for the RoPE module (when implemented):
```bash
uv run pytest tests/test_model.py::test_rope -v
```

Run the tests for the Scaled Dot-Product Attention:
```bash
uv run pytest tests/test_model.py::test_scaled_dot_product_attention -v
uv run pytest tests/test_model.py::test_4d_scaled_dot_product_attention -v
```

Run the tests for the Multi-Head Self-Attention:
```bash
uv run pytest tests/test_model.py::test_multihead_self_attention -v
uv run pytest tests/test_model.py::test_multihead_self_attention_with_rope -v
```

Run all attention-related tests:
```bash
uv run pytest tests/test_model.py -k "attention" -v
```

Run the tests for the Transformer Block (when implemented):
```bash
uv run pytest tests/test_model.py::test_transformer_block -v
```

Run the tests for the Transformer LM (when implemented):
```bash
uv run pytest tests/test_model.py::test_transformer_lm -v
```

### Run All Transformer Tests
```bash
uv run pytest tests/test_model.py -v
```

### Run Demo Scripts
```bash
uv run python cs336_basics/transformer/test_linear_demo.py
```

## Assignment Progress

- [x] Linear Module (1 point)
- [ ] Embedding Module (1 point) - **框架已创建，需要实现forward方法**
- [x] RMSNorm (1 point) - **✅ 已完成实现**
- [ ] SwiGLU Feed-Forward (2 points)
- [ ] RoPE (2 points)
- [x] Scaled Dot-Product Attention (5 points) - **✅ 已完成**
- [x] Multi-Head Self-Attention (5 points) - **✅ 已完成**
- [ ] Transformer Block (3 points)
- [ ] Transformer LM (3 points)

**Total: 23 points for Transformer implementation**
