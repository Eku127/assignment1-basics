# Training Infrastructure for CS336 Assignment 1

This directory contains the complete training infrastructure for Transformer language models in CS336 Assignment 1.

## 📋 **Implementation Roadmap**

This module covers **Sections 4, 5, and 6** of the assignment:
- **Section 4**: Training a Transformer LM (Loss, Optimizers, Scheduling)
- **Section 5**: Training Loop (Data Loading, Checkpointing, Training Infrastructure)
- **Section 6**: Generating Text (Decoding, Temperature Scaling, Top-p Sampling)

### 🎯 **Current Status**

| Component | Module | Status | Points | Tests | Command |
|-----------|--------|--------|--------|-------|---------|
| Cross-entropy Loss | `loss.py` | ✅ DONE | 1 | `test_cross_entropy` | `uv run pytest tests/test_nn_utils.py::test_cross_entropy -v` |
| SGD Optimizer | `optimizer.py` | ✅ DONE | - | - | See `demo/learning_rate_tuning.py` |
| AdamW Optimizer | `optimizer.py` | ✅ DONE | 2 | `test_adamw` | `uv run pytest tests/test_optimizer.py::test_adamw -v` |
| LR Scheduling | `lr_scheduler.py` | ✅ DONE | 1 | `test_get_lr_cosine_schedule` | `uv run pytest tests/test_optimizer.py::test_get_lr_cosine_schedule -v` |
| Gradient Clipping | `gradient_clipping.py` | ✅ DONE | 1 | `test_gradient_clipping` | `uv run pytest tests/test_nn_utils.py::test_gradient_clipping -v` |
| Data Loading | `data_loader.py` | ⬜ TODO | 2 | `test_get_batch` | `uv run pytest tests/test_data.py::test_get_batch -v` |
| Checkpointing | `checkpoint.py` | ⬜ TODO | 1 | `test_checkpointing` | `uv run pytest tests/test_checkpoint.py::test_checkpointing -v` |
| Training Loop | `train.py` | ⬜ TODO | 4 | - | Manual testing |
| Text Generation | `decode.py` | ⬜ TODO | 3 | - | Manual testing |
| **Total** | | **5/9** | **15** | | |

---

## 📚 Module Overview

### 🔴 Cross-Entropy Loss (`loss.py`)

Implements the numerically stable cross-entropy loss for language modeling:

**Mathematical Definition:**
```
ℓ(θ; D) = (1 / |D|m) * ΣΣ -log p_θ(x_{i+1} | x_{1:i})

where p(x_{i+1} | x_{1:i}) = softmax(o_i)[x_{i+1}] = exp(o_i[x_{i+1}]) / Σ exp(o_i[a])
```

**Key Features:**
- Numerical stability through log-sum-exp trick
- Cancels log and exp where possible
- Handles arbitrary batch dimensions
- Returns average loss across batch

**Implementation Requirements:**
- Subtract largest element for numerical stability
- Use log-sum-exp for stable computation
- Support batched inputs: `(batch_size, seq_len, vocab_size)`
- Return scalar loss (averaged across batch and sequence)

**Usage:**
```python
from cs336_basics.training import cross_entropy

# Logits from model: (batch_size, seq_len, vocab_size)
logits = model(input_ids)
# Targets: (batch_size, seq_len)
targets = target_ids

# Compute loss
loss = cross_entropy(logits, targets)
loss.backward()
```

**Perplexity:**
For evaluation, we also compute perplexity:
```
perplexity = exp((1/m) * Σ ℓ_i)
```

**Test:**
```bash
# Run the test
uv run pytest tests/test_nn_utils.py::test_cross_entropy -v

# Expected behavior:
# - Should match PyTorch's F.cross_entropy output
# - Should handle large logit values (1000+) without overflow
# - Numerical stability: atol=1e-4
```

**Test Code:**
```python
# In tests/test_nn_utils.py
def test_cross_entropy():
    inputs = torch.tensor([
        [[0.1, 0.5, 0.3, 0.8, 0.2],
         [0.4, 0.7, 0.3, 0.4, 0.3]],
        [[0.3, 0.9, 0.7, 0.8, 0.8],
         [0.6, 0.4, 0.1, 0.6, 0.4]],
    ])  # (2, 2, 5)
    targets = torch.tensor([[1, 0], [4, 1]])  # (2, 2)
    
    # Your implementation
    loss = run_cross_entropy(inputs.view(-1, 5), targets.view(-1))
    
    # PyTorch reference
    expected = F.cross_entropy(inputs.view(-1, 5), targets.view(-1))
    
    # Should match within tolerance
    assert torch.allclose(loss, expected, atol=1e-4)
```

---

### 🔵 SGD Optimizer (`optimizer.py`)

Stochastic Gradient Descent with learning rate decay.

**Algorithm:**
```
θ_{t+1} ← θ_t - α_t ∇L(θ_t; B_t)

With learning rate decay:
θ_{t+1} = θ_t - (α / √(t+1)) * ∇L(θ_t; B_t)
```

**Key Features:**
- Subclasses `torch.optim.Optimizer`
- Supports learning rate decay
- Per-parameter state tracking
- Parameter groups support

**Implementation:**
- Must implement `__init__(self, params, lr=1e-3)`
- Must implement `step(self, closure=None)`
- Uses `self.state` for iteration tracking
- Updates via `p.data -= lr * p.grad`

**Example:**
```python
from cs336_basics.training import SGD

# Create optimizer
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
optimizer = SGD([weights], lr=1.0)

# Training loop
for t in range(100):
    optimizer.zero_grad()
    loss = (weights**2).mean()
    loss.backward()
    optimizer.step()
    print(f"Step {t}: Loss = {loss.item():.4f}")
```

---

### 🟢 AdamW Optimizer (`optimizer.py`)

Implements AdamW [Loshchilov & Hutter, 2019] with decoupled weight decay.

**Algorithm:**
```
m ← β_1 * m + (1 - β_1) * g          # First moment estimate
v ← β_2 * v + (1 - β_2) * g²         # Second moment estimate
α_t ← α * √(1 - β_2^t) / (1 - β_1^t) # Bias correction
θ ← θ - α_t * m / (√v + ε)           # Parameter update
θ ← θ - α * λ * θ                    # Weight decay
```

**Key Features:**
- Stateful optimizer (tracks first and second moments)
- Bias correction for moments
- Decoupled weight decay
- Commonly used hyperparameters:
  - `β_1 = 0.9, β_2 = 0.999` (default)
  - `β_1 = 0.9, β_2 = 0.95` (LLaMA, GPT-3)
  - `ε = 1e-8` (stability)
  - `λ` (weight decay rate)

**Implementation Requirements:**
- Subclass `torch.optim.Optimizer`
- Accept `lr`, `betas=(β_1, β_2)`, `eps`, `weight_decay=λ`
- Maintain state: `m`, `v`, `t` (iteration number)
- Note: iteration `t` starts at 1, not 0

**Memory Usage:**
AdamW requires **3× parameter memory** (parameters + m + v)

**Test:**
```bash
# Run the test
uv run pytest tests/test_optimizer.py::test_adamw -v

# Expected behavior:
# - Should match PyTorch's torch.optim.AdamW
# - Properly tracks moments (m, v) and iteration (t)
# - Applies weight decay separately from gradient update
# - Bias correction should work correctly
```

**Test Code:**
```python
# In tests/test_optimizer.py
def test_adamw():
    # Create test parameters
    params_yours = [torch.nn.Parameter(torch.randn(10, 10)) for _ in range(3)]
    params_pytorch = [torch.nn.Parameter(p.clone()) for p in params_yours]
    
    # Your AdamW
    opt_yours = get_adamw_cls()(params_yours, lr=0.001, betas=(0.9, 0.999), 
                                 eps=1e-8, weight_decay=0.01)
    
    # PyTorch AdamW
    opt_pytorch = torch.optim.AdamW(params_pytorch, lr=0.001, betas=(0.9, 0.999),
                                     eps=1e-8, weight_decay=0.01)
    
    # Run several steps
    for _ in range(10):
        # Compute same loss and gradients
        loss_yours = sum((p**2).sum() for p in params_yours)
        loss_pytorch = sum((p**2).sum() for p in params_pytorch)
        
        opt_yours.zero_grad()
        opt_pytorch.zero_grad()
        
        loss_yours.backward()
        loss_pytorch.backward()
        
        opt_yours.step()
        opt_pytorch.step()
    
    # Parameters should match
    for p_yours, p_pytorch in zip(params_yours, params_pytorch):
        assert torch.allclose(p_yours, p_pytorch, atol=1e-6)
```

---

### 🟡 Learning Rate Scheduling (`lr_scheduler.py`)

Implements cosine annealing schedule with linear warmup (used in LLaMA).

**Schedule Definition:**

```
         ⎧ (t / T_w) * α_max                              if t < T_w      (warmup)
α_t =    ⎨ α_min + 0.5 * (1 + cos((t-T_w)/(T_c-T_w)*π)) * (α_max - α_min)  if T_w ≤ t ≤ T_c  (cosine)
         ⎩ α_min                                          if t > T_c      (post-anneal)
```

**Parameters:**
- `t`: current iteration
- `α_max`: maximum learning rate (peak after warmup)
- `α_min`: minimum learning rate (floor)
- `T_w`: number of warmup iterations
- `T_c`: total cosine annealing iterations

**Key Features:**
- Linear warmup from 0 to `α_max`
- Smooth cosine decay from `α_max` to `α_min`
- Constant `α_min` after annealing complete
- Prevents sudden learning rate changes

**Usage:**
```python
from cs336_basics.training import get_lr_cosine_schedule

# Typical schedule: 5000 warmup steps, 100k total steps
for t in range(max_steps):
    lr = get_lr_cosine_schedule(
        t=t,
        max_lr=3e-4,
        min_lr=3e-5,
        warmup_iters=5000,
        cosine_cycle_iters=100000
    )
    # Update optimizer learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

**Test:**
```bash
# Run the test
uv run pytest tests/test_optimizer.py::test_get_lr_cosine_schedule -v

# Expected behavior:
# - Warmup: linear increase from 0 to max_lr
# - Cosine: smooth decay from max_lr to min_lr
# - Post-anneal: constant min_lr
```

**Test Code:**
```python
# In tests/test_optimizer.py
def test_get_lr_cosine_schedule():
    max_lr = 1.0
    min_lr = 0.1
    warmup_iters = 100
    cosine_cycle_iters = 1000
    
    # Test warmup phase (t < warmup_iters)
    lr_start = run_get_lr_cosine_schedule(0, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    assert lr_start == 0.0  # Should start at 0
    
    lr_mid_warmup = run_get_lr_cosine_schedule(50, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    assert 0 < lr_mid_warmup < max_lr  # Should be increasing
    
    lr_end_warmup = run_get_lr_cosine_schedule(100, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    assert abs(lr_end_warmup - max_lr) < 1e-6  # Should reach max_lr
    
    # Test cosine phase
    lr_mid_cosine = run_get_lr_cosine_schedule(500, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    assert min_lr < lr_mid_cosine < max_lr  # Should be between min and max
    
    # Test post-anneal phase (t > cosine_cycle_iters)
    lr_post = run_get_lr_cosine_schedule(1500, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    assert abs(lr_post - min_lr) < 1e-6  # Should be at min_lr
```

---

### 🟣 Gradient Clipping (`gradient_clipping.py`)

Prevents exploding gradients by clipping the global gradient norm.

**Algorithm:**
```
∥g∥_2 = √(Σ ∥g_p∥²)  for all parameters p

If ∥g∥_2 > M:
    g ← g * M / (∥g∥_2 + ε)
```

**Parameters:**
- `M`: maximum allowed gradient norm
- `ε = 1e-6`: numerical stability (PyTorch default)

**Key Features:**
- Clips based on global norm (all parameters combined)
- In-place modification of gradients
- Applied after `loss.backward()` but before `optimizer.step()`

**Usage:**
```python
from cs336_basics.training import clip_gradients

# Training loop
loss.backward()
grad_norm = clip_gradients(model.parameters(), max_norm=1.0)
if grad_norm > 1.0:
    print(f"Clipped gradients: {grad_norm:.2f} -> 1.0")
optimizer.step()
```

**Test:**
```bash
# Run the test
uv run pytest tests/test_nn_utils.py::test_gradient_clipping -v

# Expected behavior:
# - Should match PyTorch's torch.nn.utils.clip_grad_norm_
# - Computes global L2 norm of all gradients
# - Scales all gradients if norm > max_norm
# - Returns original norm before clipping
```

**Test Code:**
```python
# In tests/test_nn_utils.py
def test_gradient_clipping():
    # Create parameters with large gradients
    params = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(3)]
    max_norm = 0.01  # Very small threshold to ensure clipping
    
    # Compute loss and gradients
    loss = sum((p**2).sum() for p in params)
    loss.backward()
    
    # Compute expected norm
    total_norm = torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None))
    
    # Apply gradient clipping
    run_gradient_clipping(params, max_norm)
    
    # Check that norm is now <= max_norm
    new_norm = torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None))
    assert new_norm <= max_norm + 1e-6
    
    # Should match PyTorch's implementation
    params_pytorch = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(3)]
    loss_pytorch = sum((p**2).sum() for p in params_pytorch)
    loss_pytorch.backward()
    
    # Clone gradients before clipping
    grads_yours = [p.grad.clone() for p in params]
    grads_pytorch = [p.grad.clone() for p in params_pytorch]
    
    run_gradient_clipping(params, max_norm)
    torch.nn.utils.clip_grad_norm_(params_pytorch, max_norm)
    
    # Results should be close
    for g1, g2 in zip(grads_yours, grads_pytorch):
        assert torch.allclose(g1, g2, atol=1e-6)
```

---

### 🟠 Data Loading (`data_loader.py`)

Efficient batch sampling from tokenized sequences.

**Data Format:**
- Input: Single sequence of token IDs `x = (x_1, ..., x_n)`
- Output: Batched sequences for training

**Batch Structure:**
- Batch size: `B` sequences
- Context length: `m` tokens per sequence
- Input sequences: `(batch_size, context_length)` - tokens `x_i`
- Target sequences: `(batch_size, context_length)` - next tokens `x_{i+1}`

**Example:**
```
Given x = [1, 2, 3, 4, 5, 6, 7, 8, ...]
With B=2, m=3:
  Input:  [[2, 3, 4], [5, 6, 7]]
  Target: [[3, 4, 5], [6, 7, 8]]
```

**Key Features:**
- Random sampling from dataset
- No padding needed (uniform length)
- Memory-efficient with `np.memmap` for large datasets
- Device-aware (CPU/CUDA/MPS support)

**Implementation:**
```python
from cs336_basics.training import get_batch

# Load tokenized data
data = np.load('tokens.npy', mmap_mode='r')

# Sample a batch
inputs, targets = get_batch(
    data=data,
    batch_size=32,
    context_length=512,
    device='cuda:0'
)
```

**Memory-Mapped Loading:**
For large datasets that don't fit in memory:
```python
# Use memory mapping
data = np.load('tokens.npy', mmap_mode='r')
# or
data = np.memmap('tokens.npy', dtype=np.uint16, mode='r')
```

**Test:**
```bash
# Run the test
uv run pytest tests/test_data.py::test_get_batch -v

# Expected behavior:
# - Returns (inputs, targets) both shape (batch_size, context_length)
# - targets[i][j] = inputs[i][j+1] (next token prediction)
# - Random sampling from dataset
# - Tensors placed on correct device
# - Dtype should be torch.long (integer type)
```

**Test Code:**
```python
# In tests/test_data.py
def test_get_batch():
    # Create synthetic dataset
    dataset = np.arange(1000, dtype=np.int64)  # [0, 1, 2, ..., 999]
    batch_size = 4
    context_length = 10
    device = 'cpu'
    
    # Sample a batch
    inputs, targets = run_get_batch(dataset, batch_size, context_length, device)
    
    # Check shapes
    assert inputs.shape == (batch_size, context_length)
    assert targets.shape == (batch_size, context_length)
    
    # Check device
    assert inputs.device.type == device
    assert targets.device.type == device
    
    # Check dtype
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    
    # Check that targets are next tokens
    # For at least one batch element, verify the relationship
    # (Can't verify all since sampling is random)
    assert all(inputs[i].max() < len(dataset) for i in range(batch_size))
    assert all(targets[i].max() < len(dataset) for i in range(batch_size))
    
    # Sample multiple batches to test randomness
    inputs1, _ = run_get_batch(dataset, batch_size, context_length, device)
    inputs2, _ = run_get_batch(dataset, batch_size, context_length, device)
    # Should be different (with high probability)
    assert not torch.all(inputs1 == inputs2)
```

---

### 🔴 Checkpointing (`checkpoint.py`)

Save and restore training state for resumable training.

**Checkpoint Contents:**
1. Model weights (`model.state_dict()`)
2. Optimizer state (`optimizer.state_dict()`) - includes moment estimates for AdamW
3. Iteration number (for LR scheduling)

**Implementation:**

**Save Checkpoint:**
```python
def save_checkpoint(model, optimizer, iteration, out):
    """
    Args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        iteration: int (current training step)
        out: str | Path | BinaryIO (destination)
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)
```

**Load Checkpoint:**
```python
def load_checkpoint(src, model, optimizer):
    """
    Args:
        src: str | Path | BinaryIO (source)
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
    
    Returns:
        iteration: int (saved training step)
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']
```

**Usage:**
```python
from cs336_basics.training import save_checkpoint, load_checkpoint

# During training: save every N steps
if step % save_every == 0:
    save_checkpoint(model, optimizer, step, f'checkpoint_{step}.pt')

# Resume training
iteration = load_checkpoint('checkpoint_10000.pt', model, optimizer)
print(f"Resuming from iteration {iteration}")
```

**Test:**
```bash
# Run the test
uv run pytest tests/test_checkpoint.py::test_checkpointing -v

# Expected behavior:
# - Save: creates checkpoint file with model, optimizer, iteration
# - Load: restores exact model and optimizer state
# - Returns correct iteration number
# - Works with both file paths and file-like objects
```

**Test Code:**
```python
# In tests/test_checkpoint.py
def test_checkpointing():
    import tempfile
    import os
    
    # Create a simple model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Do a few training steps to create state
    for _ in range(5):
        loss = (model.weight ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Save checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        checkpoint_path = f.name
        run_save_checkpoint(model, optimizer, iteration=42, out=checkpoint_path)
    
    # Create new model and optimizer (different state)
    model_new = torch.nn.Linear(10, 10)
    optimizer_new = torch.optim.SGD(model_new.parameters(), lr=0.01)
    
    # Load checkpoint
    loaded_iteration = run_load_checkpoint(checkpoint_path, model_new, optimizer_new)
    
    # Check iteration
    assert loaded_iteration == 42
    
    # Check model weights match
    for p1, p2 in zip(model.parameters(), model_new.parameters()):
        assert torch.allclose(p1, p2)
    
    # Check optimizer state matches
    assert optimizer.state_dict()['state'].keys() == optimizer_new.state_dict()['state'].keys()
    
    # Clean up
    os.unlink(checkpoint_path)
```

---

### 🟢 Training Loop (`train.py`)

Main training script that integrates all components.

**Required Features:**
1. **Hyperparameter Configuration**
   - Model parameters (d_model, num_layers, num_heads, etc.)
   - Training parameters (batch_size, learning_rate, etc.)
   - Optimizer settings (betas, weight_decay, etc.)
   - Command-line argument support

2. **Data Loading**
   - Memory-mapped dataset loading (`np.memmap`)
   - Separate training and validation sets
   - Efficient batch sampling

3. **Training Loop**
   - Forward pass → loss computation
   - Backward pass → gradient computation
   - Gradient clipping
   - Optimizer step with LR scheduling
   - Periodic validation

4. **Checkpointing**
   - Save checkpoints periodically
   - Resume from checkpoint
   - Keep best/latest checkpoints

5. **Logging**
   - Training/validation loss
   - Learning rate tracking
   - Gradient norms
   - Perplexity metrics
   - Wallclock time
   - Integration with Weights & Biases (optional)

**Typical Training Loop Structure:**
```python
# Initialize
model = TransformerLM(...)
optimizer = AdamW(model.parameters(), lr=3e-4)
train_data = np.load('train.npy', mmap_mode='r')
val_data = np.load('val.npy', mmap_mode='r')

# Training loop
for step in range(max_steps):
    # Get learning rate for this step
    lr = get_lr_cosine_schedule(step, ...)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Sample batch
    inputs, targets = get_batch(train_data, batch_size, context_length, device)
    
    # Forward pass
    optimizer.zero_grad()
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    
    # Backward pass
    loss.backward()
    clip_gradients(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Logging
    if step % log_every == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}, LR: {lr:.2e}")
    
    # Validation
    if step % val_every == 0:
        val_loss = evaluate(model, val_data, ...)
        print(f"Validation Loss: {val_loss:.4f}")
    
    # Checkpointing
    if step % save_every == 0:
        save_checkpoint(model, optimizer, step, f'checkpoint_{step}.pt')
```

---

### 🟡 Text Generation (`decode.py`)

Generate text from trained language models using various sampling strategies.

**Core Decoding Process:**
```
For each timestep t:
  1. Run model: logits = TransformerLM(x_{1:t})[-1]  # Last position
  2. Apply temperature scaling (optional)
  3. Apply top-p filtering (optional)
  4. Sample next token: x_{t+1} ~ softmax(logits)
  5. Append to sequence
  6. Repeat until <|endoftext|> or max_length reached
```

**Temperature Scaling:**
```
softmax(v, τ)_i = exp(v_i / τ) / Σ exp(v_j / τ)

τ → 0: More deterministic (peaks at argmax)
τ = 1: Standard softmax
τ > 1: More random (flatter distribution)
```

**Top-p (Nucleus) Sampling:**
```
1. Sort vocabulary by probability (descending)
2. Find smallest set V(p) such that Σ_{i ∈ V(p)} q_i ≥ p
3. Renormalize and sample from V(p) only

p = 0.9: Use top 90% probability mass
p = 1.0: Use full vocabulary (no filtering)
```

**Implementation:**
```python
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = 'cuda'
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: Trained TransformerLM
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input text to continue
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (default: 1.0)
        top_p: Nucleus sampling threshold (default: 1.0, no filtering)
        device: Device to run on
    
    Returns:
        Generated text string
    """
    # Implementation details...
```

**Usage Examples:**
```python
from cs336_basics.training import generate_text

# Greedy decoding (most likely tokens)
text = generate_text(model, tokenizer, "Once upon a time", 
                     temperature=0.0, max_tokens=100)

# Diverse sampling
text = generate_text(model, tokenizer, "Once upon a time",
                     temperature=0.8, top_p=0.9, max_tokens=100)

# Very creative (high temperature)
text = generate_text(model, tokenizer, "Once upon a time",
                     temperature=1.2, max_tokens=100)
```

**Decoding Strategies:**
- **Greedy** (`temperature=0`): Always pick most likely token
- **Sampling** (`temperature=1`): Sample from full distribution
- **Temperature > 1**: More random/creative outputs
- **Temperature < 1**: More focused/deterministic outputs
- **Top-p sampling**: Filters low-probability tokens before sampling

---

## 🧪 Testing

Run tests for each component:

```bash
# Test cross-entropy loss
uv run pytest -k test_cross_entropy -v

# Test AdamW optimizer
uv run pytest -k test_adamw -v

# Test learning rate schedule
uv run pytest -k test_get_lr_cosine_schedule -v

# Test gradient clipping
uv run pytest -k test_gradient_clipping -v

# Test data loading
uv run pytest -k test_get_batch -v

# Test checkpointing
uv run pytest -k test_checkpointing -v

# Run all training-related tests
uv run pytest tests/test_training.py -v
```

---

## 📊 Resource Accounting (AdamW)

**Memory Requirements (float32):**

For a model with `P` parameters:
- **Parameters**: `4P` bytes
- **Gradients**: `4P` bytes
- **Optimizer State** (AdamW): `8P` bytes (m and v)
- **Activations**: Depends on batch_size, context_length, model architecture
- **Total**: `16P + activations` bytes

**For GPT-2 XL** (1.5B parameters):
- Parameters: 6 GB
- Gradients: 6 GB
- Optimizer: 12 GB
- Base: ~24 GB + activations

**FLOPs per Training Step:**
- Forward pass: ~2 * P * tokens FLOPs
- Backward pass: ~2 × forward = ~4 * P * tokens FLOPs
- Optimizer: Negligible compared to forward/backward
- **Total**: ~6 * P * tokens FLOPs per step

---

## 🚀 Quick Start Guide

1. **Implement Loss Function**:
   ```bash
   # Edit cs336_basics/training/loss.py
   uv run pytest -k test_cross_entropy -v
   ```

2. **Implement AdamW Optimizer**:
   ```bash
   # Edit cs336_basics/training/optimizer.py
   uv run pytest -k test_adamw -v
   ```

3. **Implement LR Scheduling**:
   ```bash
   # Edit cs336_basics/training/lr_scheduler.py
   uv run pytest -k test_get_lr_cosine_schedule -v
   ```

4. **Implement Remaining Components**:
   - Gradient clipping
   - Data loading
   - Checkpointing

5. **Build Training Loop**:
   - Integrate all components in `train.py`
   - Add logging and experiment tracking
   - Test on small dataset (TinyStories)

6. **Implement Text Generation**:
   - Basic sampling from model
   - Temperature scaling
   - Top-p filtering

---

## 📖 References

- **AdamW**: Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization"
- **Learning Rate Schedules**: Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models"
- **Gradient Clipping**: Pascanu et al. (2013). "On the difficulty of training recurrent neural networks"
- **Top-p Sampling**: Holtzman et al. (2020). "The Curious Case of Neural Text Degeneration"

---

## 💡 Tips

### Debugging Training Issues

1. **Loss is NaN**:
   - Check learning rate (too high?)
   - Verify gradient clipping is working
   - Check for numerical stability in cross-entropy

2. **Loss not decreasing**:
   - Verify gradient flow (`torch.autograd.grad_check`)
   - Check learning rate (too low?)
   - Verify optimizer is updating parameters
   - Check data loading (are batches random?)

3. **Training is slow**:
   - Use GPU if available
   - Increase batch size (within memory limits)
   - Profile with `torch.profiler`
   - Check data loading bottleneck

### Best Practices

1. **Start Small**: Test on tiny model/dataset first
2. **Log Everything**: Loss, LR, grad norms, perplexity, time
3. **Validate Often**: Catch overfitting early
4. **Save Checkpoints**: Resume interrupted runs
5. **Monitor GPU**: Watch memory and utilization
6. **Experiment Systematically**: Change one thing at a time

### Common Hyperparameters (TinyStories, 17M params)

```python
# Model
d_model = 288
num_layers = 6
num_heads = 6
d_ff = 4 * d_model  # 1152

# Training
batch_size = 64
context_length = 256
max_steps = 50000
warmup_steps = 5000

# Optimizer
learning_rate = 3e-4
min_lr = 3e-5
betas = (0.9, 0.95)
weight_decay = 0.1

# Regularization
gradient_clip = 1.0

