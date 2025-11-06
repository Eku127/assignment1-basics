# AdamW Memory Analysis for Transformer Training

## 📋 Problem Statement

Calculate the **peak memory** required when training a Transformer language model with AdamW optimizer using **float32** precision.

**Given Parameters:**
- `vocab_size`: Vocabulary size
- `context_length`: Maximum sequence length
- `d_model`: Model dimension
- `num_layers`: Number of transformer layers
- `num_heads`: Number of attention heads
- `batch_size`: Training batch size
- `d_ff = 4 × d_model`: Feed-forward network dimension

**Memory Components to Calculate:**
1. Parameters
2. Activations
3. Gradients
4. Optimizer State (AdamW)

**Assumption:** float32 (4 bytes per element)

---

## 🔢 Detailed Calculation

### 1️⃣ **Parameters Memory**

#### **Input Embedding Layer:**
```
vocab_size × d_model
```

#### **Per Transformer Block:**

**a) Pre-RMSNorm:**
```
d_model  (gain parameter only, no bias)
```

**b) Multi-head Self-Attention:**
- Q, K, V projection matrices:
  ```
  3 × (d_model × d_model) = 3d_model²
  ```
- Output projection:
  ```
  d_model × d_model = d_model²
  ```
- Subtotal: `4d_model²`

**c) Post-RMSNorm:**
```
d_model
```

**d) Position-wise Feed-Forward Network:**
- W1 (first layer):
  ```
  d_model × d_ff = d_model × 4d_model = 4d_model²
  ```
- W2 (second layer):
  ```
  d_ff × d_model = 4d_model × d_model = 4d_model²
  ```
- Subtotal: `8d_model²`

**Total Parameters per Transformer Block:**
```
d_model + 4d_model² + d_model + 8d_model² = 2d_model + 12d_model²
```

#### **Final Layers:**
- Final RMSNorm: `d_model`
- Output Embedding (assuming weight sharing with input): `0`

#### **Total Parameters:**
```
P = vocab_size × d_model + num_layers × (12d_model² + 2d_model) + d_model

Simplified:
P = vocab_size × d_model + num_layers × 12d_model² + (2 × num_layers + 1) × d_model
```

**Memory (bytes):**
```
Memory_params = 4P bytes
```

---

### 2️⃣ **Activations Memory**

Activations depend on `batch_size` and `context_length`. We calculate activations that must be stored during forward pass for backward pass.

**Notation:**
- `B = batch_size`
- `L = context_length`
- `N = num_layers`
- `d = d_model`
- `h = num_heads`

#### **Per Transformer Block Activations:**

**a) RMSNorm Input/Output:**
```
B × L × d
```

**b) Multi-head Attention:**
- QKV projections output:
  ```
  3 × B × L × d
  ```
- Attention scores (Q^T K):
  ```
  B × h × L × L
  ```
- Softmax output (saved for backprop):
  ```
  B × h × L × L
  ```
- Attention output (weighted sum of values):
  ```
  B × L × d
  ```
- Output projection:
  ```
  B × L × d
  ```

**c) Second RMSNorm:**
```
B × L × d
```

**d) Feed-Forward Network:**
- W1 output (before SiLU):
  ```
  B × L × d_ff = B × L × 4d
  ```
- SiLU output:
  ```
  B × L × 4d
  ```
- W2 output:
  ```
  B × L × d
  ```

**e) Residual Connections:**
Need to save residual inputs (2 times):
```
2 × B × L × d
```

#### **Total Activations per Block:**
```
Activations_per_block = B × L × d × (1 + 3 + 1 + 1 + 1 + 1 + 4 + 1 + 2)
                      + 2 × B × h × L × L
                      = 15BLd + 2BhL²
```

#### **All Transformer Layers:**
```
Activations_transformer = N × (15BLd + 2BhL²)
```

#### **Other Activations:**
- Input Embedding output: `B × L × d`
- Final RMSNorm output: `B × L × d`
- Output logits (for cross-entropy): `B × L × vocab_size`

#### **Total Activations:**
```
A = N × (15BLd + 2BhL²) + 2BLd + BL × vocab_size

Simplified:
A = BLd(15N + 2) + 2NBhL² + BL × vocab_size
```

**Memory (bytes):**
```
Memory_activations = 4A bytes
```

---

### 3️⃣ **Gradients Memory**

Each parameter has a corresponding gradient, so gradient memory equals parameter memory:

```
G = P = vocab_size × d_model + num_layers × 12d_model² + (2 × num_layers + 1) × d_model
```

**Memory (bytes):**
```
Memory_gradients = 4G = 4P bytes
```

---

### 4️⃣ **Optimizer State (AdamW) Memory**

AdamW maintains two states per parameter:
- **First moment (m)**: Same size as parameters
- **Second moment (v)**: Same size as parameters

```
Optimizer_state = 2P
```

**Memory (bytes):**
```
Memory_optimizer = 4 × 2P = 8P bytes
```

---

## 📊 Total Memory Formula

### **Total Peak Memory:**

```
Total_Memory = Memory_params + Memory_activations + Memory_gradients + Memory_optimizer
             = 4P + 4A + 4P + 8P
             = 16P + 4A
```

Where:
```
P = vocab_size × d_model + 12 × num_layers × d_model² + (2 × num_layers + 1) × d_model

A = batch_size × context_length × d_model × (15 × num_layers + 2) 
    + 2 × batch_size × num_heads × num_layers × context_length²
    + batch_size × context_length × vocab_size
```

### **Component Breakdown:**

| Component | Memory (bytes) | Formula |
|-----------|---------------|---------|
| **Parameters** | `4P` | `4 × [vocab_size × d_model + 12 × num_layers × d_model² + (2 × num_layers + 1) × d_model]` |
| **Activations** | `4A` | `4 × [batch_size × context_length × d_model × (15 × num_layers + 2) + 2 × batch_size × num_heads × num_layers × context_length² + batch_size × context_length × vocab_size]` |
| **Gradients** | `4P` | Same as Parameters |
| **Optimizer** | `8P` | `2 × 4P` (m and v moments) |
| **Total** | `16P + 4A` | Sum of above |

---

## 🎯 Simplified Symbolic Formula

**Notation:**
- `V = vocab_size`
- `L = context_length`
- `N = num_layers`
- `d = d_model`
- `h = num_heads`
- `B = batch_size`

**Parameters:**
```
P = Vd + 12Nd² + (2N + 1)d
  ≈ Vd + 12Nd²  (dominant terms)
```

**Activations:**
```
A = BLd(15N + 2) + 2NBhL² + BLV
  = BL[d(15N + 2) + 2NhL + V]
```

**Total Memory:**
```
Total = 16P + 4A
      = 16[Vd + 12Nd² + (2N + 1)d] + 4BL[d(15N + 2) + 2NhL + V]
```

**In bytes (float32):**
```
Total_bytes = 4 × (16P + 4A)
```

**In GB:**
```
Total_GB = (16P + 4A) × 4 / (1024³)
```

---

## 💡 Concrete Example

### **Small Model Configuration:**
```python
vocab_size = 50,000
context_length = 512
d_model = 768
num_layers = 12
num_heads = 12
batch_size = 8
d_ff = 4 × 768 = 3,072
```

### **Parameters Calculation:**
```
P = 50,000 × 768 + 12 × 12 × 768² + (2×12 + 1) × 768
  = 38,400,000 + 84,934,656 + 19,200
  = 123,353,856
  ≈ 123M parameters
```

**Memory:** `123M × 4 bytes = 492 MB`

### **Activations Calculation:**
```
A = 8 × 512 × 768 × (15×12 + 2) 
    + 2 × 8 × 12 × 12 × 512² 
    + 8 × 512 × 50,000

  = 8 × 512 × 768 × 182
    + 2 × 8 × 12 × 12 × 262,144
    + 204,800,000

  = 571,113,472 + 603,979,776 + 204,800,000
  = 1,379,893,248
  ≈ 1,380M elements
```

**Memory:** `1,380M × 4 bytes = 5,520 MB`

### **Gradients:**
```
G = P = 123M parameters
```

**Memory:** `123M × 4 bytes = 492 MB`

### **Optimizer State (AdamW):**
```
Optimizer = 2P = 2 × 123M = 246M elements
```

**Memory:** `246M × 4 bytes = 984 MB`

### **Total Memory:**
```
Total = Parameters + Activations + Gradients + Optimizer
      = 492 MB + 5,520 MB + 492 MB + 984 MB
      = 7,488 MB
      ≈ 7.5 GB
```

### **Breakdown by Component:**
| Component | Elements | Memory (MB) | Percentage |
|-----------|----------|-------------|------------|
| Parameters | 123M | 492 | 6.6% |
| Activations | 1,380M | 5,520 | 73.7% |
| Gradients | 123M | 492 | 6.6% |
| Optimizer State | 246M | 984 | 13.1% |
| **Total** | **1,872M** | **7,488** | **100%** |

**Key Insights:**
- 📊 **Activations dominate** memory usage (73.7%)
- 🔧 **Optimizer adds 13.1%** overhead (AdamW needs 2× parameter memory)
- 💡 **Reducing batch_size** most effectively reduces memory
- 🎯 **Gradient checkpointing** can reduce activation memory

---

## 📝 Final Deliverable

### **Answer to Question (a):**

**Parameters (bytes):**
```
4 × [vocab_size × d_model + 12 × num_layers × d_model² + (2 × num_layers + 1) × d_model]
```

**Activations (bytes):**
```
4 × [batch_size × context_length × d_model × (15 × num_layers + 2) 
     + 2 × batch_size × num_heads × num_layers × context_length²
     + batch_size × context_length × vocab_size]
```

**Gradients (bytes):**
```
4 × [vocab_size × d_model + 12 × num_layers × d_model² + (2 × num_layers + 1) × d_model]
(Same as Parameters)
```

**Optimizer State (bytes):**
```
8 × [vocab_size × d_model + 12 × num_layers × d_model² + (2 × num_layers + 1) × d_model]
(2× Parameters: one for first moment m, one for second moment v)
```

**Total Peak Memory (bytes):**
```
16 × [vocab_size × d_model + 12 × num_layers × d_model² + (2 × num_layers + 1) × d_model]
+ 4 × [batch_size × context_length × d_model × (15 × num_layers + 2) 
       + 2 × batch_size × num_heads × num_layers × context_length²
       + batch_size × context_length × vocab_size]
```

Or in compact form:
```
Total = 16P + 4A

where:
  P = vocab_size × d_model + 12 × num_layers × d_model² + (2 × num_layers + 1) × d_model
  A = batch_size × context_length × [d_model(15 × num_layers + 2) + 2 × num_heads × num_layers × context_length + vocab_size]
```

---

## 🔍 Memory Optimization Strategies

1. **Gradient Checkpointing**: Trade computation for memory by recomputing activations during backward pass
   - Can reduce activation memory by ~50-75%
   - Increases training time by ~20-30%

2. **Mixed Precision (FP16/BF16)**: Use 16-bit floats instead of 32-bit
   - Reduces memory by ~50%
   - Requires careful handling of numerical stability

3. **Gradient Accumulation**: Split batch into micro-batches
   - Effective batch size = micro_batch × accumulation_steps
   - Reduces activation memory proportionally

4. **Model Parallelism**: Split model across multiple GPUs
   - Distributes parameter, gradient, and optimizer memory
   - Useful for very large models

5. **Optimizer State Offloading**: Store optimizer states in CPU memory
   - Reduces GPU memory usage
   - May slow down training due to data transfer

---

## 📚 References

- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR.
- Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.
- Shoeybi, M., et al. (2019). Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.

---

## 📄 Document Information

- **Course**: CS336 - Language Modeling from Scratch
- **Assignment**: Assignment 1 - Basics
- **Problem**: 4.2.2 - AdamW Accounting (Part a)
- **Author**: Generated for educational purposes
- **Date**: 2025

---

*This document provides a detailed breakdown of memory requirements for training Transformer models with AdamW optimizer, including all intermediate calculations and a concrete example.*

