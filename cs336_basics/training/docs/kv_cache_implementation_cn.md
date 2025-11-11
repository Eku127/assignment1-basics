# KV Cache 实现指南

## 📋 目录

1. [什么是 KV Cache？](#什么是-kv-cache)
2. [为什么要使用 KV Cache？](#为什么要使用-kv-cache)
3. [实现 KV Cache 需要修改哪些文件？](#实现-kv-cache-需要修改哪些文件)
4. [各文件具体修改内容](#各文件具体修改内容)
5. [核心实现逻辑](#核心实现逻辑)

---

## 什么是 KV Cache？

**KV Cache 是缓存之前已计算过的 Key 和 Value 矩阵。**

在自回归生成（autoregressive generation）过程中：

### 无 KV Cache 的情况

每次生成新 token 时，都需要重新计算所有已生成 token 的 K 和 V：

- **第 1 步**：输入 `[token1]`，计算 `K1, V1`
- **第 2 步**：输入 `[token1, token2]`，**重新计算** `K1, V1` 和 `K2, V2`
- **第 3 步**：输入 `[token1, token2, token3]`，**重新计算** `K1, V1, K2, V2, K3, V3`
- ...

### 有 KV Cache 的情况

只计算新 token 的 K 和 V，已计算的 K、V 从缓存中读取：

- **第 1 步**：计算 `K1, V1`，存入 cache
- **第 2 步**：从 cache 读取 `K1, V1`，只计算 `K2, V2`，更新 cache
- **第 3 步**：从 cache 读取 `K1, V1, K2, V2`，只计算 `K3, V3`，更新 cache
- ...

---

## 为什么要使用 KV Cache？

### 性能提升

**大幅减少计算量，显著加速文本生成。**

#### 计算复杂度对比

- **无 KV Cache**：
  - 生成 N 个 token 需要计算 `1 + 2 + 3 + ... + N = N(N+1)/2` 次注意力
  - 时间复杂度：`O(N²)`

- **有 KV Cache**：
  - 生成 N 个 token 只需要计算 `N` 次注意力
  - 时间复杂度：`O(N)`

#### 实际效果

- **计算量**：从二次复杂度降到线性复杂度
- **生成速度**：显著提升（特别是生成长序列时）
- **内存开销**：需要额外存储 `(N-1) × (2 × d_model)` 的 K、V 缓存

### 适用场景

**KV Cache 主要用于生成（inference）阶段，训练阶段不需要。**

- **训练时**：每个 batch 的序列是独立的，不需要 cache
- **生成时**：自回归生成，每次只新增一个 token，之前计算的 K、V 可以复用

---

## 实现 KV Cache 需要修改哪些文件？

实现 KV Cache **主要需要在生成（generate）代码中修改**，训练代码不需要改动。

### 需要修改的文件列表

1. **`cs336_basics/training/decode.py`** - 生成函数（管理 cache 生命周期）
2. **`cs336_basics/transformer/transformer_lm.py`** - 模型主类（传递 cache）
3. **`cs336_basics/transformer/transformer_block.py`** - Transformer Block（传递 cache）
4. **`cs336_basics/transformer/multihead_attention.py`** - 多头注意力（核心实现）
5. **`cs336_basics/transformer/attention.py`** - 注意力计算（通常不需要修改）

---

## 各文件具体修改内容

### 1. `cs336_basics/training/decode.py` (生成函数)

**修改内容：**

- 在 `generate_text()` 函数中：
  - **初始化 KV cache**：为每个 layer 创建一个 cache（存储 K 和 V）
  - **修改生成循环**：
    - 第一次调用模型时传入 `None`（表示没有 cache）
    - 后续调用传入更新后的 cache
  - **接收并更新 cache**：每次生成后接收模型返回的更新后的 cache
  - **输入处理**：只传入新 token（而不是整个序列）给模型

**伪代码示例：**
```python
# 初始化 cache（每个 layer 一个）
past_key_values = [None] * model.num_layers

# 第一次生成：传入整个 prompt
prompt_ids = tokenizer.encode(prompt)  # [token1, token2, token3]
prompt_positions = torch.arange(len(prompt_ids))  # [0, 1, 2]
logits, past_key_values = model(
    prompt_ids, 
    token_positions=prompt_positions,
    past_key_values=past_key_values
)

# 后续生成：只传入新 token
current_position = len(prompt_ids)  # 从 prompt 长度开始
for _ in range(max_tokens):
    # 只传入最后一个 token
    new_token = input_ids[:, -1:]  # shape: (1, 1)
    # ⚠️ 重要：token_positions 需要更新为新 token 的绝对位置
    new_token_positions = torch.tensor([current_position])  # [3], [4], [5], ...
    logits, past_key_values = model(
        new_token, 
        token_positions=new_token_positions,
        past_key_values=past_key_values
    )
    # ... 采样新 token ...
    current_position += 1  # 更新位置计数器
```

**关键点：**
- ✅ **每次只传入新 token**：形状 `(1, 1)`，而不是整个序列
- ✅ **token_positions 必须更新**：每次传入新 token 的**绝对位置**（从 0 开始计数）
  - 第一次：`[0, 1, 2, ..., prompt_len-1]`（整个 prompt）
  - 第二次：`[prompt_len]`（第一个新 token）
  - 第三次：`[prompt_len + 1]`（第二个新 token）
  - ...
- ✅ **位置是连续的**：新 token 的位置 = 之前序列的长度

---

### 2. `cs336_basics/transformer/transformer_lm.py` (模型主类)

**修改内容：**

- 在 `forward()` 方法中：
  - **添加参数**：
    - `past_key_values`: KV cache 列表（每个 layer 一个）
    - `use_cache`: 是否使用 cache（默认 True）
  - **传递 cache**：将 cache 传递给每个 `transformer_block`
  - **返回 cache**：返回更新后的 cache（用于下次生成）

**伪代码示例：**
```python
def forward(self, token_ids, past_key_values=None, use_cache=True):
    # ... embedding ...
    
    # 传递 cache 给每个 block
    new_past_key_values = []
    for i, block in enumerate(self.transformer_blocks):
        layer_cache = past_key_values[i] if past_key_values else None
        x, new_cache = block(x, token_positions, past_key_values=layer_cache, use_cache=use_cache)
        new_past_key_values.append(new_cache)
    
    # ... final norm, lm_head ...
    
    return logits, new_past_key_values
```

---

### 3. `cs336_basics/transformer/transformer_block.py` (Transformer Block)

**修改内容：**

- 在 `forward()` 方法中：
  - **添加参数**：
    - `past_key_values`: 该层的 K, V cache
    - `use_cache`: 是否使用 cache
  - **传递 cache**：将 cache 传递给 `attention` 层
  - **返回 cache**：返回更新后的 cache

**伪代码示例：**
```python
def forward(self, x, token_positions=None, past_key_values=None, use_cache=True):
    # Attention sublayer
    residual1 = x
    normed1 = self.norm1(x)
    attention_out, new_cache = self.attention(
        normed1, 
        token_positions, 
        past_key_values=past_key_values,
        use_cache=use_cache
    )
    y = residual1 + attention_out
    
    # FFN sublayer (不需要 cache)
    normed2 = self.norm2(y)
    ff_out = self.feed_forward(normed2)
    z = y + ff_out
    
    return z, new_cache
```

---

### 4. `cs336_basics/transformer/multihead_attention.py` (多头注意力) ⭐ **核心修改**

**修改内容：**

- 在 `forward()` 方法中：
  - **添加参数**：
    - `past_key_values`: K, V cache（tuple 或 None）
    - `use_cache`: 是否使用 cache
  - **Cache 处理逻辑**：
    - **如果 cache 存在**：
      - 只计算新 token 的 Q, K, V（而不是整个序列）
      - 将新的 K, V 追加到 cache
      - 使用 `cache + 新 K/V` 进行 attention 计算
    - **如果 cache 为 None（第一次）**：
      - 正常计算所有 token 的 K, V
      - 返回 K, V 作为新的 cache
  - **返回 cache**：返回更新后的 cache

**伪代码示例：**
```python
def forward(self, x, token_positions=None, past_key_values=None, use_cache=True):
    # Project to Q, K, V
    Q, K, V = self._project_qkv(x)
    Q = self._reshape_for_heads(Q)
    K = self._reshape_for_heads(K)
    V = self._reshape_for_heads(V)
    
    # Apply RoPE to new tokens (before concatenating with cache)
    # ⚠️ 注意：如果使用 cache，past_K 和 past_V 已经应用过 RoPE
    # 所以只需要对新 token 的 Q, K 应用 RoPE
    if self.use_rope and token_positions is not None:
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
    
    # Handle cache (after RoPE for new tokens)
    if past_key_values is not None:
        # 从 cache 中读取之前的 K, V（已经应用过 RoPE）
        past_K, past_V = past_key_values
        # 拼接：cache + 新的 K, V（在序列维度 dim=-2）
        K = torch.cat([past_K, K], dim=-2)  # (..., num_heads, past_len + new_len, d_k)
        V = torch.cat([past_V, V], dim=-2)  # (..., num_heads, past_len + new_len, d_v)
    
    # 创建 causal mask（基于总序列长度）
    total_seq_len = K.shape[-2]  # cache 长度 + 新 token 长度
    causal_mask = self._create_causal_mask(total_seq_len, x.device)
    
    # 计算 attention（使用完整的 K, V）
    attention_output = scaled_dot_product_attention(Q, K, V, causal_mask)
    attention_output = self._combine_heads(attention_output)
    output = self.o_proj(attention_output)
    
    # 返回新的 cache
    new_cache = (K, V) if use_cache else None
    return output, new_cache
```

**关于 token_positions 的说明：**
- 如果 `past_key_values` 为 `None`（第一次生成）：`token_positions` 包含所有 token 的位置 `[0, 1, 2, ...]`
- 如果 `past_key_values` 不为 `None`（后续生成）：`token_positions` 只包含新 token 的位置 `[current_pos]`
- RoPE 会对新 token 的 Q, K 应用位置编码，然后与已编码的 cache 拼接

---

### 5. `cs336_basics/transformer/attention.py` (注意力计算函数)

**修改内容：**

- **通常不需要修改**：这是一个纯函数，只处理给定的 Q, K, V
- **可选优化**：如果需要支持变长 mask，可能需要调整 mask 生成逻辑

---

## 核心实现逻辑

### 1. 第一次生成（初始化 cache）

```python
# 输入：整个 prompt
prompt_ids = [token1, token2, token3]  # shape: (1, 3)

# 计算所有 token 的 K, V
Q, K, V = project_qkv(prompt_ids)  # K, V shape: (1, 3, d_model)

# 存入 cache
cache = (K, V)

# 计算 attention
attention = scaled_dot_product_attention(Q, K, V)
```

### 2. 后续生成（使用 cache）

```python
# 输入：只有新 token
new_token = [token4]  # shape: (1, 1)

# 只计算新 token 的 K, V
Q_new, K_new, V_new = project_qkv(new_token)  # K_new, V_new shape: (1, 1, d_model)

# 从 cache 读取之前的 K, V
K_past, V_past = cache  # shape: (1, 3, d_model)

# 拼接：cache + 新的 K, V
K = torch.cat([K_past, K_new], dim=1)  # shape: (1, 4, d_model)
V = torch.cat([V_past, V_new], dim=1)  # shape: (1, 4, d_model)

# 更新 cache
cache = (K, V)

# 计算 attention（使用完整的 K, V）
attention = scaled_dot_product_attention(Q_new, K, V)
```

### 3. 关键点

- **输入处理**：生成时只传入新 token（形状 `(1, 1)`），而不是整个序列
- **Token Positions 更新**：⚠️ **非常重要**
  - 每次生成时，`token_positions` 必须传入新 token 的**绝对位置**
  - 第一次：`[0, 1, 2, ..., prompt_len-1]`（整个 prompt 的位置）
  - 后续：`[prompt_len]`, `[prompt_len+1]`, `[prompt_len+2]`, ...（新 token 的绝对位置）
  - 位置从 0 开始，连续递增
- **Cache 格式**：每个 layer 的 cache 是一个 tuple `(K, V)`
- **序列维度拼接**：在序列长度维度（通常是 `-2` 或 `dim=1`）拼接 cache 和新计算的 K, V
- **Causal Mask**：需要根据当前序列长度（cache 长度 + 新 token 长度）动态生成 mask

---

## 总结

实现 KV Cache 需要修改 **4 个核心文件**：

1. **`decode.py`** - 管理 cache 的生命周期
2. **`transformer_lm.py`** - 传递 cache 给 blocks
3. **`transformer_block.py`** - 传递 cache 给 attention
4. **`multihead_attention.py`** - 实现 cache 的更新和使用逻辑（**核心修改**）

**核心思想**：缓存已计算的 K、V，避免重复计算，将生成复杂度从 `O(N²)` 降到 `O(N)`，显著加速自回归生成。

---

## 参考资源

- [Hugging Face Transformers - Generation with KV Cache](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [LLM Inference Optimization Techniques](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)

