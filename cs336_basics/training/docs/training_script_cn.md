# `train.py` 训练脚本使用指南

本文档为 `train.py` 脚本提供了一份详细的中文说明。该脚本是训练 Transformer 语言模型的核心，它整合了数据加载、模型构建、优化、日志记录和检查点等所有关键组件。

---

## 1. 脚本概述

`train.py` 是一个功能完备、高度可配置的训练启动器。它的设计遵循了现代深度学习项目的最佳实践，其核心特点是**通过命令行参数进行配置**，用户无需修改源代码即可进行各种实验。

脚本的主要工作流程包括：
1.  解析用户通过命令行传入的参数。
2.  初始化 Weights & Biases (W&B) 日志（如果启用）。
3.  使用内存映射（`memmap`）高效加载训练和验证数据集。
4.  根据指定的超参数（如 `d_model`, `num_layers` 等）初始化 Transformer 模型。
5.  初始化 AdamW 优化器。
6.  （可选）从指定的检查点（checkpoint）恢复训练。
7.  进入主训练循环，执行模型的训练、验证、记录日志和保存检查点。
8.  训练结束后，保存最终的模型。

---

## 2. 如何运行脚本

你可以通过命令行直接运行此脚本。以下是一个基本的运行示例：

```bash
python cs336_basics/training/train.py \
    --train_data /path/to/train_tokens.npy \
    --val_data /path/to/val_tokens.npy \
    --vocab_size 50257 \
    --d_model 288 \
    --num_layers 6 \
    --num_heads 6 \
    --batch_size 64 \
    --max_steps 50000 \
    --learning_rate 3e-4 \
    --checkpoint_dir ./checkpoints/my_model \
    --device cuda
```

---

## 3. 命令行参数详解

脚本的所有行为都由命令行参数控制。这些参数可以分为以下几类：

### a. 数据参数
-   `--train_data` (必需): 训练数据的 `.npy` 文件路径。
-   `--val_data` (必需): 验证数据的 `.npy` 文件路径。

### b. 模型参数
-   `--vocab_size` (必需): 词汇表的大小。
-   `--context_length`: 模型的上下文窗口大小（默认为 `256`）。
-   `--d_model`: 模型的隐藏层维度（默认为 `288`）。
-   `--num_layers`: Transformer 层的数量（默认为 `6`）。
-   `--num_heads`: 多头注意力机制中的头数（默认为 `6`）。
-   `--d_ff`: 前馈网络（FFN）的内部维度（默认为 `4 * d_model`）。
-   `--use_rope`: 是否使用 RoPE 位置编码（默认为 `True`）。

### c. 训练参数
-   `--batch_size`: 批次大小（默认为 `64`）。
-   `--max_steps`: 最大训练步数（默认为 `50000`）。
-   `--learning_rate`: 学习率调度器中的最大学习率（默认为 `3e-4`）。
-   `--min_lr`: 学习率调度器中的最小学习率（默认为 `3e-5`）。
-   `--warmup_steps`: 学习率预热（warmup）的步数（默认为 `5000`）。
-   `--weight_decay`: AdamW 优化器的权重衰减系数（默认为 `0.1`）。
-   `--grad_clip`: 梯度裁剪的阈值（默认为 `1.0`）。

### d. 日志与检查点参数
-   `--log_every`: 每隔多少步记录一次训练日志（默认为 `100`）。
-   `--val_every`: 每隔多少步进行一次验证（默认为 `1000`）。
-   `--save_every`: 每隔多少步保存一次检查点（默认为 `5000`）。
-   `--checkpoint_dir`: 保存检查点的目录（默认为 `./checkpoints`）。
-   `--resume_from`: 从指定的检查点文件路径恢复训练（默认为 `None`）。

### e. Weights & Biases (W&B) 参数
-   `--use_wandb`: 是否启用 W&B 进行实验跟踪（默认不启用）。
-   `--wandb_project`: W&B 项目的名称（若启用 W&B，则此项必需）。
-   `--wandb_name`: 本次运行在 W&B 上的名称（可选）。

### f. 设备参数
-   `--device`: 训练设备，如 `cuda`, `cpu`, `mps`（默认为 `cuda`）。

---

## 4. 核心功能实现

### a. `main()` 函数
-   **作用**: 程序的入口。它使用 `argparse` 定义并解析上述所有命令行参数，然后将这些参数作为关键字参数传递给 `train()` 函数。

### b. `train()` 函数
-   **作用**: 包含了训练的全部核心逻辑。
-   **训练循环 (Training Loop)**: 这是脚本的心脏，在 `for step in range(...)` 循环中，每一步都执行以下操作：
    1.  **更新学习率**: 调用 `get_lr_cosine_schedule` 计算当前步数对应的学习率，并更新优化器。
    2.  **获取数据批次**: 调用 `get_batch` 从 `train_data` 中采样一个批次的数据。
    3.  **前向传播**: 将输入数据送入模型，得到 `logits`。
    4.  **计算损失**: 使用 `cross_entropy` 计算模型输出与目标之间的损失。
    5.  **反向传播**: 调用 `loss.backward()` 计算梯度。
    6.  **梯度裁剪**: 调用 `clip_gradients` 防止梯度爆炸。
    7.  **参数更新**: 调用 `optimizer.step()` 更新模型权重。
    8.  **日志记录**: 定期（`log_every`）打印训练指标（Loss, PPL, LR 等）到控制台，并（如果启用）通过 `wandb.log` 发送到 W&B。
    9.  **验证**: 定期（`val_every`）调用 `evaluate()` 函数在验证集上评估模型性能，并记录结果。
    10. **保存检查点**: 定期（`save_every`）调用 `save_checkpoint` 保存当前训练状态。

### c. `evaluate()` 函数
-   **作用**: 专门用于在验证集上评估模型性能。
-   **机制**: 它会在一个 `torch.no_grad()` 上下文管理器中运行，这会禁用梯度计算，从而加速评估过程并节省内存。它会采样 `num_batches` 个批次的数据，计算平均损失并返回。
