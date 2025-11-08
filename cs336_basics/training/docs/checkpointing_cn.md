# 训练中的 Checkpointing（检查点）机制

`Checkpointing`（保存检查点）是在机器学习模型训练过程中一项至关重要的工程实践。其核心思想是：**在训练期间，周期性地将模型的完整状态保存到磁盘上。**

---

## 1. 为什么需要 Checkpointing？

保存检查点主要有两个目的：

### a. 保证训练的可恢复性 (Resilience)

长时间的训练任务（可能持续数天或数周）很容易因各种意外而中断，例如：
- 程序崩溃或代码 Bug
- 服务器宕机或重启
- 集群任务超时

如果没有检查点，任何中断都意味着之前的训练时间和计算资源全部白费。通过加载最近的检查点，训练可以从中断处继续，最大限度地减少损失。

### b. 分析和使用中间模型 (Analysis & Intermediate Use)

我们不仅关心最终训练完成的模型，中间过程也同样重要。检查点使我们能够：
- **研究训练动态：** 分析模型性能（如损失、准确率）在训练过程中的变化趋势。
- **获取中间模型：** 从不同训练阶段的模型进行采样或评估，观察其能力演变。
- **选择最佳模型：** 在训练结束后，从所有保存的检查点中，挑选在验证集上表现最好的模型作为最终版本，而非默认使用训练结束时的最后一个模型。

---

## 2. 一个完整的 Checkpoint 应该包含什么？

为了能够精确地“恢复现场”，一个检查点通常需要包含以下三个核心组件：

1.  **模型权重 (Model Weights):**
    这是模型学到的所有参数，是检查点的最基本构成。
    -   在 PyTorch 中通过 `model.state_dict()` 获取。

2.  **优化器状态 (Optimizer State):**
    对于像 `AdamW` 这样有状态的优化器，必须保存其内部状态（例如，动量和方差的估计值）。如果仅恢复模型权重而不恢复优化器状态，会严重影响后续训练的稳定性和收敛效果。
    -   在 PyTorch 中通过 `optimizer.state_dict()` 获取。

3.  **当前迭代步数 (Iteration Number):**
    为了准确地接续学习率调度（Learning Rate Schedule），必须记录训练中断时进行到了第几步（epoch 或 iteration）。
    -   通常是一个整数。

---

## 3. 如何在 PyTorch 中实现？

PyTorch 提供了简单易用的 API 来实现 Checkpointing：

1.  **将所有状态打包到一个字典中：**
    ```python
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': current_iteration
    }
    ```

2.  **保存到文件：**
    使用 `torch.save()` 可以将包含各种 Python 对象的字典轻松保存到磁盘。
    ```python
    # ".pt" or ".pth" are common extensions
    torch.save(checkpoint, 'my_checkpoint.pt')
    ```

3.  **从文件加载：**
    使用 `torch.load()` 读回字典，然后用 `load_state_dict()` 方法将状态恢复到模型和优化器中。
    ```python
    checkpoint = torch.load('my_checkpoint.pt')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']

    # Now, you can resume training from this state
    ```

**总结：** Checkpointing 是确保长时间模型训练过程鲁棒、可靠和可分析的标准实践。
