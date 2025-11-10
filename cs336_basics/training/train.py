"""
Main training script for Transformer language models.

Integrates all training components:
- Data loading
- Model initialization
- Optimizer and learning rate scheduling
- Training loop with validation
- Checkpointing
- Logging

This script serves as a template for training your models.
You should customize it based on your specific needs.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Optional

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import our training utilities
from cs336_basics.training.loss import cross_entropy, perplexity
from cs336_basics.training.optimizer import AdamW
from cs336_basics.training.lr_scheduler import get_lr_cosine_schedule
from cs336_basics.training.gradient_clipping import clip_gradients
from cs336_basics.training.data_loader import get_batch
from cs336_basics.training.checkpoint import save_checkpoint, load_checkpoint

# Import model
from cs336_basics.transformer.transformer_lm import TransformerLM


def train(
    # Data parameters
    train_data_path: str,
    val_data_path: str,
    # Model parameters
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: Optional[int] = None,
    use_rope: bool = True,
    # Training parameters
    batch_size: int = 64,
    max_steps: int = 50000,
    learning_rate: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 5000,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    # Logging and checkpointing
    log_every: int = 100,
    val_every: int = 1000,
    save_every: int = 5000,
    checkpoint_dir: str = './checkpoints',
    resume_from: Optional[str] = None,
    # Wandb (optional)
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    # Device
    device: str = 'cuda',
):
    """
    Main training function.
    
    Args:
        train_data_path: Path to training data (.npy file with token IDs)
        val_data_path: Path to validation data (.npy file with token IDs)
        vocab_size: Size of vocabulary
        context_length: Maximum sequence length
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: FFN inner dimension (default: 4 * d_model)
        use_rope: Whether to use RoPE positional encoding
        batch_size: Training batch size
        max_steps: Maximum training steps
        learning_rate: Maximum learning rate
        min_lr: Minimum learning rate
        warmup_steps: Number of warmup steps
        betas: AdamW beta parameters
        weight_decay: Weight decay coefficient
        grad_clip: Gradient clipping threshold
        log_every: Log training metrics every N steps
        val_every: Run validation every N steps
        save_every: Save checkpoint every N steps
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: Wandb project name (required if use_wandb=True)
        wandb_name: Wandb run name (optional)
        device: Device to train on ('cpu', 'cuda', 'mps')
    """
    
    # Initialize wandb if requested
    if use_wandb:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        if wandb_project is None:
            raise ValueError("wandb_project must be specified when use_wandb=True")
        
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                # Data
                'train_data_path': train_data_path,
                'val_data_path': val_data_path,
                # Model
                'vocab_size': vocab_size,
                'context_length': context_length,
                'd_model': d_model,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'd_ff': d_ff,
                'use_rope': use_rope,
                # Training
                'batch_size': batch_size,
                'max_steps': max_steps,
                'learning_rate': learning_rate,
                'min_lr': min_lr,
                'warmup_steps': warmup_steps,
                'betas': betas,
                'weight_decay': weight_decay,
                'grad_clip': grad_clip,
                # Logging
                'log_every': log_every,
                'val_every': val_every,
                'save_every': save_every,
                'device': device,
            }
        )
        print(f"Wandb initialized: project={wandb_project}, name={wandb_name}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data (memory-mapped for efficiency)
    print(f"Loading training data from {train_data_path}...")
    train_data = np.load(train_data_path, mmap_mode='r')
    print(f"Training data: {len(train_data):,} tokens")
    
    print(f"Loading validation data from {val_data_path}...")
    val_data = np.load(val_data_path, mmap_mode='r')
    print(f"Validation data: {len(val_data):,} tokens")
    
    # Initialize model
    if d_ff is None:
        d_ff = 4 * d_model
    
    print("\nInitializing model...")
    print(f"  vocab_size={vocab_size}, context_length={context_length}")
    print(f"  d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}")
    print(f"  d_ff={d_ff}, use_rope={use_rope}")
    
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        use_rope=use_rope,
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Log model info to wandb
    if use_wandb:
        wandb.config.update({'num_params': num_params})
        wandb.config.update({'num_params_M': num_params / 1e6})
    
    # Initialize optimizer
    print("\nInitializing optimizer...")
    print(f"  lr={learning_rate}, min_lr={min_lr}")
    print(f"  betas={betas}, weight_decay={weight_decay}")
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from is not None:
        print(f"\nResuming from checkpoint: {resume_from}")
        start_step = load_checkpoint(resume_from, model, optimizer)
        print(f"Resumed from step {start_step}")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    model.train()
    
    for step in range(start_step, max_steps):
        # Update learning rate
        lr = get_lr_cosine_schedule(
            t=step,
            max_lr=learning_rate,
            min_lr=min_lr,
            warmup_iters=warmup_steps,
            cosine_cycle_iters=max_steps,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Sample batch
        inputs, targets = get_batch(
            train_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = clip_gradients(model.parameters(), max_norm=grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if step % log_every == 0:
            ppl = torch.exp(loss).item()
            print(f"Step {step:6d} | Loss: {loss.item():.4f} | "
                  f"PPL: {ppl:7.2f} | LR: {lr:.2e} | "
                  f"GradNorm: {grad_norm:.2f}")
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/perplexity': ppl,
                    'train/learning_rate': lr,
                    'train/grad_norm': grad_norm,
                    'step': step,
                }, step=step)
        
        # Validation
        if step % val_every == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                val_loss = evaluate(
                    model, val_data, batch_size, context_length, device
                )
                val_ppl = torch.exp(val_loss).item()
                print(f"\n{'='*80}")
                print(f"Validation at step {step}")
                print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
                print(f"{'='*80}\n")
                
                # Log validation metrics to wandb
                if use_wandb:
                    wandb.log({
                        'val/loss': val_loss.item(),
                        'val/perplexity': val_ppl,
                        'step': step,
                    }, step=step)
            model.train()
        
        # Checkpointing
        if step % save_every == 0 and step > 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f'checkpoint_{step}.pt'
            )
            save_checkpoint(model, optimizer, step, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, 'checkpoint_final.pt')
    save_checkpoint(model, optimizer, max_steps, final_path)
    print(f"\nTraining complete! Final checkpoint saved to {final_path}")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()


def evaluate(
    model: nn.Module,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int = 100,
) -> torch.Tensor:
    """
    Evaluate model on validation data.
    
    Args:
        model: Model to evaluate
        data: Validation data
        batch_size: Batch size
        context_length: Sequence length
        device: Device
        num_batches: Number of batches to evaluate on
    
    Returns:
        avg_loss: Average validation loss
    """
    total_loss = 0.0
    
    for _ in range(num_batches):
        inputs, targets = get_batch(data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        total_loss += loss.item()
    
    return torch.tensor(total_loss / num_batches)


def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(description='Train a Transformer LM')
    
    # Data
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data (.npy)')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation data (.npy)')
    
    # Model
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=288)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=None)
    parser.add_argument('--use_rope', action='store_true', default=True)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=3e-5)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW β1 parameter')
    parser.add_argument('--beta2', type=float, default=0.95, help='AdamW β2 parameter')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Logging
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--val_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume_from', type=str, default=None)
    
    # Wandb
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='LLM_Learning',
                       help='Wandb project name (required if --use_wandb)')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name (optional)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_rope=args.use_rope,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        val_every=args.val_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        device=args.device,
    )


if __name__ == '__main__':
    main()

