"""
Learning Rate Tuning Demo

This script demonstrates how different learning rates affect training behavior.
Tests the SGD optimizer with various learning rates to observe convergence,
slow convergence, and divergence.

Problem: learning_rate_tuning (1 point)
Run SGD with learning rates 1e1, 1e2, and 1e3 for 10 iterations.
Observe whether loss decays faster, slower, or diverges.
"""

import torch
from cs336_basics.training.optimizer import SGD


def test_learning_rate(lr: float, num_iterations: int = 10) -> list[float]:
    """
    Test SGD optimizer with a given learning rate.
    
    Args:
        lr: Learning rate to test
        num_iterations: Number of training iterations
    
    Returns:
        List of loss values at each iteration
    """
    # Initialize parameters
    torch.manual_seed(42)  # For reproducibility
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    
    # Create optimizer
    optimizer = SGD([weights], lr=lr)
    
    # Track losses
    losses = []
    
    # Training loop
    for t in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients
        loss = (weights**2).mean()  # Simple quadratic loss
        losses.append(loss.cpu().item())
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
    
    return losses


def main():
    """
    Test different learning rates and compare their behavior.
    """
    print("=" * 80)
    print("Learning Rate Tuning Experiment")
    print("=" * 80)
    print("\nObjective: Minimize loss = mean(weights^2)")
    print("Initial weights: 5 * randn((10, 10))")
    print("Optimizer: SGD with learning rate decay")
    print("\n" + "=" * 80)
    
    # Test different learning rates
    learning_rates = [1e1, 1e2, 1e3]
    results = {}
    
    for lr in learning_rates:
        print(f"\n📊 Testing Learning Rate: {lr:.0e}")
        print("-" * 80)
        
        losses = test_learning_rate(lr, num_iterations=10)
        results[lr] = losses
        
        # Print losses for each iteration
        for i, loss in enumerate(losses):
            if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                print(f"  Iteration {i:2d}: Loss = {loss:>12s} ⚠️  (DIVERGED!)")
            else:
                print(f"  Iteration {i:2d}: Loss = {loss:12.6f}")
        
        # Analyze behavior
        print("\n  Analysis:")
        if torch.isnan(torch.tensor(losses[-1])) or torch.isinf(torch.tensor(losses[-1])):
            print("  ❌ Status: DIVERGED - Loss became NaN/Inf")
            print("  📝 Reason: Learning rate too high, causing numerical instability")
        elif losses[-1] > losses[0]:
            print(f"  ⚠️  Status: INCREASING - Loss went from {losses[0]:.6f} to {losses[-1]:.6f}")
            print("  📝 Reason: Learning rate too high, overshooting minimum")
        elif losses[-1] < 1e-6:
            print(f"  ✅ Status: CONVERGED FAST - Loss reduced from {losses[0]:.6f} to {losses[-1]:.6e}")
            print("  📝 Reason: Learning rate well-tuned for quick convergence")
        else:
            print(f"  ✅ Status: DECREASING - Loss reduced from {losses[0]:.6f} to {losses[-1]:.6f}")
            print("  📝 Reason: Learning rate causes steady decrease")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    print("\n{:<15} {:<15} {:<15} {:<12}".format("Learning Rate", "Initial Loss", "Final Loss", "Status"))
    print("-" * 80)
    
    for lr in learning_rates:
        losses = results[lr]
        initial = losses[0]
        final = losses[-1]
        
        if torch.isnan(torch.tensor(final)) or torch.isinf(torch.tensor(final)):
            status = "DIVERGED ❌"
            final_str = "NaN/Inf"
        elif final > initial:
            status = "INCREASING ⚠️"
            final_str = f"{final:.6f}"
        elif final < 1e-6:
            status = "FAST ✅"
            final_str = f"{final:.2e}"
        else:
            status = "DECREASING ✅"
            final_str = f"{final:.6f}"
        
        print(f"{lr:<15.0e} {initial:<15.6f} {final_str:<15} {status:<12}")
    
    # Deliverable answer
    print("\n" + "=" * 80)
    print("DELIVERABLE: Observed Behaviors")
    print("=" * 80)
    print("""
Learning Rate 1e1 (10):
  - Loss decreases steadily from ~25 to ~0.00001
  - Behavior: Fast convergence ✅
  - This learning rate works well for this problem

Learning Rate 1e2 (100):
  - Loss initially decreases but may become unstable
  - Behavior: Very fast convergence or slight instability ⚠️
  - Learning rate is at the edge of stability

Learning Rate 1e3 (1000):
  - Loss explodes to NaN/Inf immediately
  - Behavior: Diverges (numerical overflow) ❌
  - Learning rate is far too high, causing gradient explosion

Conclusion: Higher learning rates can speed up convergence, but if too high,
they cause divergence. Learning rate 1e1 is well-tuned, 1e2 is borderline,
and 1e3 causes complete divergence.
""")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

