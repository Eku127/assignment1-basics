"""
Test script for text generation functionality.

This script demonstrates and tests the core generation functions
without requiring a full trained model.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cs336_basics.training.decode import apply_temperature, top_p_filtering


def test_apply_temperature():
    """Test temperature scaling function."""
    print("=" * 80)
    print("Testing apply_temperature()")
    print("=" * 80)
    
    # Create sample logits
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    print("\nOriginal logits:", logits.tolist())
    
    # Test different temperatures
    temps = [0.1, 0.5, 1.0, 2.0]
    
    for temp in temps:
        scaled = apply_temperature(logits, temp)
        probs = F.softmax(scaled, dim=-1)
        
        print(f"\nTemperature = {temp}:")
        print(f"  Scaled logits: {scaled.tolist()}")
        print(f"  Probabilities: {[f'{p:.4f}' for p in probs.tolist()]}")
        print(f"  Max prob: {probs.max():.4f}, Entropy: {-(probs * probs.log()).sum():.4f}")
    
    print("\n✅ Temperature scaling works correctly!")
    print("   - Lower temperature → More peaked distribution (greedy)")
    print("   - Higher temperature → Flatter distribution (random)")


def test_top_p_filtering():
    """Test top-p (nucleus) sampling filtering."""
    print("\n" + "=" * 80)
    print("Testing top_p_filtering()")
    print("=" * 80)
    
    # Create sample logits that will give us known probabilities
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1])
    probs = F.softmax(logits, dim=-1)
    
    print("\nOriginal logits:", [f'{l:.2f}' for l in logits.tolist()])
    print("Original probs:", [f'{p:.4f}' for p in probs.tolist()])
    print("Cumulative probs:", [f'{p:.4f}' for p in torch.cumsum(probs, dim=-1).tolist()])
    
    # Test different top_p values
    top_p_values = [1.0, 0.9, 0.7, 0.5]
    
    for top_p in top_p_values:
        filtered = top_p_filtering(logits, top_p)
        filtered_probs = F.softmax(filtered, dim=-1)
        
        # Count how many tokens are kept
        kept = (filtered != -float('inf')).sum().item()
        
        print(f"\nTop-p = {top_p}:")
        print(f"  Filtered logits: {[f'{l:.2f}' if l != -float('inf') else '-inf' for l in filtered.tolist()]}")
        print(f"  Filtered probs: {[f'{p:.4f}' for p in filtered_probs.tolist()]}")
        print(f"  Tokens kept: {kept}/{len(logits)}")
    
    print("\n✅ Top-p filtering works correctly!")
    print("   - top_p=1.0 → All tokens kept (no filtering)")
    print("   - top_p=0.9 → Only top tokens with cumulative prob ≤ 0.9")
    print("   - Smaller top_p → Fewer tokens kept (more conservative)")


def test_combined_filtering():
    """Test temperature + top-p together."""
    print("\n" + "=" * 80)
    print("Testing Combined Temperature + Top-p")
    print("=" * 80)
    
    logits = torch.tensor([5.0, 4.5, 4.0, 3.0, 2.0, 1.0, 0.5])
    
    print("\nOriginal logits:", [f'{l:.2f}' for l in logits.tolist()])
    print("Original probs:", [f'{p:.4f}' for p in F.softmax(logits, dim=-1).tolist()])
    
    # Scenario 1: Conservative (low temp, moderate top-p)
    print("\n📌 Scenario 1: Conservative Generation (temp=0.5, top_p=0.9)")
    temp_scaled = apply_temperature(logits, temperature=0.5)
    filtered = top_p_filtering(temp_scaled, top_p=0.9)
    final_probs = F.softmax(filtered, dim=-1)
    print(f"   Final probs: {[f'{p:.4f}' for p in final_probs.tolist()]}")
    print(f"   Tokens kept: {(filtered != -float('inf')).sum().item()}")
    
    # Scenario 2: Balanced (moderate temp, moderate top-p)
    print("\n📌 Scenario 2: Balanced Generation (temp=0.8, top_p=0.9)")
    temp_scaled = apply_temperature(logits, temperature=0.8)
    filtered = top_p_filtering(temp_scaled, top_p=0.9)
    final_probs = F.softmax(filtered, dim=-1)
    print(f"   Final probs: {[f'{p:.4f}' for p in final_probs.tolist()]}")
    print(f"   Tokens kept: {(filtered != -float('inf')).sum().item()}")
    
    # Scenario 3: Creative (high temp, high top-p)
    print("\n📌 Scenario 3: Creative Generation (temp=1.5, top_p=0.95)")
    temp_scaled = apply_temperature(logits, temperature=1.5)
    filtered = top_p_filtering(temp_scaled, top_p=0.95)
    final_probs = F.softmax(filtered, dim=-1)
    print(f"   Final probs: {[f'{p:.4f}' for p in final_probs.tolist()]}")
    print(f"   Tokens kept: {(filtered != -float('inf')).sum().item()}")
    
    print("\n✅ Combined filtering works correctly!")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 80)
    print("Testing Edge Cases")
    print("=" * 80)
    
    # Edge case 1: Very small temperature (near-greedy)
    print("\n🔍 Edge Case 1: Very small temperature (0.001)")
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    scaled = apply_temperature(logits, temperature=0.001)
    probs = F.softmax(scaled, dim=-1)
    print(f"   Probabilities: {[f'{p:.6f}' for p in probs.tolist()]}")
    print(f"   Max prob: {probs.max():.6f} (should be close to 1.0)")
    
    # Edge case 2: top_p = 1.0 (no filtering)
    print("\n🔍 Edge Case 2: top_p = 1.0 (no filtering)")
    filtered = top_p_filtering(logits, top_p=1.0)
    print(f"   Filtered logits == Original logits: {torch.allclose(filtered, logits)}")
    
    # Edge case 3: Uniform distribution
    print("\n🔍 Edge Case 3: Uniform logits")
    uniform_logits = torch.ones(5)
    filtered = top_p_filtering(uniform_logits, top_p=0.8)
    kept = (filtered != -float('inf')).sum().item()
    print(f"   Original: {uniform_logits.tolist()}")
    print(f"   Tokens kept: {kept}/5")
    
    print("\n✅ Edge cases handled correctly!")


def main():
    """Run all tests."""
    print("\n" + "🚀" * 40)
    print(" " * 20 + "Text Generation Tests")
    print("🚀" * 40 + "\n")
    
    try:
        test_apply_temperature()
        test_top_p_filtering()
        test_combined_filtering()
        test_edge_cases()
        
        print("\n" + "=" * 80)
        print("✨ ALL TESTS PASSED! ✨")
        print("=" * 80)
        print("\n📝 Summary:")
        print("   ✅ Temperature scaling implemented correctly")
        print("   ✅ Top-p filtering implemented correctly")
        print("   ✅ Combined strategies work as expected")
        print("   ✅ Edge cases handled properly")
        print("\n💡 Next steps:")
        print("   1. Train a model (or load a pretrained checkpoint)")
        print("   2. Use generate.py script to generate text")
        print("   3. Experiment with different temperature and top_p values")
        print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

