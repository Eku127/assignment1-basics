"""
Text Generation Script for Transformer Language Models

This script loads a trained Transformer LM checkpoint and generates text
using various decoding strategies (temperature sampling, top-p/nucleus sampling).

Usage:
    python cs336_basics/training/generate.py \\
        --checkpoint /path/to/checkpoint.pt \\
        --vocab /path/to/vocab.json \\
        --merges /path/to/merges.txt \\
        --prompt "Once upon a time" \\
        --max_new_tokens 100 \\
        --temperature 0.8 \\
        --top_p 0.9
"""

import argparse
import torch
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cs336_basics.transformer import TransformerLM
from cs336_basics.bpe import Tokenizer
from cs336_basics.training import generate_text, generate_text_v2, generate_text_no_cache


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Transformer LM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Greedy decoding (deterministic)
    python cs336_basics/training/generate.py \\
        --checkpoint model.pt \\
        --vocab vocab.json \\
        --merges merges.txt \\
        --prompt "Once upon a time" \\
        --temperature 0.01

    # Diverse sampling with top-p
    python cs336_basics/training/generate.py \\
        --checkpoint model.pt \\
        --vocab vocab.json \\
        --merges merges.txt \\
        --prompt "Once upon a time" \\
        --temperature 0.8 \\
        --top_p 0.9 \\
        --max_new_tokens 200

    # Creative generation (high temperature)
    python cs336_basics/training/generate.py \\
        --checkpoint model.pt \\
        --vocab vocab.json \\
        --merges merges.txt \\
        --prompt "Once upon a time" \\
        --temperature 1.5 \\
        --max_new_tokens 150
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file (.pt)"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to vocabulary JSON file"
    )
    parser.add_argument(
        "--merges",
        type=str,
        required=True,
        help="Path to BPE merges text file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt text to continue from"
    )
    
    # Model architecture (needed to reconstruct model)
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="Vocabulary size (default: 10000)"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=1024,
        help="Maximum context length (default: 1024)"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Model dimension (default: 512)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of Transformer layers (default: 6)"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=2048,
        help="Feed-forward dimension (default: 2048)"
    )
    parser.add_argument(
        "--use_rope",
        action="store_true",
        default=True,
        help="Use Rotary Positional Embedding (default: True)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0). Lower = more conservative, higher = more creative"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling threshold (default: 1.0, no filtering)"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: cuda if available, else cpu)"
    )
    
    # Special tokens
    parser.add_argument(
        "--eos_token",
        type=str,
        default="<|endoftext|>",
        help="End-of-sequence token (default: <|endoftext|>)"
    )
    
    # Output options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def load_model(args, device):
    """Load model from checkpoint."""
    # Create model with specified architecture
    # 构建和checkpoint一致的模型架构
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_rope=args.use_rope,
        device=device,
    )
    
    # Load checkpoint
    if args.verbose:
        print(f"Loading checkpoint from {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        if args.verbose and 'iteration' in checkpoint:
            print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
    else:
        # Assume checkpoint is the model state dict directly
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def load_tokenizer(args):
    """Load tokenizer from vocab and merges files."""
    if args.verbose:
        print(f"Loading tokenizer from {args.vocab} and {args.merges}")
    
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=[args.eos_token]
    )
    
    # Set eos_token_id for easier access
    eos_tokens = tokenizer.encode(args.eos_token)
    if len(eos_tokens) > 0:
        tokenizer.eos_token_id = eos_tokens[0]
    else:
        tokenizer.eos_token_id = None
    
    if args.verbose:
        print(f"Vocabulary size: {len(tokenizer.vocab)}")
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            print(f"EOS token ID: {tokenizer.eos_token_id}")
    
    return tokenizer


def main():
    """Main generation function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Validate arguments
    if args.temperature <= 0:
        print("Error: temperature must be positive", file=sys.stderr)
        sys.exit(1)
    if not 0 < args.top_p <= 1.0:
        print("Error: top_p must be in (0, 1]", file=sys.stderr)
        sys.exit(1)
    if args.max_new_tokens <= 0:
        print("Error: max_new_tokens must be positive", file=sys.stderr)
        sys.exit(1)
    
    # Determine device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU", file=sys.stderr)
        device = "cpu"
    
    if args.verbose:
        print(f"Using device: {device}")
        print(f"Temperature: {args.temperature}")
        print(f"Top-p: {args.top_p}")
        print(f"Max new tokens: {args.max_new_tokens}")
        print()
    
    # Load tokenizer and model
    tokenizer = load_tokenizer(args)
    model = load_model(args, device)
    
    # Print prompt
    print("=" * 80)
    print("PROMPT:")
    print("-" * 80)
    print(args.prompt)
    print("=" * 80)
    print("GENERATED TEXT:")
    print("-" * 80)
    
    # Generate text
    try:
        # Here you can try different generation functions
        # generate_text: self-implemented generation function
        # generate_text_v2: uses model.generate() method
        # generate_text_no_cache: uses model.generate() method without cache
        
        # Record start time
        start_time = time.time()
        
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            eos_token_id=getattr(tokenizer, 'eos_token_id', None),
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        print(generated_text)
        print("=" * 80)
        
        # Print timing information
        print(f"Generation time: {elapsed_time:.2f} seconds ({elapsed_time:.4f} s)")
        
        # Print statistics
        if args.verbose:
            print()
            prompt_tokens = len(tokenizer.encode(args.prompt))
            generated_tokens = len(tokenizer.encode(generated_text))
            new_tokens = generated_tokens - prompt_tokens
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Generated tokens: {new_tokens}")
            print(f"Total tokens: {generated_tokens}")
            if new_tokens > 0:
                tokens_per_second = new_tokens / elapsed_time
                print(f"Generation speed: {tokens_per_second:.2f} tokens/second")
        
    except Exception as e:
        print(f"Error during generation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

