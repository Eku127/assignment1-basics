# Old BPE Implementation Code

This directory contains legacy BPE tokenizer implementations and experimental code that were used during the development process.

## Contents

### Legacy Implementations

- **`bpe.py`**: Original BPE implementation before refactoring
- **`tokenizer.py`**: Original tokenizer class implementation
- **`tokenizer_improved.py`**: Improved tokenizer with priority-based greedy merging (used as reference for optimization)
- **`pretokenization_example.py`**: Examples demonstrating pre-tokenization

### Experimental Code (`bpe_creation_code/`)

This directory contains various experimental implementations and optimization attempts:

- **`train_bpe_with_progress.py`**: Training script with detailed progress bars and debug features
  - Used as reference for adding tqdm progress bars to the main implementation
  - Contains both incremental update and full rebuild merge strategies
  
- **`train_bpe_optimized.py`**: Optimized training implementation with various performance improvements

- **`compare_bpe_performance.py`**: Performance comparison scripts

- **`BPE_OPTIMIZATION_README.md`**: Documentation of optimization strategies

- **`tokenizer_experiments/`**: Various experimental encoding approaches
  - `encode_datasets.py`: Basic encoding implementation
  - `encode_datasets_parallel.py`: Parallel encoding experiments
  - `encode_datasets_streaming.py`: Streaming encoding experiments
  - `encode_datasets_simple_parallel.py`: Simplified parallel encoding

- **`learn_bpe/`**: Additional learning resources and experiments

## Current Implementation

The current, production-ready BPE implementation is in the parent directory:

- `../training.py`: BPE training with tqdm progress bars
- `../tokenizer.py`: Optimized tokenizer with priority-based greedy merging
- `../utils.py`: Utility functions for pre-tokenization
- `../__init__.py`: Module exports

## Why This Code Was Preserved

This code is kept for:
1. **Historical reference**: Understanding the evolution of the implementation
2. **Performance comparisons**: Benchmarking against older approaches
3. **Learning**: Demonstrating different implementation strategies
4. **Debugging**: Reference implementations for troubleshooting

## Note

⚠️ **Do not use this code in production.** Use the refactored implementation in the parent `bpe/` directory instead.

