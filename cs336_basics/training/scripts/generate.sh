#!/bin/bash
# Text generation script for Transformer Language Models
# This script loads a trained checkpoint and generates text using various decoding strategies

set -e  # Exit on error

# ============================================================================
# TOKENIZER PATHS
# ============================================================================
# Default paths (can be overridden by environment variables)
VOCAB_FILE="${VOCAB_FILE:-./data/tokenizers/tinystories_vocab.json}"
MERGES_FILE="${MERGES_FILE:-./data/tokenizers/tinystories_merges.txt}"

# ============================================================================
# CHECKPOINT PATH
# ============================================================================
# Checkpoint path (can be overridden by command line argument or environment variable)
CHECKPOINT="${CHECKPOINT:-./checkpoints/tinystories/lr_test/lr3e-3_bs64_layers4_heads16_d512_warmup2000_beta0.9-0.999_wd0p01/checkpoint_final.pt}"

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# Model hyperparameters (needed to reconstruct model from checkpoint)
# These should match the training configuration
VOCAB_SIZE="${VOCAB_SIZE:-10000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
D_MODEL="${D_MODEL:-512}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-16}"
D_FF="${D_FF:-1344}"
USE_ROPE="${USE_ROPE:-true}"

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================
PROMPT="${PROMPT:-Once upon a time}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1000}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.92}"
EOS_TOKEN="${EOS_TOKEN:-<|endoftext|>}"

# ============================================================================
# OTHER OPTIONS
# ============================================================================
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
VERBOSE="${VERBOSE:-false}"

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --vocab)
            VOCAB_FILE="$2"
            shift 2
            ;;
        --merges)
            MERGES_FILE="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --vocab_size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        --context_length)
            CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --d_model)
            D_MODEL="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --num_heads)
            NUM_HEADS="$2"
            shift 2
            ;;
        --d_ff)
            D_FF="$2"
            shift 2
            ;;
        --no_rope)
            USE_ROPE="false"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --eos_token)
            EOS_TOKEN="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --checkpoint PATH    Path to model checkpoint file (.pt)"
            echo ""
            echo "Optional:"
            echo "  --vocab PATH         Path to vocabulary JSON file (default: $VOCAB_FILE)"
            echo "  --merges PATH        Path to BPE merges text file (default: $MERGES_FILE)"
            echo "  --prompt TEXT        Input prompt text (default: \"$PROMPT\")"
            echo "  --max_new_tokens N   Maximum tokens to generate (default: $MAX_NEW_TOKENS)"
            echo "  --temperature T      Sampling temperature (default: $TEMPERATURE)"
            echo "  --top_p P           Top-p sampling threshold (default: $TOP_P)"
            echo "  --vocab_size N      Vocabulary size (default: $VOCAB_SIZE)"
            echo "  --context_length N  Context length (default: $CONTEXT_LENGTH)"
            echo "  --d_model N         Model dimension (default: $D_MODEL)"
            echo "  --num_layers N      Number of layers (default: $NUM_LAYERS)"
            echo "  --num_heads N       Number of heads (default: $NUM_HEADS)"
            echo "  --d_ff N            Feed-forward dimension (default: $D_FF)"
            echo "  --no_rope           Disable RoPE positional encoding"
            echo "  --device DEVICE     Device to use (default: $DEVICE)"
            echo "  --seed N            Random seed for reproducibility"
            echo "  --eos_token TOKEN   End-of-sequence token (default: $EOS_TOKEN)"
            echo "  --verbose           Print detailed information"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic generation"
            echo "  $0 --checkpoint ./checkpoints/model.pt --prompt \"Once upon a time\""
            echo ""
            echo "  # Creative generation with high temperature"
            echo "  $0 --checkpoint ./checkpoints/model.pt --prompt \"The robot\" --temperature 1.5 --max_new_tokens 200"
            echo ""
            echo "  # Deterministic generation (low temperature)"
            echo "  $0 --checkpoint ./checkpoints/model.pt --prompt \"Once upon\" --temperature 0.1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# VALIDATION
# ============================================================================
# Check if checkpoint is provided
if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if checkpoint file exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Check if vocab file exists
if [ ! -f "$VOCAB_FILE" ]; then
    echo "Error: Vocabulary file not found: $VOCAB_FILE"
    echo "Please set VOCAB_FILE environment variable or use --vocab option"
    exit 1
fi

# Check if merges file exists
if [ ! -f "$MERGES_FILE" ]; then
    echo "Error: Merges file not found: $MERGES_FILE"
    echo "Please set MERGES_FILE environment variable or use --merges option"
    exit 1
fi

# Validate generation parameters
# Check temperature (using awk for floating point comparison)
if awk "BEGIN {exit !($TEMPERATURE <= 0)}"; then
    echo "Error: temperature must be positive"
    exit 1
fi

# Check top_p (using awk for floating point comparison)
if awk "BEGIN {exit !($TOP_P <= 0 || $TOP_P > 1.0)}"; then
    echo "Error: top_p must be in (0, 1]"
    exit 1
fi

if [ "$MAX_NEW_TOKENS" -le 0 ]; then
    echo "Error: max_new_tokens must be positive"
    exit 1
fi

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================
echo "================================================================================"
echo "Text Generation Configuration"
echo "================================================================================"
echo ""
echo "CHECKPOINT:"
echo "  $CHECKPOINT"
echo ""
echo "TOKENIZER:"
echo "  vocab: $VOCAB_FILE"
echo "  merges: $MERGES_FILE"
echo "  eos_token: $EOS_TOKEN"
echo ""
echo "MODEL ARCHITECTURE:"
echo "  vocab_size=$VOCAB_SIZE, context_length=$CONTEXT_LENGTH"
echo "  d_model=$D_MODEL, num_layers=$NUM_LAYERS, num_heads=$NUM_HEADS"
echo "  d_ff=$D_FF, use_rope=$USE_ROPE"
echo ""
echo "GENERATION:"
echo "  prompt: \"$PROMPT\""
echo "  max_new_tokens=$MAX_NEW_TOKENS"
echo "  temperature=$TEMPERATURE"
echo "  top_p=$TOP_P"
echo ""
echo "DEVICE: $DEVICE"
if [ -n "$SEED" ]; then
    echo "SEED: $SEED"
fi
echo "================================================================================"
echo ""

# ============================================================================
# BUILD COMMAND
# ============================================================================
CMD="uv run python -m cs336_basics.training.generate"

# Required arguments
CMD="$CMD --checkpoint \"$CHECKPOINT\""
CMD="$CMD --vocab \"$VOCAB_FILE\""
CMD="$CMD --merges \"$MERGES_FILE\""
CMD="$CMD --prompt \"$PROMPT\""

# Model architecture arguments
CMD="$CMD --vocab_size $VOCAB_SIZE"
CMD="$CMD --context_length $CONTEXT_LENGTH"
CMD="$CMD --d_model $D_MODEL"
CMD="$CMD --num_layers $NUM_LAYERS"
CMD="$CMD --num_heads $NUM_HEADS"
CMD="$CMD --d_ff $D_FF"
if [ "$USE_ROPE" = "true" ]; then
    CMD="$CMD --use_rope"
fi

# Generation arguments
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top_p $TOP_P"
CMD="$CMD --eos_token \"$EOS_TOKEN\""

# Device
CMD="$CMD --device $DEVICE"

# Optional arguments
if [ -n "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi

if [ "$VERBOSE" = "true" ]; then
    CMD="$CMD --verbose"
fi

# ============================================================================
# EXECUTE
# ============================================================================
echo "Generating text..."
echo "Command: $CMD"
echo ""

eval $CMD

