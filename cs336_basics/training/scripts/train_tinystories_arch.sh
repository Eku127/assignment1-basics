#!/bin/bash
# Training script for TinyStories dataset
# This script configures all hyperparameters explicitly and creates checkpoint directories
# with names that include key training parameters.

set -e  # Exit on error

# ============================================================================
# DATA PATHS
# ============================================================================
# Default paths (can be overridden by environment variables)
# Modify these paths according to your data location
TRAIN_DATA="${TRAIN_DATA:-./data/encoded/tinystories_train.npy}"
VAL_DATA="${VAL_DATA:-./data/encoded/tinystories_valid.npy}"

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
VOCAB_SIZE=10000
CONTEXT_LENGTH=256
D_MODEL=512
NUM_LAYERS=4
NUM_HEADS=16
D_FF=1344
USE_ROPE=true

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
# 64, 128, 256, 512, 1024
BATCH_SIZE="${BATCH_SIZE:-64}"

# Calculate MAX_STEPS to maintain total tokens = 327,680,000
# Formula: batch_size × max_steps × context_length = 327,680,000
MAX_STEPS="${MAX_STEPS:-$((327680000 / (BATCH_SIZE * CONTEXT_LENGTH)))}"


LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MIN_LR="${MIN_LR:-0.00001}"
# Warmup steps: 10% of total training steps
WARMUP_STEPS="${WARMUP_STEPS:-$((MAX_STEPS / 10))}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.999}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP=1.0

# ============================================================================
# LOGGING AND CHECKPOINTING
# ============================================================================
LOG_EVERY=100
VAL_EVERY=1000
SAVE_EVERY=5000
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-./checkpoints/tinystories/bs_test}"

# Create checkpoint directory name with key parameters
# Format: lr{lr}_bs{bs}_layers{layers}_heads{heads}_d{dmodel}_warmup{warmup}_beta{beta1}-{beta2}_wd{weightdecay}
# Convert learning rate to a filesystem-friendly string (e.g., 3e-4 -> 3e-4, 0.0003 -> 0p0003)
LR_STR=$(echo "$LEARNING_RATE" | sed 's/^0\./0p/g' | sed 's/\./p/g')
BETA_STR="${BETA1}-${BETA2}"
WD_STR=$(echo "$WEIGHT_DECAY" | sed 's/^0\./0p/g' | sed 's/\./p/g')
CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/lr${LR_STR}_bs${BATCH_SIZE}_layers${NUM_LAYERS}_heads${NUM_HEADS}_d${D_MODEL}_warmup${WARMUP_STEPS}_beta${BETA_STR}_wd${WD_STR}"

# ============================================================================
# WANDB (OPTIONAL)
# ============================================================================
USE_WANDB="${USE_WANDB:-true}"

WANDB_PROJECT="${WANDB_PROJECT:-CS336_TinyStories_BS_Test}"
WANDB_NAME="${WANDB_NAME:-lr${LR_STR}_bs${BATCH_SIZE}_layers${NUM_LAYERS}_heads${NUM_HEADS}_d${D_MODEL}_warmup${WARMUP_STEPS}_beta${BETA_STR}_wd${WD_STR}}"

# ============================================================================
# DEVICE
# ============================================================================
DEVICE="${DEVICE:-cuda:4}"

# ============================================================================
# VALIDATION
# ============================================================================
# Check if data files exist
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data file not found: $TRAIN_DATA"
    echo "Please set TRAIN_DATA environment variable or modify the default path in the script"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "Error: Validation data file not found: $VAL_DATA"
    echo "Please set VAL_DATA environment variable or modify the default path in the script"
    exit 1
fi

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================
echo "================================================================================"
echo "TinyStories Training Configuration"
echo "================================================================================"
echo ""
echo "DATA:"
echo "  Train: $TRAIN_DATA"
echo "  Val:   $VAL_DATA"
echo ""
echo "MODEL:"
echo "  vocab_size=$VOCAB_SIZE, context_length=$CONTEXT_LENGTH"
echo "  d_model=$D_MODEL, num_layers=$NUM_LAYERS, num_heads=$NUM_HEADS"
echo "  d_ff=$D_FF, use_rope=$USE_ROPE"
echo ""
echo "TRAINING:"
echo "  batch_size=$BATCH_SIZE, max_steps=$MAX_STEPS"
TOTAL_TOKENS=$((BATCH_SIZE * MAX_STEPS * CONTEXT_LENGTH))
echo "  Total tokens: $TOTAL_TOKENS (target: 327,680,000)"
echo ""
echo "OPTIMIZER (AdamW):"
echo "  learning_rate=$LEARNING_RATE, min_lr=$MIN_LR"
echo "  warmup_steps=$WARMUP_STEPS"
echo "  betas=($BETA1, $BETA2), weight_decay=$WEIGHT_DECAY"
echo "  grad_clip=$GRAD_CLIP"
echo ""
echo "LOGGING:"
echo "  log_every=$LOG_EVERY, val_every=$VAL_EVERY, save_every=$SAVE_EVERY"
echo "  checkpoint_dir=$CHECKPOINT_DIR"
if [ "$USE_WANDB" = "true" ]; then
    echo "  wandb_project=$WANDB_PROJECT, wandb_name=$WANDB_NAME"
fi
echo ""
echo "DEVICE: $DEVICE"
echo "================================================================================"
echo ""

# ============================================================================
# BUILD COMMAND
# ============================================================================
CMD="python -m cs336_basics.training.train"

# Data arguments
CMD="$CMD --train_data \"$TRAIN_DATA\""
CMD="$CMD --val_data \"$VAL_DATA\""

# Model arguments
CMD="$CMD --vocab_size $VOCAB_SIZE"
CMD="$CMD --context_length $CONTEXT_LENGTH"
CMD="$CMD --d_model $D_MODEL"
CMD="$CMD --num_layers $NUM_LAYERS"
CMD="$CMD --num_heads $NUM_HEADS"
CMD="$CMD --d_ff $D_FF"
if [ "$USE_ROPE" = "true" ]; then
    CMD="$CMD --use_rope"
fi

# Training arguments
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --max_steps $MAX_STEPS"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --min_lr $MIN_LR"
CMD="$CMD --warmup_steps $WARMUP_STEPS"
CMD="$CMD --beta1 $BETA1"
CMD="$CMD --beta2 $BETA2"
CMD="$CMD --weight_decay $WEIGHT_DECAY"
CMD="$CMD --grad_clip $GRAD_CLIP"

# Logging arguments
CMD="$CMD --log_every $LOG_EVERY"
CMD="$CMD --val_every $VAL_EVERY"
CMD="$CMD --save_every $SAVE_EVERY"
CMD="$CMD --checkpoint_dir \"$CHECKPOINT_DIR\""

# Wandb arguments
if [ "$USE_WANDB" = "true" ]; then
    CMD="$CMD --use_wandb"
    CMD="$CMD --wandb_project \"$WANDB_PROJECT\""
    if [ -n "$WANDB_NAME" ]; then
        CMD="$CMD --wandb_name \"$WANDB_NAME\""
    fi
fi

# Device
CMD="$CMD --device $DEVICE"

# ============================================================================
# EXECUTE
# ============================================================================
echo "Starting training..."
echo "Command: $CMD"
echo ""

eval $CMD

