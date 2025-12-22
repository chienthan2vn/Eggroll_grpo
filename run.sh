#!/bin/bash
#
# ES Translation Fine-tuning Launch Script
# 
# Run this script on a multi-GPU machine to train with ES.
#

# Configuration - Edit these paths
MODEL_PATH=""  # Path to your Seq2Seq model
DATA_PATH=""   # Path to your training data JSON

# Training hyperparameters
NUM_GPUS=8
POPULATION_SIZE=64  # Should be divisible by NUM_GPUS
BATCH_SIZE=8
EPOCHS=100
SIGMA=0.001
LR=0.001

# LoRA config
LORA_R=8
LORA_ALPHA=8
TARGET_MODULES="q_proj,k_proj,fc2"

# Output
OUTPUT_DIR="./outputs/$(date +%Y%m%d_%H%M%S)"

# Logging
USE_WANDB=""  # Set to "--wandb" to enable
WANDB_PROJECT="es-translation"

# Run training
echo "======================================"
echo "ES Translation Fine-tuning"
echo "======================================"
echo "GPUs: $NUM_GPUS"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "======================================"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    main.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --target_modules "$TARGET_MODULES" \
    --sigma $SIGMA \
    --lr $LR \
    --population_size $POPULATION_SIZE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --antithetic \
    --rank_transform \
    --eval_every 10 \
    --save_every 50 \
    $USE_WANDB \
    --wandb_project "$WANDB_PROJECT"

echo ""
echo "Training complete!"
echo "Output saved to: $OUTPUT_DIR"
