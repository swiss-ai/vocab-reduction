#!/bin/bash

# hf_base_runner.sh - Generic script to run evaluation jobs for multiple models
# Usage: hf_base_runner.sh <model_type_description>
# 
# This script expects MODEL_CHECKPOINTS associative array to be defined before calling
# and optionally WANDB_ENTITY, WANDB_PROJECT, and APPLY_CHAT_TEMPLATE environment variables

# Get model type description from argument (for display purposes)
MODEL_TYPE_DESC=${1:-"models"}

# Set default values for optional environment variables
export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals}
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}

# Launch evaluation jobs for each model
echo "Launching evaluation jobs for ${#MODEL_CHECKPOINTS[@]} ${MODEL_TYPE_DESC}..."
echo "WANDB Project: ${WANDB_PROJECT}"
echo "Apply Chat Template: ${APPLY_CHAT_TEMPLATE}"
echo ""

job_count=0
for MODEL in "${!MODEL_CHECKPOINTS[@]}"; do
    CKPT_PATH="${MODEL_CHECKPOINTS[$MODEL]}"
    job_count=$((job_count + 1))
    
    echo "Launching job $job_count/${#MODEL_CHECKPOINTS[@]}: $MODEL"
    echo "  Checkpoint path: $CKPT_PATH"
    
    sbatch --job-name eval-$MODEL scripts/evaluate_hf.sbatch "$CKPT_PATH" "$MODEL"
    
    # Add a small delay between submissions to avoid overwhelming the scheduler
    sleep 1
done
