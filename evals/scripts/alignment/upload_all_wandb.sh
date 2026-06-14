#!/bin/bash

# Script to upload all model evaluation results to W&B
# Uses the same entity and project defaults as evaluate_hf.sbatch

# Set default values (same as evaluate_hf.sbatch)
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals-v0.0.1}
LOGS_ROOT=${LOGS_ROOT:-/iopsstor/scratch/cscs/ihakimi/eval-logs}

echo "Uploading all model results to W&B..."
echo "Entity: $WANDB_ENTITY"
echo "Project: $WANDB_PROJECT"
echo "Logs root: $LOGS_ROOT"
echo ""

# Run the Python script
python -m scripts.alignment.update_wandb_all_models \
    --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT" \
    --logs_root "$LOGS_ROOT"
