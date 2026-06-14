#!/bin/bash

# Define MODEL:CKPT_PATH pairs using an associative array
declare -A MODEL_CHECKPOINTS=(
    # EuroLLM models
    ["EuroLLM-1.7B"]="utter-project/EuroLLM-1.7B"
    ["EuroLLM-9B"]="utter-project/EuroLLM-9B"
    ["EuroLLM-22B-Preview"]="utter-project/EuroLLM-22B-Preview"
    
    # OLMo models (base)
    ["OLMo-2-1124-7B"]="allenai/OLMo-2-1124-7B"
    ["OLMo-2-0325-32B"]="allenai/OLMo-2-0325-32B"
    
    # Qwen 2.5 models
    ["Qwen2.5-7B"]="Qwen/Qwen2.5-7B"
)

export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals}

# Base model configurations
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}

# Call the common runner script
source examples/alignment/hf_base_runner.sh "Base models"
