#!/bin/bash

# Define MODEL:CKPT_PATH pairs using an associative array
declare -A MODEL_CHECKPOINTS=(
    # EuroLLM models
    ["EuroLLM-1.7B-Instruct"]="utter-project/EuroLLM-1.7B-Instruct"
    ["EuroLLM-9B-Instruct"]="utter-project/EuroLLM-9B-Instruct"
    ["EuroLLM-22B-Instruct-Preview"]="utter-project/EuroLLM-22B-Instruct-Preview"
    
    # Gemma models
    ["gemma-3-4b-it"]="google/gemma-3-4b-it"
    ["gemma-3-12b-it"]="google/gemma-3-12b-it"
    ["gemma-3-27b-it"]="google/gemma-3-27b-it"
    
    # K2 models
    ["K2-Chat"]="K2-Chat"

    # Llama models
    ["Llama-3.1-8B-Instruct"]="meta-llama/Llama-3.1-8B-Instruct"
    ["Llama-3.3-70B-Instruct"]="meta-llama/Llama-3.3-70B-Instruct"
    
    # OLMo models (fine-tuned variants)
    ["OLMo-2-1124-7B-SFT"]="allenai/OLMo-2-1124-7B-SFT"
    ["OLMo-2-1124-7B-DPO"]="allenai/OLMo-2-1124-7B-DPO"
    ["OLMo-2-1124-7B-Instruct"]="allenai/OLMo-2-1124-7B-Instruct"
    ["OLMo-2-0325-32B-SFT"]="allenai/OLMo-2-0325-32B-SFT"
    ["OLMo-2-0325-32B-DPO"]="allenai/OLMo-2-0325-32B-DPO"
    ["OLMo-2-0325-32B-Instruct"]="allenai/OLMo-2-0325-32B-Instruct"
    
    # Qwen 2.5 models
    ["Qwen2.5-7B"]="Qwen/Qwen2.5-7B"
    ["Qwen2.5-7B-Instruct"]="Qwen/Qwen2.5-7B-Instruct"
    ["Qwen2.5-14B-Instruct"]="Qwen/Qwen2.5-14B-Instruct"
    ["Qwen2.5-32B-Instruct"]="Qwen/Qwen2.5-32B-Instruct"
    ["Qwen2.5-72B-Instruct"]="Qwen/Qwen2.5-72B-Instruct"
    
    # Qwen 3 models
    ["Qwen3-1.7B"]="Qwen/Qwen3-1.7B"
    ["Qwen3-4B"]="Qwen/Qwen3-4B"
    ["Qwen3-8B"]="Qwen/Qwen3-8B"
    ["Qwen3-14B"]="Qwen/Qwen3-14B"
    ["Qwen3-32B"]="Qwen/Qwen3-32B"

    # SmolLM models
    ["SmolLM3-3B"]="HuggingFaceTB/SmolLM3-3B"
)

export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals}

# SFT model configurations
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}

# Call the common runner script
source examples/alignment/hf_base_runner.sh "SFT models"
