#!/bin/bash

# Define MODEL:CKPT_PATH pairs using an associative array
declare -A MODEL_CHECKPOINTS=(
    # RLVR models
    # ["Apertus3-8B-sft-RLVR-105"]="/capstor/store/cscs/swissai/infra01/reasoning/models/global_step_105/hf_actor"
    # ["Apertus3-8B-sft-RLVR-450"]="/capstor/store/cscs/swissai/infra01/reasoning/models/global_step_450/hf_actor"
    # ["Apertus3-8B-sft-RLVR-560"]="/capstor/store/cscs/swissai/infra01/reasoning/models/global_step_560/hf_actor"
    # ["Apertus3-8B-sft-RLVR-MR-800"]="/capstor/store/cscs/swissai/infra01/reasoning/models/mr_800/hf_actor"
    ["Apertus3-8B-sft-RLVR-2STG-2000"]="/capstor/store/cscs/swissai/infra01/reasoning/models/2stg_2000/hf_actor"
    
    
    # ["Apertus-8B-7.04T--tulu_special_token"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/chat-template/Apertus8B-tokens7.04T-it1678000-tulu_special_token-swissai-tulu-3-sft-0225/checkpoints/9b811fb20bdd09a4/checkpoint-13446"
    # ["Apertus-8B-7.04T-tulu"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/chat-template/Apertus8B-tokens7.04T-it1678000-tulu-swissai-tulu-3-sft-0225/checkpoints/6d5f11d2873ecb4d/checkpoint-13446"
    # SFT models
    ["Apertus-8B-SFT"]="/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus3-8B_iter_1678000-tulu3-sft/checkpoint-13446/"
    ["Apertus8B-tokens7.2T-ademamix-grad_norm_0.1"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/8b-cooldown-bug-experiments/Apertus8B-tokens7.4T-it1728000-ademamix-swissai-tulu-3-sft-0225-max_grad_norm_0.1/checkpoints/206c53edb3c43a3a/checkpoint-13000/"
    ["Apertus-70B-15T-it1155828--Tulu3-SFT"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/token-count/Apertus70B-tokens15T-it1155828-ademamix-swissai-tulu-3-sft-0225/checkpoints/c9b2910640c220b1/checkpoint-13446"

    ["Apertus70B-tokens15T-mixture1"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-1/checkpoints/53d4338a0a4a4834/checkpoint-12976"
    ["Apertus8B-tokens7.2T-mixture1-fast"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/dataset-mixtures-fast/Apertus8B-tokens7.2T-it1728000-ademamix-apertus-sft-mixture-1/checkpoints/55212b68b8cb44a9/checkpoint-1622"
    ["Apertus70B-tokens15T-mixture1-fast"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures-fast/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-1/checkpoints/30f93081a9efa4fa/checkpoint-1622"
    
    ["Apertus70B-tokens15T-mixture2"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-2/checkpoints/03e6f5ffc256bd37/checkpoint-31312"
    ["Apertus70B-tokens15T-mixture2-fast"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures-fast/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-2/checkpoints/d1efecec81835dd2/checkpoint-3914"

    ["Apertus70B-tokens15T-mixture3"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-3/checkpoints/e5c6472115344007/checkpoint-36620"
    ["Apertus70B-tokens15T-mixture3-fast"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures-fast/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-3/checkpoints/416753d72bb96b59/checkpoint-4577"


    ["Apertus8B-tokens7.2T-mixture4-fast"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/dataset-mixtures-fast/Apertus8B-tokens7.2T-it1728000-ademamix-apertus-sft-mixture-4/checkpoints/754c761b6c8b6898/checkpoint-4793"
    ["Apertus70B-tokens15T-mixture4-fast"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures-fast/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-4/checkpoints/7f9593ea352b3b54/checkpoint-4792"
)

export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals}

# SFT model configurations
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}

# Call the common runner script
source examples/alignment/hf_base_runner.sh "Apertus SFT models"
