#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=11:59:00
#SBATCH --job-name=apertus-300m-mm-ablation
#SBATCH --output=/iopsstor/scratch/cscs/%u/main_run_megatron/Megatron-LM/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/main_run_megatron/Megatron-LM/logs/slurm/training/%x-%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --signal=SIGUSR2@600    # Send SIGUSR2 600 seconds before hitting the time limit
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --no-requeue    # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

################ Configs ################
# Stage-1 multimodal data root
STAGE1_DATA_DIR=/capstor/store/cscs/swissai/infra01/users/xyixuan/apertus-1p5-stage1

# Tokens per batch: 256 * 4096 = 1,048,576
MBS=2 # Micro batch size
GBS=256 # Global batch size
SEQ_LEN=4096 # Sequence length

# 8000 iters * 1M tokens ~= 8.39B tokens total
TRAINING_STEPS=8000
CHECKPOINT_STEPS=1000

RESUME_TRAINING=false
AUTO_JOB_REQUEUE=false

#### Debugging ####
LOG_NCCL=false
NSYS_PROFILER=false
MOCK_DATA=false
###################

# Container environment used by srun
CONTAINER_ENV=/capstor/store/cscs/swissai/infra01/containers/ngc_25-12-alps2.toml

# Megatron source and dataset cache
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache
PRETRAIN_GPT_PY=$MEGATRON_LM_DIR/pretrain_gpt.py
BACKUP_CODEBASE=false 

# Logging directories & artifacts
PROJECT_NAME=apertus-300m
EXP_NAME=apertus-300m-vanilla-$SLURM_NNODES-nodes
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME

#########################################

EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
METADATA_DIR=$EXP_DIR/metadata
DATA_MANIFEST=$METADATA_DIR/data_path_list.txt

# Set up ENV
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))

export WANDB__FILE_STREAM_RETRY_MAX=10
export HF_HUB_OFFLINE=1

TORCH_INDUCTOR_CACHE_DIR=/tmp/.torch_inductor
TRITON_HOME_DIR=/tmp/.triton
PYTHON_CACHE_DIR=/tmp/.python_cache

export TRITON_HOME=$TRITON_HOME_DIR
export TRITON_CACHE_DIR=$TRITON_HOME_DIR/cache

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8972
export WORLD_SIZE=$SLURM_NPROCS

ulimit -c 0

# Set up directories
mkdir -p $CKPT_DIR
mkdir -p $PROJECT_DIR
mkdir -p $DEBUG_DIR
mkdir -p $LOGGING_DIR
mkdir -p $METADATA_DIR
mkdir -p $TENSORBOARD_DIR

cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

############################
# Dataset manifest (Multimodal)
############################

if [ "$RESUME_TRAINING" = true ]; then
    if [ ! -f "$DATA_MANIFEST" ]; then
        echo "[$(date)] ERROR: Resume requested but dataset manifest not found: $DATA_MANIFEST"
        exit 1
    fi
    mapfile -t DATA_PATH_LIST < "$DATA_MANIFEST"
    echo "[$(date)] RESUME MODE: loaded ${#DATA_PATH_LIST[@]} dataset prefixes from manifest"
else
    mapfile -t DATA_PATH_LIST < <(
        find -L \
            "$STAGE1_DATA_DIR/text" \
            "$STAGE1_DATA_DIR/vision" \
            "$STAGE1_DATA_DIR/audio" \
            -name '*.bin' |
        sed 's/\.bin$//' |
        sort |
        while read -r p; do
            if [ -e "${p}.bin" ] && [ -e "${p}.idx" ]; then
                printf '%s\n' "$p"
            else
                echo "[$(date)] Skipping broken dataset prefix: $p" >&2
            fi
        done
    )

    if [ ${#DATA_PATH_LIST[@]} -eq 0 ]; then
        echo "[$(date)] ERROR: No valid dataset prefixes found under $STAGE1_DATA_DIR"
        exit 1
    fi

    printf '%s\n' "${DATA_PATH_LIST[@]}" > "$DATA_MANIFEST"
    echo "[$(date)] FRESH MODE: discovered and saved ${#DATA_PATH_LIST[@]} dataset prefixes"
fi

#### Megatron Args #### 

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --log-params-norm
)

# 300M Architecture with xIELU
NETWORK_SIZE_ARGS=(
        --num-layers 12
        --hidden-size 1024
        --ffn-hidden-size 4096 
        --num-attention-heads 16 
        --group-query-attention
        --num-query-groups 4 
        --max-position-embeddings $SEQ_LEN
        --position-embedding-type rope
        --rotary-base 500000
        --use-rope-scaling
        --rope-scaling-factor 32
        --make-vocab-size-divisible-by 128
        --normalization RMSNorm
        --xielu
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --tensorboard-dir $TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --weight-decay 0.1
        --clip-grad 1.0
        --adam-beta1 0.9
        --adam-beta2 0.95
)

TRAINING_ARGS=(
        --micro-batch-size $MBS
        --global-batch-size $GBS
        --no-check-for-nan-in-loss-and-grad
        --train-iters $TRAINING_STEPS
        --log-interval 1
        --cross-entropy-loss-fusion
        --disable-bias-linear
        --optimizer adam
        --dataloader-type single
        --manual-gc
        --manual-gc-interval 5000
        --exit-signal-handler
        --eval-interval 10000000
        --eval-iters 0
)

INITIALIZATION_ARGS=(
        --seed 28
        --init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
        --lr 0.0003
        --min-lr 0.00003
        --lr-decay-style cosine
        --lr-warmup-iters 500
)

CHECKPOINTING_ARGS=(
        --save $CKPT_DIR
        --save-interval $CHECKPOINT_STEPS
        --ckpt-format torch_dist
        --async-save
)

if [ "$RESUME_TRAINING" = true ]; then
    CHECKPOINTING_ARGS+=(
        --load $CKPT_DIR
        --ckpt-fully-parallel-load
        --dist-ckpt-strictness assume_ok_unexpected
    )
fi

MIXED_PRECISION_ARGS=(
        --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --wgrad-deferral-limit 50
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# Multimodal Tokenizer
TOKENIZER_ARGS=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok
)

# Multimodal Data Args (Packed seqs, EOD loss masking)
DATA_ARGS=(
        --split 100,0,0
        --seq-length $SEQ_LEN
        --reset-position-ids
        --use-packed-seq-params
        --no-create-attention-mask-in-dataloader
        --eod-mask-loss
        --num-workers 2
        --num-dataset-builder-threads 4
        --goldfish-loss
        --goldfish-k 50
        --goldfish-h 50
)

if [ "$MOCK_DATA" = true ]; then
        DATA_ARGS+=(--mock-data)
else
        DATA_ARGS+=(--data-path "${DATA_PATH_LIST[@]}" --data-cache-path "$DATASET_CACHE_DIR")
fi

TRAINING_CMD="python3 $PRETRAIN_GPT_PY \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

# WANDB Logging
if [ -n "$WANDB_API_KEY" ]; then
  export WANDB_ENTITY="faruk-zahiragic-epfl"
  TRAINING_CMD="$TRAINING_CMD \
    --wandb-save-dir $LOGGING_DIR \
    --wandb-project $PROJECT_NAME \
    --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
  export WANDB_MODE=disabled
fi

CMD_PREFIX=""
if [ "$LOG_NCCL" = true ]; then
  CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

if [ "$NSYS_PROFILER" = true ]; then
    NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace-hostname-\$SLURMD_NODENAME-procid-\$SLURM_PROCID.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
fi

# Launch
srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    --network=disable_rdzv_get \
    -lu \
    bash -c "
    mkdir -p $TORCH_INDUCTOR_CACHE_DIR
    mkdir -p $TRITON_HOME_DIR
    mkdir -p $PYTHON_CACHE_DIR
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX $TRAINING_CMD
    "

echo "END TIME: $(date)"