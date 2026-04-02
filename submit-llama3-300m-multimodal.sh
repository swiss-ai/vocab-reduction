    #!/bin/bash

    #SBATCH --account=infra01
    #SBATCH --time=12:00:00
    #SBATCH --job-name=apertus-1p5-llama300m-mm-scratch
    #SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/Megatron-LM/logs/slurm/training/%x-%j.out
    #SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/Megatron-LM/logs/slurm/training/%x-%j.err
    #SBATCH --nodes=8
    #SBATCH --ntasks-per-node=4
    #SBATCH --cpus-per-task=72
    #SBATCH --reservation=PA-2338-RL
    #SBATCH --exclude=nid006931,nid007062
    #SBATCH --no-requeue

    echo "START TIME: $(date)"

    ############################
    # High-level intent
    #
    # - Small Llama-style model, trained from scratch
    # - New Slurm-native launch style from the Apertus script
    # - Multimodal stage-1 dataset (text / vision / audio)
    # - Multimodal tokenizer from day 1
    # - No base checkpoint loading
    # - Resume only from this run's own checkpoints when called with --resume
    # - Core model and optimizer settings match the original llama300m job
    ############################

    # Container environment used by srun
    CONTAINER_ENV=/capstor/store/cscs/swissai/infra01/containers/ngc_25-12-alps2.toml

    # Stage-1 multimodal data root
    STAGE1_DATA_DIR=/capstor/store/cscs/swissai/infra01/users/xyixuan/apertus-1p5-stage1

    # Megatron source and dataset cache
    MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/apertus/Megatron-LM
    DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache

    # IMPORTANT:
    # Set this to the actual pretrain_gpt.py location in your checkout.
    PRETRAIN_GPT_PY=$MEGATRON_LM_DIR/pretrain_gpt.py

    ############################
    # Parse command-line arguments
    ############################

    RESUME_TRAINING=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --resume)
                echo "---> Resume Training <---"
                RESUME_TRAINING=true
                shift
                ;;
            *)
                echo "Unknown argument: $1"
                echo "Usage: $0 [--resume]"
                exit 1
                ;;
        esac
    done

    ############################
    # Optional job chaining
    ############################

    AUTO_JOB_REQUEUE=false
    if [ "$AUTO_JOB_REQUEUE" = true ]; then
        echo "[$(date)] Submitting follow-up job with singleton dependency and --resume"
        sbatch --dependency=singleton --job-name=$SLURM_JOB_NAME $0 --resume
    fi

    ############################
    # Debug toggles
    ############################

    LOG_NCCL=false
    NSYS_PROFILER=false
    MOCK_DATA=false

    ############################
    # Local cache dirs on compute nodes
    ############################

    TORCH_INDUCTOR_CACHE_DIR=/tmp/.torch_inductor
    TRITON_HOME_DIR=/tmp/.triton
    PYTHON_CACHE_DIR=/tmp/.python_cache



    if [ ! -f "$PRETRAIN_GPT_PY" ]; then
        echo "[$(date)] ERROR: pretrain_gpt.py not found at $PRETRAIN_GPT_PY"
        exit 1
    fi

    ############################
    # Small-model scale
    #
    # Restore original llama300m training scale:
    # - MBS=2
    # - GBS=256
    # - SEQ_LEN=4096
    # - target tokens ~= 8.39B, matching 8000 iters in the original script
    ############################

    MBS=2
    GBS=256
    SEQ_LEN=4096

    # Exact token budget from the original llama300m script:
    # 8000 * 256 * 4096 = 8,388,608,000
    TARGET_TOKENS=8388608000

    CHECKPOINT_STEPS=1000
    TRAINING_STEPS=$((TARGET_TOKENS / (GBS * SEQ_LEN)))
    TRAINING_STEPS=$((((TRAINING_STEPS + CHECKPOINT_STEPS/2) / CHECKPOINT_STEPS) * CHECKPOINT_STEPS))

    echo "[$(date)] TRAINING_STEPS=$TRAINING_STEPS"

    ############################
    # Logging and checkpoint dirs
    ############################

    PROJECT_NAME=main-runs-apertus-1p5-llama300m-scratch
    EXP_NAME=apertus-1p5-llama300m-stage1-mm-scratch
    PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME

    EXP_DIR=$PROJECT_DIR/$EXP_NAME
    CKPT_DIR=$EXP_DIR/checkpoints
    DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
    LOGGING_DIR=$EXP_DIR/logging
    TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

    mkdir -p $PROJECT_DIR
    mkdir -p $EXP_DIR
    mkdir -p $CKPT_DIR
    mkdir -p $DEBUG_DIR
    mkdir -p $LOGGING_DIR
    mkdir -p $TENSORBOARD_DIR

    ############################
    # Environment setup
    ############################




    export WANDB__FILE_STREAM_RETRY_MAX=10
    export HF_HUB_OFFLINE=1

    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

    export TRITON_HOME=$TRITON_HOME_DIR
    export TRITON_CACHE_DIR=$TRITON_HOME_DIR/cache

    export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

    # Slurm-native distributed environment
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=8888
    export WORLD_SIZE=$SLURM_NPROCS

    echo "[$(date)] MASTER_ADDR=$MASTER_ADDR"
    echo "[$(date)] MASTER_PORT=$MASTER_PORT"
    echo "[$(date)] WORLD_SIZE=$WORLD_SIZE"


    ############################
    # Dataset manifest
    #
    # Fresh run:
    # - discover valid dataset prefixes
    # - save exact prefix list to a manifest file
    #
    # Resume run:
    # - reuse the exact same prefix list from the manifest
    ############################

    METADATA_DIR=$EXP_DIR/metadata
    DATA_MANIFEST=$METADATA_DIR/data_path_list.txt
    RUN_METADATA=$METADATA_DIR/run_metadata.txt

    mkdir -p "$METADATA_DIR"

    if [ "$RESUME_TRAINING" = true ]; then
        if [ ! -f "$DATA_MANIFEST" ]; then
            echo "[$(date)] ERROR: Resume requested but dataset manifest not found: $DATA_MANIFEST"
            exit 1
        fi

        mapfile -t DATA_PATH_LIST < "$DATA_MANIFEST"

        if [ ${#DATA_PATH_LIST[@]} -eq 0 ]; then
            echo "[$(date)] ERROR: Dataset manifest exists but is empty: $DATA_MANIFEST"
            exit 1
        fi

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

        cat > "$RUN_METADATA" <<EOF
    created_at=$(date)
    stage1_data_dir=$STAGE1_DATA_DIR
    dataset_manifest=$DATA_MANIFEST
    num_dataset_prefixes=${#DATA_PATH_LIST[@]}
    tokenizer_model=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok
    seq_len=$SEQ_LEN
    mbs=$MBS
    gbs=$GBS
    target_tokens=$TARGET_TOKENS
    training_steps=$TRAINING_STEPS
    EOF

        echo "[$(date)] FRESH MODE: discovered and saved ${#DATA_PATH_LIST[@]} dataset prefixes"
        echo "[$(date)] Dataset manifest saved to $DATA_MANIFEST"
    fi


    ############################
    # Megatron args
    #
    # These match the original llama300m choices.
    ############################

    TRANSFORMER_ENGINE_ARGS=(
        --transformer-impl transformer_engine
        --use-precision-aware-optimizer
        --main-grads-dtype bf16
        --log-params-norm
    )

    NETWORK_SIZE_ARGS=(
        --num-layers 6
        --hidden-size 2048
        --ffn-hidden-size 6144
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
        --swiglu
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
        --eval-interval 100000000000
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

    MIXED_PRECISION_ARGS=(
        --bf16
    )

    ############################
    # Distributed args
    ############################

    DISTRIBUTED_ARGS=(
        --tensor-model-parallel-size 1
        --pipeline-model-parallel-size 1
        --context-parallel-size 1
        --wgrad-deferral-limit 50
        --use-distributed-optimizer
        --overlap-grad-reduce
        --overlap-param-gather
    )

    ############################
    # Tokenizer
    ############################

    TOKENIZER_ARGS=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok
    )

    ############################
    # Data args
    #
    # Keep the multimodal packed-seq / cross-document settings from the newer job.
    ############################

    DATA_ARGS=(
        --split 100,0,0
        --seq-length $SEQ_LEN
        --reset-position-ids
        --use-packed-seq-params
        --no-create-attention-mask-in-dataloader
        --eod-mask-loss
        --num-workers 64
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

    ############################
    # Checkpointing
    ############################

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
        echo "[$(date)] RESUME MODE: loading from $CKPT_DIR"
    else
        echo "[$(date)] FRESH MODE: training from scratch"
    fi

    ############################
    # Assemble command
    ############################

    TRAINING_CMD=$(cat <<EOF
    python3 $PRETRAIN_GPT_PY \
    ${TRANSFORMER_ENGINE_ARGS[*]} \
    ${NETWORK_SIZE_ARGS[*]} \
    ${LOGGING_ARGS[*]} \
    ${REGULARIZATION_ARGS[*]} \
    ${TRAINING_ARGS[*]} \
    ${INITIALIZATION_ARGS[*]} \
    ${LEARNING_RATE_ARGS[*]} \
    ${CHECKPOINTING_ARGS[*]} \
    ${MIXED_PRECISION_ARGS[*]} \
    ${DISTRIBUTED_ARGS[*]} \
    ${TOKENIZER_ARGS[*]} \
    ${DATA_ARGS[*]}
    EOF
    )

    ############################
    # WANDB
    ############################

    if [ -n "$WANDB_API_KEY" ]; then
        echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
        TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOGGING_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
    else
        export WANDB_MODE=disabled
        echo "[$(date)] No WANDB API key found. WANDB logging disabled."
    fi

    ############################
    # Optional debug prefixes
    ############################

    CMD_PREFIX=""

    if [ "$LOG_NCCL" = true ]; then
        CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
    fi

    if [ "$NSYS_PROFILER" = true ]; then
        NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace-hostname-\$SLURMD_NODENAME-procid-\$SLURM_PROCID.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
        TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
    fi

    ############################
    # Launch
    ############################

    srun \
        --cpus-per-task $SLURM_CPUS_PER_TASK \
        --mpi=pmix \
        --environment=$CONTAINER_ENV \
        --network=disable_rdzv_get \
        -lu \
        bash -c "
        mkdir -p $TORCH_INDUCTOR_CACHE_DIR
        mkdir -p $TRITON_HOME_DIR
        mkdir -p $TRITON_CACHE_DIR
        mkdir -p $PYTHON_CACHE_DIR
        RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX $TRAINING_CMD
        "

    echo "END TIME: $(date)"