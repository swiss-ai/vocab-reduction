export TOKENIZER=alehc/swissai-tokenizer
export BOS=true
export SIZE=70
export TASKS=$(uv run src/evals/get_tasks.py)
export EXTRA_PIPS="nvidia-modelopt==0.27.0"

MODEL=Apertus-${SIZE}B
TOK_PER_IT=$((4096*2048)):523519,$((4096*4096)):

# Newer checkpoints.
#ITS=(800000 740000 680000 620000 560000 480000)
ITS=(1155828)
CKPT_PATH=/capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints-512-noOverlap/
for IT in "${ITS[@]}"; do
	sbatch --job-name eval-$MODEL-$IT scripts/evaluate.sbatch $CKPT_PATH $IT $TOK_PER_IT $MODEL
done

# Not that recent checkpoints.
#ITS=(360000)
ITS=()
CKPT_PATH=/capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints/
for IT in "${ITS[@]}"; do
	export TASKS=english
	sbatch --job-name eval-$MODEL-$TASKS-$IT scripts/evaluate.sbatch $CKPT_PATH $IT $TOK_PER_IT $MODEL
	export TASKS=multilingual
	sbatch --job-name eval-$MODEL-$TASKS-$IT scripts/evaluate.sbatch $CKPT_PATH $IT $TOK_PER_IT $MODEL
done

# Very old checkpoints.
#ITS=(240000 120000)
ITS=()
CKPT_PATH=/capstor/scratch/cscs/schlag/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints/
for IT in "${ITS[@]}"; do
	export TASKS=english
	sbatch --job-name eval-$MODEL-$TASKS-$IT scripts/evaluate.sbatch $CKPT_PATH $IT $TOK_PER_IT $MODEL
	export TASKS=multilingual
	sbatch --job-name eval-$MODEL-$TASKS-$IT scripts/evaluate.sbatch $CKPT_PATH $IT $TOK_PER_IT $MODEL
done
