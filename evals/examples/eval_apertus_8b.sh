export TOKENIZER=alehc/swissai-tokenizer
export BOS=true

MODEL=Apertus-8B
TOK_PER_IT=$(( 4096*1024 ))

ITS=(250000 500000 750000 1000000 1250000 1500000)
CKPT_PATH=/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-8b-128-nodes/checkpoints/
for IT in "${ITS[@]}"; do
	sbatch --job-name eval-$MODEL-$IT scripts/evaluate.sbatch $CKPT_PATH $IT $TOK_PER_IT $MODEL
done
