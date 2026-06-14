MODEL=SmolLM2-1.7B
TOK_PER_IT=2097152
ITS=(125000 1000000 2000000 3000000 4000000 5000000)

for IT in "${ITS[@]}"; do
	export REVISION=step-$IT
	sbatch --job-name eval-$MODEL-$IT scripts/evaluate.sbatch HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints $IT $TOK_PER_IT $MODEL
done
