MODEL=OLMo2-7B
TOK_PER_IT=4194304
ITS=(251000 502000 753000 928646)

for IT in "${ITS[@]}"; do
	export REVISION=stage1-step$IT-tokens$(( (TOK_PER_IT*IT + 1000000000) / 1000000000 ))B
	sbatch --job-name eval-$MODEL-$IT scripts/evaluate.sbatch allenai/OLMo-2-1124-7B $IT $TOK_PER_IT $MODEL
done
