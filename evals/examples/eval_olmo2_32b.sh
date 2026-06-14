export SIZE=32

MODEL=OLMo2-32B
TOK_PER_IT=8388888
ITS=(121000 239000 360000 477000 596000 721901)

for IT in "${ITS[@]}"; do
	export REVISION=stage1-step$IT-tokens$(( (TOK_PER_IT*IT + 1000000000) / 1000000000 ))B
	sbatch --job-name eval-$MODEL-$IT scripts/evaluate.sbatch allenai/OLMo-2-0325-32B $IT $TOK_PER_IT $MODEL
done
