export SIZE=70

MODEL=Llama3.1-70B

export TASKS=english
sbatch --job-name eval-$MODEL-$TASKS scripts/evaluate.sbatch meta-llama/Meta-Llama-3.1-70B 1 15000000000000 $MODEL
export TASKS=multilingual
sbatch --job-name eval-$MODEL-$TASKS scripts/evaluate.sbatch meta-llama/Meta-Llama-3.1-70B 1 15000000000000 $MODEL
