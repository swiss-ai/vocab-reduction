export SIZE=70

MODEL=Llama3.3-70B-Instruct

export TASKS=english
sbatch --job-name eval-$MODEL-$TASKS scripts/evaluate.sbatch meta-llama/Llama-3.3-70B-Instruct 1 15000000000000 $MODEL
export TASKS=multilingual
sbatch --job-name eval-$MODEL-$TASKS scripts/evaluate.sbatch meta-llama/Llama-3.3-70B-Instruct 1 15000000000000 $MODEL
