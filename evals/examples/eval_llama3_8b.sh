MODEL=Llama3.1-8B
sbatch --job-name eval-$MODEL scripts/evaluate.sbatch meta-llama/Meta-Llama-3.1-8B 1 15000000000000 $MODEL
