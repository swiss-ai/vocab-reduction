MODEL=Qwen2.5-7B
sbatch --job-name eval-$MODEL scripts/evaluate.sbatch Qwen/Qwen2.5-7B 1 18000000000000 $MODEL
