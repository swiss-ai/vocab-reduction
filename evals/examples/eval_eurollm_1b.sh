MODEL=EuroLLM-1.7B
sbatch --job-name eval-$MODEL scripts/evaluate.sbatch utter-project/$MODEL 1 4000000000000 $MODEL
