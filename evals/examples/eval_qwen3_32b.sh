export SIZE=32
export TRANSFORMERS_BRANCH=v4.52.4 

MODEL=Qwen3-${SIZE}B
sbatch --job-name eval-$MODEL scripts/evaluate.sbatch Qwen/$MODEL 1 36000000000000 $MODEL

