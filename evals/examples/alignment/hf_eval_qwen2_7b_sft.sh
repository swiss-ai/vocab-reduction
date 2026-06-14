MODEL=OLMo-2-0325-32B
CKPT_PATH=allenai/OLMo-2-0325-32B
sbatch --job-name eval-$MODEL scripts/evaluate_hf.sbatch $CKPT_PATH $MODEL