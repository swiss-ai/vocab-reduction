export TOKENIZER=alehc/swissai-tokenizer
export BOS=true

MODEL=Apertus3-8B_iter_1678000-tulu3-sft-RLVR-105
CKPT_PATH=/capstor/store/cscs/swissai/infra01/reasoning/models/global_step_105/hf_actor
sbatch --job-name eval-$MODEL scripts/evaluate_hf.sbatch $CKPT_PATH $MODEL
