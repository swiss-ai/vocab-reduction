#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=vanilla_bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

# 1. Force environment activation via absolute paths
source /users/faruk_zahiragic/miniconda/bin/activate /users/faruk_zahiragic/miniconda/envs/apertus-bench

# 2. Set your HF Home
export HF_HOME="${HF_HOME:-$SCRATCH/hf_home/}"

# 3. Run the benchmark
python memory_latency_benchmark.py