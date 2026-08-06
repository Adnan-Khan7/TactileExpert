#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=eval_gpt_test
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/ak1998/vlm_finetune/logs/eval_gpt_test_%j.out
#SBATCH --error=/home/ak1998/vlm_finetune/logs/eval_gpt_test_%j.err

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

# repo / research-workspace anchors (overridable for containers)
REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"


export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export PYTHONUNBUFFERED=1

python "$REPO/research/experiments/eval_gpt_test.py"
