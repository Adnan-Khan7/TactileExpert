#!/bin/bash
#SBATCH --job-name=prescore_vlm
#SBATCH --account=def-majidk
#SBATCH --time=01:10:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/prescore_%j.out
# 7B judge, 5 options per image. Resumable: cached scores are skipped, so a
# walltime kill costs only a resubmit.
set -e

cd "${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
source venv/bin/activate
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u "$HOME/TactileExpert/research/corruption/prescore_vlm.py" "$@"
