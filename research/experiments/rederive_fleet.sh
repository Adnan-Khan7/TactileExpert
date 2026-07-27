#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=rederive_fleet
#SBATCH --time=01:30:00
# Re-scores collected data with the deployed v5+7B fleet, re-derives
# ENSEMBLE_WEIGHTS + Platt calibrators. 7B inference on ~260 image-pairs.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/eval/rederive_fleet_%j.out
#SBATCH --error=logs/eval/rederive_fleet_%j.err
module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate
export HF_HOME=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache PYTHONUNBUFFERED=1
# repo / research-workspace anchors (overridable for containers)
REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
python "$REPO/research/experiments/rederive_fleet.py"
echo "Finished: $(date)"
