#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=rescore_traj
#SBATCH --time=02:00:00
# WALLTIME BASIS: rederive_fleet.sh requested 1:30 for 309 image-pairs on the
# same fleet. This scores the 393-image union (1.27x), so 2h carries ~35%
# margin. Billing is on USED time, not requested — do not pad further.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/eval/rescore_traj_%j.out
#SBATCH --error=logs/eval/rescore_traj_%j.err

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate
export HF_HOME=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

# repo / research-workspace anchors (overridable for containers)
REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs/eval

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"

python "$REPO/research/experiments/rescore_trajectories.py"

echo "Finished: $(date)"
