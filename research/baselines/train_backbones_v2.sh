#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=backbone_train_v2
#SBATCH --time=01:00:00   # both backbones ran 3m54s total; 1h margin, fits 3h partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=logs/backbone_v2_%j.out
#SBATCH --error=logs/backbone_v2_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"

# repo / research-workspace anchors (overridable for containers)
REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs

python "$REPO/research/baselines/train_backbones_v2.py" --backbone resnet50
python "$REPO/research/baselines/train_backbones_v2.py" --backbone vit_b_16

echo "Finished: $(date)"
