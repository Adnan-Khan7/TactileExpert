#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=clip_train_v2
#SBATCH --time=01:00:00   # ran 6m05s historically; 1h margin, fits 3h partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/clip_train_v2_%j.out
#SBATCH --error=logs/clip_train_v2_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
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

# Train all 4 heads on splits_v2 data (QN/non_conformant removed from taxonomy)
# QL and QT share CLIP features, so features are cached after the first head —
# subsequent heads in the same dim/cache folder skip re-extraction.
python "$REPO/research/baselines/train_clip_heads_v2.py" --dim QL
python "$REPO/research/baselines/train_clip_heads_v2.py" --dim QP
python "$REPO/research/baselines/train_clip_heads_v2.py" --dim QT
python "$REPO/research/baselines/train_clip_heads_v2.py" --dim QE

echo "Finished: $(date)"
