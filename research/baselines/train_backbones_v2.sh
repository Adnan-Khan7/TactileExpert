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

# Without this a crashed trainer still exits 0 (the trailing echo succeeds)
# and sacct reports COMPLETED — a silent failure that looks like a result.
set -e

# extra args are forwarded to every trainer invocation (e.g. --seed 7)
EXTRA_ARGS=("$@")

# MODELS_OUT selects where the checkpoints land. Default is the v1 fleet:
#   sbatch --export=ALL,MODELS_OUT=<dir> train_backbones_v2.sh --seed N
# Vary --seed across runs to measure the noise floor a result must clear.
MODELS_V1="${MODELS_OUT:-$TACTILE_DATA_ROOT/TactileEval_Context_And_Data/models_v1}"

python "$REPO/research/baselines/train_backbones_v2.py" --backbone resnet50 \
    --splits-dir "$TACTILE_DATA_ROOT/data/splits_v3_unified" --model-dir "$MODELS_V1" "${EXTRA_ARGS[@]}"
python "$REPO/research/baselines/train_backbones_v2.py" --backbone vit_b_16 \
    --splits-dir "$TACTILE_DATA_ROOT/data/splits_v3_unified" --model-dir "$MODELS_V1" "${EXTRA_ARGS[@]}"

echo "Finished: $(date)"
