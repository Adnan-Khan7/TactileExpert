#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=dino_probe_v2
#SBATCH --time=01:00:00   # ran 2m39s historically; 1h margin, fits 3h partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/dino_probe_v2_%j.out
#SBATCH --error=logs/dino_probe_v2_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Point torch hub cache to scratch so the DINOv2 weights are found
export TORCH_HOME=$SCRATCH/torch_cache

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

# Feature extraction happens once (shared cache across all dims).
# QL goes first — caches train/val/test features for all subsequent dims.
# QN (non_conformant) excluded — removed from pipeline per advisor feedback.
# QT uses higher gamma_neg=8 to suppress missing_texture false positives.
SPLITS="$TACTILE_DATA_ROOT/data/splits_v3_unified"
# Overridable so an ablation arm trains without clobbering the main fleet:
#   sbatch --export=ALL,MODELS_OUT=<dir> train_dino_probe_v2.sh
DINO_V1="${MODELS_OUT:-$TACTILE_DATA_ROOT/TactileEval_Context_And_Data/models_v1}/dino_probe"

python "$REPO/research/baselines/train_dino_probe_v2.py" --dim QL --splits-dir "$SPLITS" --model-dir "$DINO_V1" "${EXTRA_ARGS[@]}"
python "$REPO/research/baselines/train_dino_probe_v2.py" --dim QP --splits-dir "$SPLITS" --model-dir "$DINO_V1" "${EXTRA_ARGS[@]}"
python "$REPO/research/baselines/train_dino_probe_v2.py" --dim QT --gamma-neg 8.0 --splits-dir "$SPLITS" --model-dir "$DINO_V1" "${EXTRA_ARGS[@]}"
python "$REPO/research/baselines/train_dino_probe_v2.py" --dim QE --splits-dir "$SPLITS" --model-dir "$DINO_V1" "${EXTRA_ARGS[@]}"

echo "Finished: $(date)"
