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

# Without this a crashed trainer still exits 0 (the trailing echo succeeds)
# and sacct reports COMPLETED — a silent failure that looks like a result.
set -e

# extra args are forwarded to every trainer invocation (e.g. --seed 7)
EXTRA_ARGS=("$@")

# Train all 4 heads on splits_v3_unified (the v1 dataset).
# Feature caches are namespaced by split directory, so the splits_v2 caches on
# disk cannot be picked up — they hold splits_v2 LABELS as well as features.
# --model-dir is REQUIRED: the default is the deployed fleet's only backup.
SPLITS="$TACTILE_DATA_ROOT/data/splits_v3_unified"
# Overridable so an ablation arm trains without clobbering the main fleet:
#   sbatch --export=ALL,MODELS_OUT=<dir> train_clip_heads_v2.sh
CLIP_V1="${MODELS_OUT:-$TACTILE_DATA_ROOT/TactileEval_Context_And_Data/models_v1}/clip_probe"

python "$REPO/research/baselines/train_clip_heads_v2.py" --dim QL --splits-dir "$SPLITS" --model-dir "$CLIP_V1" "${EXTRA_ARGS[@]}"
python "$REPO/research/baselines/train_clip_heads_v2.py" --dim QP --splits-dir "$SPLITS" --model-dir "$CLIP_V1" "${EXTRA_ARGS[@]}"
python "$REPO/research/baselines/train_clip_heads_v2.py" --dim QT --splits-dir "$SPLITS" --model-dir "$CLIP_V1" "${EXTRA_ARGS[@]}"
python "$REPO/research/baselines/train_clip_heads_v2.py" --dim QE --splits-dir "$SPLITS" --model-dir "$CLIP_V1" "${EXTRA_ARGS[@]}"

echo "Finished: $(date)"
