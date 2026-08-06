#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=qwen2vl_v4_retry
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/qwen2vl_v4_retry_%j.out
#SBATCH --error=logs/qwen2vl_v4_retry_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

# VLM v4 RETRY (July 14): the July-9 v4 run peaked at epoch 3 (val 0.6304)
# and was rejected at the deploy gate. Retry with a DIFFERENT SEED and
# LONGER PATIENCE on the updated merge (64 pairs; holdout unchanged).
# Data recipe identical to v4 (build_vqa_v2.py, sampling seed 42) —
# only the training seed/patience/epochs differ.
#
# Deploy gate afterwards (GPU):
#   python "$REPO/research/experiments/eval_full_v4.py" \
#       --vlm-ckpt output_v4_retry/best_checkpoint \
#       --out experiments_out/eval_v4_retry_gate.json
# Deploy ONLY if it beats the deployed v3 on BOTH test sets
# (baseline: experiments_out/eval_deployed_fleet.json).

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
mkdir -p $HF_HOME
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

# Step 1 — rebuild VQA dataset from the merged splits (fast, CPU only)
echo "Building VQA v2 dataset..."
# Pinned to splits_v2/vqa_v2: this is a HISTORICAL run kept reproducible.
# build_vqa_v2.py now defaults to the v1 unified splits.
python "$REPO/research/vlm/build_vqa_v2.py" \
    --splits-dir data/splits_v2 --out-dir data/vqa_v2

# Step 2 — fine-tune with new seed + longer early-stopping patience
echo "Fine-tuning Qwen2-VL-2B (v4 retry: seed 43, patience 5)..."
python "$REPO/research/vlm/step9_finetune_vlm.py" \
    --vqa-dir data/vqa_v2 \
    --out-dir output_v4_retry \
    --seed 43 \
    --patience 5 \
    --epochs 10 \
    --log

echo "Finished: $(date)"
