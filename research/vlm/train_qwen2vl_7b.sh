#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=qwen2vl_7b
#SBATCH --time=04:00:00
# WALLTIME BASIS: measured 2h37m on the 154-pair merge (job 18514080, early-
# stop epoch 9, batch-size 2, no OOM). 4h = ~50% margin. If a future run
# needs batch-size 1, roughly double and revisit.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/qwen2vl_7b_%j.out
#SBATCH --error=logs/qwen2vl_7b_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

# 7B JUDGE-CAPACITY EXPERIMENT (Ch5 §5.3, July 23): identical recipe to the
# 2B v5 run (same 154-pair merge, seed 43, patience 5, epochs 10) with only
# the base model scaled to Qwen2-VL-7B-Instruct. Quantifies judge-capacity
# effects for the thesis. Depends on the SAME data/vqa_v2 build as v5 —
# submit AFTER (or alongside) the v5 job; the build step here is idempotent
# and cheap, so both scripts run it defensively.
#
# NOTE: if the 7B OOMs at batch 2, resubmit with --batch-size 1.

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

echo "Building VQA dataset from the v1 unified splits (idempotent)..."
python "$REPO/research/vlm/build_vqa_v2.py" \
    --splits-dir data/splits_v3_unified --out-dir data/vqa_v3_unified

# out-dir is output_7b_v1, NOT output_7b: output_7b is the incumbent and is
# needed as the comparison arm in mcnemar_gate.py.
echo "Fine-tuning Qwen2-VL-7B on v1..."
python "$REPO/research/vlm/step9_finetune_vlm.py" \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --vqa-dir data/vqa_v3_unified \
    --out-dir output_7b_v1 \
    --seed 43 \
    --patience 5 \
    --epochs 10 \
    --log

echo "Finished: $(date)"
