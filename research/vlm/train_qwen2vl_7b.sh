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

# Without this a crashed trainer exits 0 — the trailing `echo` succeeds — and
# SLURM reports COMPLETED, which is indistinguishable from a result. This has
# already cost one run on this project.
set -e

# Output directory is the FIRST POSITIONAL ARG, defaulting to the v1 run so
# existing invocations behave as before:
#     sbatch train_qwen2vl_7b.sh output_7b_tv918
# Each label state gets its own directory. Overwriting one destroys the
# provenance of numbers already reported from it — history.json, test_metrics.json
# and train_log.txt are the audit trail for the VLM row in the README — and
# mcnemar_gate.py needs the previous arm to compare against.
OUT_DIR="${1:-output_7b_v1}"
if [ -e "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ] && [ -z "$VLM_FORCE_OVERWRITE" ]; then
    echo "REFUSING: $OUT_DIR exists and is not empty." >&2
    echo "  It holds the provenance of an already-reported run. Pass a new" >&2
    echo "  directory, or set VLM_FORCE_OVERWRITE=1 if you truly mean to replace it." >&2
    exit 1
fi
echo "Output:  $OUT_DIR"

# Rebuilds unconditionally from the splits, so a verification pass that changed
# labels is picked up. The VQA set is derived data, never a cache.
echo "Building VQA dataset from the v1 unified splits..."
python "$REPO/research/vlm/build_vqa_v2.py" \
    --splits-dir data/splits_v3_unified --out-dir data/vqa_v3_unified

echo "Fine-tuning Qwen2-VL-7B..."
python "$REPO/research/vlm/step9_finetune_vlm.py" \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --vqa-dir data/vqa_v3_unified \
    --out-dir "$OUT_DIR" \
    --seed 43 \
    --patience 5 \
    --epochs 10 \
    --log

echo "Finished: $(date)"
