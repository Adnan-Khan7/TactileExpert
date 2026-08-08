#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=vlm_sweep
#SBATCH --time=06:00:00
# WALLTIME BASIS: one 7B run measured 2h37m59s (job 19280190, early-stop epoch 9,
# batch-size 2). Two seeds per job = 5h16m; 6h leaves ~14% margin. Raise --time if
# a seed runs to the full 10 epochs without early-stopping.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/vlm_sweep_%j.out
#SBATCH --error=logs/vlm_sweep_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL
#
# Seed sweep for the Qwen2-VL-7B judge, to give it the noise floor the four vision
# judges already have.
#
# WHY IT IS NEEDED: at 918-verified the VLM is the best judge on both macro AUC
# (0.823) and macro F1 (0.615), and it is the ONLY judge showing a positive
# `too_thick` delta (+0.055 AUC) — the one option a fully verified training set did
# not fix. With a single seed none of that is quotable: the probes' `too_thick` AUC
# 2σ runs 0.037–0.102, so a +0.055 from one run sits inside the range where a probe
# delta gets dismissed as noise.
#
# WHY SEVERAL JOBS IN PARALLEL, NOT SEEDS LOOPED IN ONE:
# `seed_sweep.sh` loops seeds inside a single job because those trainers take 2–6
# minutes and queue latency dominates compute. Here one seed takes 2h38m, so
# compute dominates and the reasoning inverts — 12 seeds serially is ~31h, versus
# ~5h wall if six jobs run concurrently. Submit a few seeds per job, several jobs.
#
# THE VQA SET IS NOT REBUILT HERE. train_qwen2vl_7b.sh rebuilds
# data/vqa_v3_unified from the splits, which is correct for a single job but a data
# race across parallel ones — several jobs rewriting the same files while others
# read them. This script REQUIRES the build to exist already and refuses to run
# otherwise, so every seed trains on byte-identical data.
#
# Usage:
#   sbatch vlm_seed_sweep.sh <out_root> <seed> [seed...]
#   sbatch vlm_seed_sweep.sh .../vlm_sweep_tv918 1 2

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
mkdir -p $HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs

# Without this a crashed trainer exits 0 and SLURM reports COMPLETED.
set -e

OUT_ROOT="$1"; shift
SEEDS=("$@")
if [ -z "$OUT_ROOT" ] || [ ${#SEEDS[@]} -eq 0 ]; then
    echo "usage: sbatch vlm_seed_sweep.sh <out_root> <seed> [seed...]" >&2
    exit 1
fi

VQA="$TACTILE_DATA_ROOT/data/vqa_v3_unified"
for f in train.jsonl val.jsonl test.jsonl; do
    if [ ! -s "$VQA/$f" ]; then
        echo "REFUSING: $VQA/$f is missing or empty." >&2
        echo "  Build it ONCE before launching parallel seeds:" >&2
        echo "    python $REPO/research/vlm/build_vqa_v2.py \\" >&2
        echo "        --splits-dir data/splits_v3_unified --out-dir data/vqa_v3_unified" >&2
        exit 1
    fi
done

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "out:     $OUT_ROOT"
echo "seeds:   ${SEEDS[*]}"
echo "vqa:     $VQA (pre-built, not rebuilt here)"
echo "Started: $(date)"

for s in "${SEEDS[@]}"; do
    D="$OUT_ROOT/s$s"
    if [ -e "$D" ] && [ -n "$(ls -A "$D" 2>/dev/null)" ]; then
        echo "===== seed $s: $D exists and is non-empty — SKIPPING ====="
        continue
    fi
    echo "===== seed $s -> $D  ($(date +%H:%M:%S)) ====="
    python "$REPO/research/vlm/step9_finetune_vlm.py" \
        --model-path Qwen/Qwen2-VL-7B-Instruct \
        --vqa-dir data/vqa_v3_unified \
        --out-dir "$D" \
        --seed "$s" \
        --patience 5 \
        --epochs 10 \
        --log
done

echo "Finished: $(date)"
