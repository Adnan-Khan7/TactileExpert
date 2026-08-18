#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=seed_sweep
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=logs/seed_sweep_%j.out
#SBATCH --error=logs/seed_sweep_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL
#
# Train one model family across many seeds inside a SINGLE job.
#
# WHY ONE JOB AND NOT N JOBS: with fair-share at ~0.16 and thousands of jobs
# queued ahead, queue latency dominates compute — these trainers take 2-6
# minutes each. Twelve separate submissions would wait twelve times. One job
# waits once and then runs the sweep back to back.
#
# WHY THE SEEDS MATTER: the trainers seeded nothing until 2026-08-05, so
# checkpoint-to-checkpoint deltas carried an unknown amount of run-to-run
# noise. A sweep at a FIXED label state measures that noise directly, and any
# later comparison has to clear it. Per-option confidence intervals shrink as
# 1/sqrt(n_seeds), which is the whole point of running twelve rather than three:
# too_thick has only 26 positives in test and needs the extra resolution.
#
# All seeds share ONE feature cache (--cache-dir). Features depend on the
# images, not the seed, so extracting them once turns the CLIP sweep from
# twelve 6-minute runs into one extraction plus twelve head fits.
#
# Usage:
#   sbatch seed_sweep.sh backbones <out_root> [seeds...]
#   sbatch seed_sweep.sh clip      <out_root> [seeds...]

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs

set -e

FAMILY="$1"; OUT_ROOT="$2"; shift 2
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12)

# Overridable so the SAME models can be re-evaluated on a different TEST set.
# Keep train/ and val/ identical and vary only test.jsonl: seeds are fixed, so the
# checkpoints come out bit-identical and the only thing that moves is the
# evaluation set. That isolates a test-set change from a learning effect without
# retraining anything conceptually different.
SPLITS="${TACTILE_SPLITS:-$TACTILE_DATA_ROOT/data/splits_v3_unified}"
CACHE="$OUT_ROOT/_shared_cache"

echo "family=$FAMILY  out=$OUT_ROOT  seeds=${SEEDS[*]}"
echo "splits=$SPLITS"
echo "Started: $(date)"

for s in "${SEEDS[@]}"; do
    D="$OUT_ROOT/s$s"
    echo "===== seed $s -> $D  ($(date +%H:%M:%S)) ====="
    case "$FAMILY" in
      backbones)
        for bb in resnet50 vit_b_16; do
          python "$REPO/research/baselines/train_backbones_v2.py" \
            --backbone $bb --splits-dir "$SPLITS" --model-dir "$D" --seed "$s"
        done ;;
      clip)
        for dim in QL QP QT QE; do
          python "$REPO/research/baselines/train_clip_heads_v2.py" \
            --dim $dim --splits-dir "$SPLITS" --model-dir "$D/clip_probe" \
            --cache-dir "$CACHE/clip_probe" --seed "$s"
        done ;;
      dino)
        for dim in QL QP QT QE; do
          extra=""; [ "$dim" = "QT" ] && extra="--gamma-neg 8.0"
          python "$REPO/research/baselines/train_dino_probe_v2.py" \
            --dim $dim $extra --splits-dir "$SPLITS" --model-dir "$D/dino_probe" \
            --cache-dir "$CACHE/dino_probe" --seed "$s"
        done ;;
      *) echo "unknown family: $FAMILY" >&2; exit 1 ;;
    esac
done

echo "Finished: $(date)"
