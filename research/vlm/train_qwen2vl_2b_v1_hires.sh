#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=qwen2vl_2b_hires
#SBATCH --time=06:00:00
# WALLTIME BASIS: this is the ONLY job here that cannot use the fast 3h tier.
# Raising max_pixels 200704 -> 802816 quadruples the vision-token count per
# image (16x16 -> 32x32 tokens, two images per example), so the sequence roughly
# 4x's and step time grows faster than linearly. The 448px 2B run takes ~2h; 6h
# is a deliberate over-request in gpubase_bygpu_b2 because a walltime kill here
# wastes far more than the extra queue wait. This is a follow-up ablation, not a
# deliverable for the 2026-08-06 meeting, so a slow start is acceptable.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/qwen2vl_2b_hires_%j.out
#SBATCH --error=logs/qwen2vl_2b_hires_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL
#
# RESOLUTION ABLATION — one variable, matched control.
#
# HYPOTHESIS: the fleet's weakness on too_thick / broken_lines / extra_parts is
# an input-resolution limit, not a model-capacity limit. Those three defects are
# high-spatial-frequency (a few-pixel gap, a few-pixel width difference, a small
# spurious fragment), and they are exactly the three the fleet scores worst on,
# while the coarse defects (missing_texture, missing_parts) score best. Source
# images are commonly 736x736 and 1024x1024; at max_pixels=200704 they are
# downscaled to a 448x448 budget, where a 3px gap is sub-pixel.
#
# CONTROL: train_qwen2vl_2b_v1.sh — SAME base model, SAME data, SAME seed 43,
# patience 5, 10 epochs. The ONLY difference is max_pixels. If the high-res arm
# lifts the three fine-grained options and leaves missing_texture flat, the
# hypothesis holds and resolution is the lever to pull before swapping to
# Qwen3-VL / DINOv3 / a Siamese architecture.
#
# WHY THE 2B AND NOT THE 7B: the question is about resolution, so the cheaper
# base isolates it for roughly a third of the compute. Running it on the 7B
# would confound the answer with capacity and would not fit any sane walltime.

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"

REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs

set -e

if [ ! -f data/vqa_v3_unified/train.jsonl ]; then
    echo "ERROR: data/vqa_v3_unified/train.jsonl missing." >&2
    exit 1
fi

# 802816 = 1024*28*28 -> a 896x896 budget, 4x the control's 200704 (256*28*28).
echo "Fine-tuning Qwen2-VL-2B on v1 at max_pixels=802816 (control: 200704)..."
python "$REPO/research/vlm/step9_finetune_vlm.py" \
    --model-path Qwen/Qwen2-VL-2B-Instruct \
    --vqa-dir data/vqa_v3_unified \
    --out-dir output_2b_v1_hires \
    --max-pixels 802816 \
    --seed 43 \
    --patience 5 \
    --epochs 10 \
    --log

echo "Finished: $(date)"
