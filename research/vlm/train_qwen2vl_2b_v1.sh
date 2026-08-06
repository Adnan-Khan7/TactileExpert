#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=qwen2vl_2b_v1
#SBATCH --time=03:00:00
# WALLTIME BASIS: the 2B on the previous corpus ran 02:36:43 (job 18514079) and
# 02:04:42 (job 17602252, early-stopped at epoch 9). The v1 VQA train set is
# 4,174 records against the old 5,022, so this should be faster still. 3h keeps
# it in gpubase_bygpu_b1 — the 3h tier has by far the best turnover, which is
# what matters under a 0.16 fairshare with ~2,100 jobs queued ahead.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/qwen2vl_2b_v1_%j.out
#SBATCH --error=logs/qwen2vl_2b_v1_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL
#
# Qwen2-VL-2B LoRA on the v1 unified dataset.
#
# PURPOSE: the judge-capacity comparison, run fresh. Everything except the base
# model is identical to train_qwen2vl_7b.sh — same VQA build, same seed 43,
# same patience 5, same 10 epochs, same LoRA config. That is the whole point:
# if 2B and 7B differ on v1, the base model is the only thing that could have
# caused it.
#
# DOES NOT REBUILD THE VQA DATA. data/vqa_v3_unified is built once by the 7B
# wrapper and is deterministic (seed 42), so rebuilding would at best be a
# no-op and at worst rewrite files a concurrently running job is reading.
# If the directory is missing, this fails loudly rather than silently training
# on something else.

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
    echo "Build it first:  python \"$REPO/research/vlm/build_vqa_v2.py\" \\" >&2
    echo "    --splits-dir data/splits_v3_unified --out-dir data/vqa_v3_unified" >&2
    exit 1
fi
echo "VQA source: data/vqa_v3_unified ($(wc -l < data/vqa_v3_unified/train.jsonl) train records)"

# out-dir is output_2b_v1 — never over output_7b_v1, the two are compared.
echo "Fine-tuning Qwen2-VL-2B on v1..."
python "$REPO/research/vlm/step9_finetune_vlm.py" \
    --model-path Qwen/Qwen2-VL-2B-Instruct \
    --vqa-dir data/vqa_v3_unified \
    --out-dir output_2b_v1 \
    --seed 43 \
    --patience 5 \
    --epochs 10 \
    --log

echo "Finished: $(date)"
