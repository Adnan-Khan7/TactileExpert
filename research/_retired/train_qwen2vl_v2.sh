#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=qwen2vl_v2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/qwen2vl_v2_%j.out
#SBATCH --error=logs/qwen2vl_v2_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

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

# Step 1 — build VQA dataset from new annotations (fast, CPU only)
echo "Building VQA v2 dataset..."
# Pinned to splits_v2/vqa_v2: this is a HISTORICAL run kept reproducible.
# build_vqa_v2.py now defaults to the v1 unified splits.
python "$REPO/research/vlm/build_vqa_v2.py" \
    --splits-dir data/splits_v2 --out-dir data/vqa_v2

# Step 2 — fine-tune VLM on new data, save to output_v2/
echo "Fine-tuning Qwen2-VL-2B..."
python "$REPO/research/vlm/step9_finetune_vlm.py" \
    --vqa-dir data/vqa_v2 \
    --out-dir output_v2 \
    --log

echo "Finished: $(date)"
