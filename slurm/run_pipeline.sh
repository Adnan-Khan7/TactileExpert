#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=tactile_expert
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

# HuggingFace cache (Qwen2-VL-7B base model lives here)
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python:  $(which python)"
echo "Started: $(date)"
echo "============================================"

cd ~/TactileExpert
mkdir -p logs

PORT=7860

python -u app.py \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --port $PORT

echo "Finished: $(date)"

# ============================================================
# To access the interface: app.py runs with --share by default,
# so open the gradio.live URL printed in logs/pipeline_<jobid>.out
#
#   grep -m1 "gradio.live" logs/pipeline_<jobid>.out
# ============================================================
