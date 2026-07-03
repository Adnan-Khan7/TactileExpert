#!/bin/bash
# Launch TactileExpert on a login node (CPU-only fallback).
#
# Use during GPU maintenance windows or for UI smoke tests.
# Recommended: select only CLIP Probe or DINOv2 Probe in the UI —
# both run in a few seconds on CPU.  VLM and backbone models take
# several minutes per evaluation in float32 without a GPU.
#
#   bash run_local.sh [port]      # default port 7860
#
# Access via gradio.live share link (app.py --share is on by default).

set -e

module load StdEnv/2023 python/3.11 scipy-stack 2>/dev/null || true
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache

# Be polite on the shared login node
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

# Flush prints immediately so the gradio.live share URL reaches the log
export PYTHONUNBUFFERED=1

PORT=${1:-7860}

echo "============================================"
echo "TactileExpert — login-node (CPU) mode"
echo "Host: $(hostname)  Port: ${PORT}"
echo "Access via the gradio.live public URL printed below."
echo "============================================"

cd "$(dirname "$0")"
python app.py --port "$PORT"
