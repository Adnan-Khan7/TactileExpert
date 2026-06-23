#!/bin/bash
# Launch TactileExpert on a login node (CPU-only fallback).
#
# Use during GPU maintenance windows or for UI smoke tests. The CLIP Probe
# evaluates in a few seconds on CPU; the VLM takes a few minutes per
# evaluation (4 forward passes of Qwen2-VL-2B in float32).
#
#   bash run_local.sh [port]      # default port 7860
#
# Then from your laptop:
#   ssh -L 7860:$(hostname):7860 nibi
#   open http://localhost:7860

set -e

module load StdEnv/2023 python/3.11 scipy-stack 2>/dev/null || true
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache

# Be polite on the shared login node
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

# Flush prints immediately so the gradio.live share URL reaches the log
export PYTHONUNBUFFERED=1

PORT=${1:-7860}

echo "============================================"
echo "TactileExpert — login-node (CPU) mode"
echo "Host: $(hostname)"
echo "Tunnel from laptop:  ssh -L ${PORT}:$(hostname):${PORT} nibi"
echo "Then open:           http://localhost:${PORT}"
echo "============================================"

cd "$(dirname "$0")"
python app.py --port "$PORT"
