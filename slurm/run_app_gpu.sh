#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=tactile_app
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/app_%j.out
#SBATCH --error=logs/app_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#
# Serve the collection + evaluation interface with the FULL five-judge fleet.
#
# GPU because the VLM is a 7B — on CPU its per-image latency makes the loop
# unusable interactively. The four vision judges would run fine on a login node
# if a GPU allocation is not worth the queue wait; see run_local.sh.
#
# 12h lands this in a longer-walltime partition, so expect queue time before the
# share link appears. --mail-type=BEGIN sends mail the moment it starts.
#
# The share link is printed to logs/app_<jobid>.out. Retrieve it with:
#   grep -o 'https://[a-z0-9]*\.gradio\.live' logs/app_<jobid>.out
#
# Collection writes to generated_training_data_v2/ — deliberately NOT the v1
# corpus in generated_training_data/, which the splits and trajectories.jsonl
# still reference.

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
export TACTILE_COLLECT_DIR="${TACTILE_COLLECT_DIR:-$REPO/generated_training_data_v2}"

# gpt-image-1 credentials. Extract ONLY the export line rather than sourcing the
# file: it has accumulated free-text notes, and sourcing prose into a shell fails
# on the first stray bracket — silently leaving the key unset.
KEYFILE="$TACTILE_DATA_ROOT/TactileEval_Context_And_Data/api/api_key.txt"
if [ -f "$KEYFILE" ]; then
    KEYLINE=$(grep -m1 '^export OPENAI_API_KEY=' "$KEYFILE" || true)
    [ -n "$KEYLINE" ] && eval "$KEYLINE"
fi
if [ -n "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY: set (${#OPENAI_API_KEY} chars)"
else
    echo "OPENAI_API_KEY: NOT SET — paste it in the UI field to generate/edit"
fi

cd "$REPO"
mkdir -p logs "$TACTILE_COLLECT_DIR"

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Models:   $REPO/models"
echo "Collect:  $TACTILE_COLLECT_DIR"
echo "Started:  $(date)"
echo

# Port is overridable so a replacement instance can be brought up
# alongside a running one and only cut over once it is serving.
python app.py --share --port "${APP_PORT:-7870}"

echo "Finished: $(date)"
