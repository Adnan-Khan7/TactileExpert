#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=policy_bc_dpo
#SBATCH --time=00:30:00
# WALLTIME BASIS: measured 9m17s end to end on 2B (job 18704731, 2026-07-29:
# build + 8 BC epochs over 115 records + 4 DPO epochs over 44 pairs, MaxRSS
# 6.7G, h100). 30m is ~3x margin. The 7B run has NOT been timed — the 2B/7B
# ratio on the judge fine-tune was roughly 3x, so budget 01:00:00 for
# POLICY_MODEL=Qwen/Qwen2-VL-7B-Instruct and re-measure.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# 32G measured against MaxRSS 6.7G on the 2B run — 64G was over-requested.
# Raise for 7B.
#SBATCH --output=logs/policy_%j.out
#SBATCH --error=logs/policy_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

# Ch5 Loop 3 — edit-policy scaffold. PIPELINE VALIDATION, NOT AN H2 RESULT.
#
# Chain: extracted BC/DPO records -> chat-format splits -> behavior cloning
#        -> DPO initialised from the BC checkpoint.
#
# The 2B base is deliberate: this run exists to prove the loop closes, and 2B
# is far cheaper on a throttled allocation. Pass POLICY_MODEL to match the
# deployed 7B judge once the pipeline is known-good:
#   POLICY_MODEL=Qwen/Qwen2-VL-7B-Instruct sbatch research/policy/train_policy.sh
#
# SUBMIT FROM THE WORKSPACE so --output=logs/... resolves:
#   cd "$TACTILE_DATA_ROOT" && sbatch ~/TactileExpert/research/policy/train_policy.sh

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

REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
MODEL="${POLICY_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs

set -e

echo "=== 1/3  Building policy splits (trajectory-grouped, shared BC/DPO) ==="
python "$REPO/research/policy/build_policy_dataset.py" --seed 42

echo "=== 2/3  Behavior cloning (model: $MODEL) ==="
python "$REPO/research/policy/train_bc_policy.py" \
    --model-path "$MODEL" \
    --out-dir output_policy_bc \
    --epochs 8 \
    --patience 3 \
    --seed 42

echo "=== 3/3  DPO from the BC checkpoint (all 57 pairs, no reward filter) ==="
python "$REPO/research/policy/train_dpo_policy.py" \
    --model-path "$MODEL" \
    --init-from output_policy_bc/best_checkpoint \
    --out-dir output_policy_dpo \
    --epochs 4 \
    --patience 2 \
    --seed 42

echo "Finished: $(date)"
