#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=gate_c1_vlm
#SBATCH --time=00:30:00
# WALLTIME BASIS: gate_vlms.sh notes ~5 min per eval_full_v4 pass on an h100 and
# ran two passes in a 1h allocation. This is one pass. 30m = large margin.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/gate_c1_vlm_%j.out
#SBATCH --error=logs/gate_c1_vlm_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

# Gate the C1 (BANA-aligned prompt) 7B judge against the deployed 7B.
#
# Scores at the DEPLOYED operating point (flag = prob > 0.50) on both frozen
# sets: the 153-pair original test and the 30-session / 150-decision GPT holdout.
# This is the comparison that matters — the val-F1-max thresholds in
# output_*/test_metrics.json are NOT used in deployment.
#
# Incumbent baseline for the VLM column lives in experiments_out/eval_7b_gate.json
# (VLM 132/150 holdout; eval_current_fleet.json reuses those per-pair scores —
# "identical adapter md5"). DO NOT overwrite that file.
#
# NOTE: eval_full_v4.py prints ENSEMBLE_WEIGHTS derived from original-test F1.
# Per SESSION_NOTES, do NOT paste those anywhere — deployed weights come from
# rederive_fleet.py. Only the VLM column of this run is being gated.
#
# SUBMIT FROM THE WORKSPACE:
#   cd "$TACTILE_DATA_ROOT" && sbatch ~/TactileExpert/research/experiments/gate_c1_vlm.sh

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTHONUNBUFFERED=1

REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs

echo "Job ID:  $SLURM_JOB_ID"
echo "Started: $(date)"

set -e

echo "=== GATE: C1 BANA-aligned 7B ==="
python "$REPO/research/experiments/eval_full_v4.py" \
    --vlm-ckpt output_c1_7b/best_checkpoint \
    --base-model Qwen/Qwen2-VL-7B-Instruct \
    --out experiments_out/eval_c1_gate.json

echo
echo "=== PAIRED SIGNIFICANCE vs deployed 7B (GPT holdout) ==="
python "$REPO/research/experiments/mcnemar_gate.py" \
    --incumbent experiments_out/eval_7b_gate.json \
    --candidate experiments_out/eval_c1_gate.json \
    --split gpt_test --model "VLM Fine-tuned"

echo
echo "=== PAIRED SIGNIFICANCE vs deployed 7B (in-domain test) ==="
python "$REPO/research/experiments/mcnemar_gate.py" \
    --incumbent experiments_out/eval_7b_gate.json \
    --candidate experiments_out/eval_c1_gate.json \
    --split test --model "VLM Fine-tuned"

echo "Finished: $(date)"
