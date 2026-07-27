#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=gate_vlms
#SBATCH --time=01:00:00
# Gates BOTH VLM candidates (2B v5, 7B) on original test + GPT holdout vs the
# re-baselined deployed fleet. Two eval_full passes in one GPU allocation
# (~5 min each). 7B needs its 7B base model (adapter dim must match).
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/gate_vlms_%j.out
#SBATCH --error=logs/gate_vlms_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTHONUNBUFFERED=1

# repo / research-workspace anchors (overridable for containers)
REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
echo "=================== GATE: 2B v5 ==================="
python "$REPO/research/experiments/eval_full_v4.py" \
    --vlm-ckpt output_v5/best_checkpoint \
    --out experiments_out/eval_v5_gate.json

echo "=================== GATE: 7B ======================"
python "$REPO/research/experiments/eval_full_v4.py" \
    --vlm-ckpt output_7b/best_checkpoint \
    --base-model Qwen/Qwen2-VL-7B-Instruct \
    --out experiments_out/eval_7b_gate.json

echo "Finished: $(date)"
