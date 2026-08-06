#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=qwen2vl_v5
#SBATCH --time=03:00:00
# WALLTIME BASIS: identical 2B recipe (v4 retry, job 17602252) ran 2h04m to
# early-stop at epoch 9. 3h = ~40% margin AND fits the 3h GPU partition
# (gpubase_bygpu_b1, fastest turnover) — important under low fairshare.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/qwen2vl_v5_%j.out
#SBATCH --error=logs/qwen2vl_v5_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

# VLM v5 (July 23): 2B retrain on the 154-pair merge (train 5,022 rows;
# holdout grown to 30 pairs/150 flags). Uses the retry's healthier training
# regime (seed 43, patience 5, epochs 10). Next lever after the rejected v4
# retry was DATA VOLUME — this is that experiment.
#
# Deploy gate afterwards (both test sets, vs RE-BASELINED deployed fleet on
# the expanded 30-pair holdout — eval_deployed_fleet.json regenerated in the
# same batch):
#   python "$REPO/research/experiments/eval_full_v4.py" \
#       --vlm-ckpt output_v5/best_checkpoint \
#       --out experiments_out/eval_v5_gate.json

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

echo "Building VQA v2 dataset from merged splits..."
# Pinned to splits_v2/vqa_v2: this is a HISTORICAL run kept reproducible.
# build_vqa_v2.py now defaults to the v1 unified splits.
python "$REPO/research/vlm/build_vqa_v2.py" \
    --splits-dir data/splits_v2 --out-dir data/vqa_v2

echo "Fine-tuning Qwen2-VL-2B (v5: 154-pair merge, seed 43, patience 5)..."
python "$REPO/research/vlm/step9_finetune_vlm.py" \
    --vqa-dir data/vqa_v2 \
    --out-dir output_v5 \
    --seed 43 \
    --patience 5 \
    --epochs 10 \
    --log

echo "Finished: $(date)"
