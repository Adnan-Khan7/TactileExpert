#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=tactile_score_review
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/review_%x_%j.out
#SBATCH --error=logs/review_%x_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch --export=SPLIT=test code/review/run_review.sh
#   sbatch --export=SPLIT=val  code/review/run_review.sh
#   sbatch --export=SPLIT=train,FLAG_BY=vf code/review/run_review.sh

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python:   $(which python)"
echo "Split:    ${SPLIT:-test}"
echo "Flag by:  ${FLAG_BY:-model}"
echo "Models:   ${MODELS:-clip resnet vit vlm_ft}"
echo "Started:  $(date)"

# repo / research-workspace anchors (overridable for containers)
REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs review

SPLIT="${SPLIT:-test}"
EXTRA_ARGS=""
if [ -n "${FLAG_BY:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --flag-by ${FLAG_BY}"
fi
if [ -n "${MODELS:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --models ${MODELS}"
fi

# Step 1: Score / flag all pairs in the split
python "$REPO/research/review/score_pipeline.py" --split "$SPLIT" $EXTRA_ARGS

# Step 2: Prepare inspection folders for flagged pairs
python "$REPO/research/review/prepare_inspection.py" --split "$SPLIT"

echo "Finished: $(date)"
echo ""
echo "Outputs:"
echo "  review/${SPLIT}/scored_${SPLIT}.xlsx   — fill 'user_label' column for conflict rows"
echo "  review/${SPLIT}/inspection/             — conflict images for visual review"
echo "  review/${SPLIT}/scoring_summary.txt     — conflict counts per option"
echo ""
echo "After reviewing, apply labels with:"
echo "  python "$REPO/research/review/apply_labels.py" --split ${SPLIT}"
