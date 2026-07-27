#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=tactile_adapt
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/tactile_adapt_%j.out
#SBATCH --error=logs/tactile_adapt_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"

# repo / research-workspace anchors (overridable for containers)
REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs

PROJ_CKPT="TactileEval_Context_And_Data/models/dino_probe_v2/projection_head.pt"

# ── Step 1: Contrastive pre-training (~45 min) ───────────────────────────────
echo ""
echo "====== Step 1: TactileAug contrastive pre-training ======"
python "$REPO/research/baselines/pretrain_tactile_contrastive.py" \
    --epochs 80 \
    --batch-size 48

# ── Step 2: Train adapted DINOv2 probe on projected features ─────────────────
# DINOv2 features are cached from Step 1 (shared cache), only projection applied
echo ""
echo "====== Step 2: DINOv2 + TactileAug adapted probe ======"
ADAPTED_DIR="TactileEval_Context_And_Data/models/dino_probe_v2_adapted"
for DIM in QL QP QT QE QN; do
    echo "  → dim=${DIM}"
    python "$REPO/research/baselines/train_dino_probe_v2.py" \
        --dim ${DIM} \
        --projection-ckpt ${PROJ_CKPT} \
        --model-dir ${ADAPTED_DIR}
done

echo ""
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  TactileEval_Context_And_Data/models/dino_probe_v2/"
echo ""
echo "Compare against plain DINOv2 probe (job 16639141):"
echo "  Plain DINOv2:    models/dino_probe_v2/{ql,qp,qt,qe,qn}_evaluator.pt"
echo "  Adapted DINOv2:  models/dino_probe_v2/{ql,qp,qt,qe,qn}_evaluator.pt"
echo "  (adapted checkpoints will overwrite plain ones — submit after plain job finishes)"
