#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=clip_probe_train
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/clip_probe_%j.out
#SBATCH --error=logs/clip_probe_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python:  $(which python)"
echo "Started: $(date)"

# repo / research-workspace anchors (overridable for containers)
REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs

python TactileEval_Context_And_Data/scripts/step2_train_stable_evaluator.py --dim QL --log
python TactileEval_Context_And_Data/scripts/step2_train_stable_evaluator.py --dim QP --log
python TactileEval_Context_And_Data/scripts/step2_train_stable_evaluator.py --dim QT --log

echo "Finished: $(date)"
