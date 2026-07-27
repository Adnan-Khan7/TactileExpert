#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=vlm_eval_dual
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/vlm_eval_dual_%j.out
#SBATCH --error=logs/vlm_eval_dual_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

# repo / research-workspace anchors (overridable for containers)
REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
python "$REPO/research/vlm/step10_eval_test_dual.py"

echo "Finished at: $(date)"
