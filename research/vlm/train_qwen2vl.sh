#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=qwen2vl_lora
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/qwen2vl_%j.out
#SBATCH --error=logs/qwen2vl_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

# --- environment setup ---
module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

# --- HuggingFace cache to scratch (avoids /home inode bloat) ---
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
mkdir -p $HF_HOME

# --- CUDA memory fragmentation helper (recommended after last OOM) ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- diagnostic info in the log ---
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"
echo "Starting at: $(date)"

# --- training ---
# repo / research-workspace anchors (overridable for containers)
REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
python "$REPO/research/vlm/step9_finetune_vlm.py" --log

echo "Finished at: $(date)"