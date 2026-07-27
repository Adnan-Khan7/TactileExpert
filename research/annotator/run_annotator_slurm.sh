#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=annotator_ui
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/annotator_ui_%j.out
#SBATCH --error=logs/annotator_ui_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export PYTHONUNBUFFERED=1

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"

# repo / research-workspace anchors (overridable for containers)
REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs

python "$REPO/research/annotator/annotator.py" --share --port 7862

echo "Finished: $(date)"
