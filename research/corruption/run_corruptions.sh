#!/bin/bash
#SBATCH --job-name=corrupt_gen
#SBATCH --account=def-majidk
#SBATCH --time=01:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/corrupt_%j.out
# CPU only, network-bound. Measured ~14 s per gpt-image-1 edit, so 228 images
# land near 55 min; 1h10 keeps this in the band that starts within ~15 minutes.
#
# The script is resumable -- it skips any output already on disk -- so a
# walltime kill or a network blip costs only a resubmit, never a repeat charge.
set -e

cd "${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
source venv/bin/activate
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"

python -u "$HOME/TactileExpert/research/corruption/generate_corruptions.py" \
    --apply --sleep 0.3 "$@"
