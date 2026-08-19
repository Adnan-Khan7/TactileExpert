#!/bin/bash
#SBATCH --job-name=morph_feat
#SBATCH --account=def-majidk
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/morph_%j.out
# CPU only -- no GPU request. Measured at ~0.6 s/image single-core and 168 MB
# peak RSS, so 1,350 pairs land around 15 min. 30 min keeps this inside the
# <=1h10 band that starts within ~15 minutes at the current fair-share.
set -e

cd "${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
source venv/bin/activate
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
export OMP_NUM_THREADS=1

python -u "$HOME/TactileExpert/research/morphology/extract_morph_features.py" \
    --split all \
    --out experiments_out/morph_features_tv974.json \
    "$@"
