#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=derive_thresh
#SBATCH --time=00:30:00
# WALLTIME BASIS: scores VAL (174 pairs) with all five judges including the 7B
# VLM. The full-test scoring run takes 2m45s for 183 pairs, so val is comparable;
# 30 min is ~10x margin and lands in the short-job bracket.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/derive_thresh_%j.out
#SBATCH --error=logs/derive_thresh_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL
#
# Re-derive per-option decision thresholds on VAL for whatever fleet is currently
# installed in TactileExpert/models/.
#
# RUN THIS AFTER EVERY FLEET INSTALL. Thresholds are fitted to a fleet's score
# distribution, so they do not survive a retrain. `app.py` defaults to
# calibrated mode and reads them from evaluators/constants.py, which means an
# install without this step leaves the UI scoring a NEW fleet through an OLD
# fleet's operating points. The 2026-08-07 918-verified install hit exactly that.
#
# VAL, NEVER TEST: a threshold is a fitted parameter, 25 of them across 5 judges
# x 5 options. Choosing them on the split you report is selection bias, and
# derive_val_thresholds.py refuses --split test.
#
# The script prints paste-ready dicts for evaluators/constants.py; paste them and
# note the fleet they belong to.

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTHONUNBUFFERED=1

REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs

# Without this a crashed run exits 0 and SLURM reports COMPLETED.
set -e

if [ -f "$REPO/models/FLEET_MANIFEST.json" ]; then
    echo "installed fleet:"
    python - <<'PY'
import json, os
m = json.load(open(os.path.expanduser("~/TactileExpert/models/FLEET_MANIFEST.json")))
print(f"  label_state : {m['label_state']}")
print(f"  vision      : {m['vision_source']}")
print(f"  vlm         : {m['vlm_source']}")
print(f"  seeds       : {sorted({p['seed'] for p in m['vision_provenance']})}")
PY
else
    echo "WARNING: no models/FLEET_MANIFEST.json — cannot say which fleet these"
    echo "         thresholds will belong to. Install via research/install_fleet.py."
fi

echo "Started: $(date)"
python "$REPO/research/experiments/derive_val_thresholds.py" "$@"
echo "Finished: $(date)"
