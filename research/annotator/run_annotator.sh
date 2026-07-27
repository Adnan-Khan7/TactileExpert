#!/bin/bash
# Run the annotation tool on the login node (no GPU needed).
# Usage: bash code/annotator/run_annotator.sh
# In tmux: tmux new -s annotate  then  bash code/annotator/run_annotator.sh --share

set -e
# repo / research-workspace anchors (overridable for containers)
REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
source venv/bin/activate

# Build annotations.jsonl on first run (safe to re-run — asks before overwrite)
if [ ! -f annotations.jsonl ]; then
    echo "annotations.jsonl not found — running prepare_annotations.py first..."
    python "$REPO/research/annotator/prepare_annotations.py"
fi

echo ""
echo "Starting annotation tool..."
python "$REPO/research/annotator/annotator.py" "$@"
