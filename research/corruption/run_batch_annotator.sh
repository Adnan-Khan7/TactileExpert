#!/bin/bash
# Review the deliberate-corruption batch in the annotator.
#
# Login node, no GPU: the VLM scores are pre-computed into the batch directory
# by prescore_vlm.py, so this is a Gradio form serving PNGs.
#
# Natural images resolve from the corpus; tactile images resolve from the batch
# directory, hence --tactile-dir.
set -e
BATCH="${1:-$HOME/TactileExpert/verify_generated_batch}"
PORT="${2:-7873}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
source venv/bin/activate
exec python -u "$HOME/TactileExpert/research/annotator/annotator.py" \
    --share --port "$PORT" \
    --annotations-file "$BATCH/annotations.jsonl" \
    --tactile-dir "$BATCH/images"
