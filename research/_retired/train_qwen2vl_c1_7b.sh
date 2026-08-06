#!/bin/bash
#SBATCH --account=def-majidk
#SBATCH --job-name=qwen2vl_c1_7b
#SBATCH --time=04:00:00
# WALLTIME BASIS: the deployed 7B (job 18514080) took 2h37m on the same corpus,
# same hyperparameters, early-stop epoch 9, batch-size 2, no OOM. Only the
# question text differs here, so runtime should match. 4h = ~50% margin.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/qwen2vl_c1_7b_%j.out
#SBATCH --error=logs/qwen2vl_c1_7b_%j.err
#SBATCH --mail-user=adnankhan5@cmail.carleton.ca
#SBATCH --mail-type=END,FAIL

# C1 — BANA-ALIGNED PROMPTS (2026-07-30)
#
# Retrains the deployed 7B judge with the question text and system prompt now
# shared with inference. This is a CONTROLLED comparison against output_7b:
# identical corpus (splits_v2), identical hyperparameters (seed 43, patience 5,
# epochs 10, batch 2), identical base model. THE ONLY DIFFERENCE IS THE PROMPTS.
#
# What changed, and why it is not just rewording (see evaluators/constants.py):
#   * missing_parts   v3 asked for parts "present in the reference photograph".
#                     BANA §2.8 sanctions simplification when intent is not
#                     compromised, so v3 penalised correct practice. v4 asks
#                     whether an ESSENTIAL part is absent.
#   * broken_lines    v3 asked whether a fingertip "would encounter gaps". BANA
#                     §3.4.3.3 permits dashed styles and §2.11 says a break where
#                     lines cross IMPROVES clarity, so v3 penalised correct
#                     technique. v4 excludes deliberate design breaks.
#   * too_thick, missing_texture, system prompt: aligned; missing_texture now
#                     phrased on ADJACENCY so it covers both no-fill-anywhere and
#                     fill-present-but-undifferentiated.
#   * questions now live ONLY in evaluators/constants.py and are imported by
#     build_vqa_v2.py, so train/inference drift cannot recur.
#
# The dataset is written to data/vqa_v4_bana, NOT over data/vqa_v2, so the
# dataset the deployed checkpoint was trained on stays reproducible.
#
# AFTER THIS: gate against the incumbent before deploying anything —
#   python "$REPO/research/experiments/gate_vision_candidates.py"   (or the VLM
#   gate path) and re-derive weights + recalibrate only if it wins. The notes
#   call this the cascade; do not skip it.
#
# SUBMIT FROM THE WORKSPACE so --output=logs/... resolves:
#   cd "$TACTILE_DATA_ROOT" && sbatch ~/TactileExpert/research/vlm/train_qwen2vl_c1_7b.sh

module load StdEnv/2023 python/3.11 scipy-stack
source ~/vlm_finetune/venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
mkdir -p $HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"

REPO="${TACTILE_REPO:-$HOME/TactileExpert}"
export TACTILE_DATA_ROOT="${TACTILE_DATA_ROOT:-$HOME/vlm_finetune}"
cd "$TACTILE_DATA_ROOT"
mkdir -p logs

set -e

echo "=== 1/2  Rebuilding VQA with BANA-aligned prompts (idempotent) ==="
# Pinned to splits_v2 — this run's claim is "identical corpus, prompts only".
# NOTE: until 2026-08-04 build_vqa_v2.py had no argparse at all, so the
# --out-dir below was silently DISCARDED and the build landed on the default.
python "$REPO/research/vlm/build_vqa_v2.py" \
    --splits-dir data/splits_v2 --out-dir data/vqa_v4_bana

# Fail fast if the built questions are not the deployed ones — the whole point
# of this run is that they are identical.
python - <<'PYCHECK'
import importlib.util, json, os, sys
from pathlib import Path
repo = Path(os.environ.get("TACTILE_REPO", Path.home() / "TactileExpert"))
spec = importlib.util.spec_from_file_location("c", repo / "evaluators/constants.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
rows = [json.loads(l) for l in open("data/vqa_v4_bana/train.jsonl")]
bad = [r["option_id"] for r in rows
       if r["option_id"] in C.QUESTIONS and r["question"] != C.QUESTIONS[r["option_id"]]]
if bad:
    sys.exit(f"ABORT: built questions differ from deployed for {sorted(set(bad))}")
sysprompt = {r["messages"][0]["content"] for r in rows}
if sysprompt != {C.SYSTEM_PROMPT}:
    sys.exit("ABORT: built system prompt differs from deployed")
print(f"prompt check OK — {len(rows)} train records carry the deployed question text")
PYCHECK

echo "=== 2/2  Fine-tuning Qwen2-VL-7B (C1 prompts; else identical to output_7b) ==="
python "$REPO/research/vlm/step9_finetune_vlm.py" \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --vqa-dir data/vqa_v4_bana \
    --out-dir output_c1_7b \
    --seed 43 \
    --patience 5 \
    --epochs 10 \
    --log

echo "Finished: $(date)"
