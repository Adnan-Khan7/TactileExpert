#!/usr/bin/env python3
"""
Pre-compute evaluations for all test-set image pairs across all models.
Loads each model once, sweeps all pairs, saves to output/precomputed_results.json.

Progress is checkpointed after each model, so a partial run can be resumed
by passing --models with only the remaining model names.

Usage:
    python code/ui/precompute.py                        # all 5 models
    python code/ui/precompute.py --models "CLIP Probe" ResNet-50   # specific models only
    sbatch code/ui/precompute.sh                        # via SLURM (recommended)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import json
import sys
import time
from pathlib import Path

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_ROOT
DATA_DIR     = PROJECT_ROOT / "data" / "vqa"
IMAGES_DIR   = PROJECT_ROOT / "images"
OUTPUT_FILE  = PROJECT_ROOT / "output" / "precomputed_results.json"
MODELS_DIR   = PROJECT_ROOT / "TactileEval_Context_And_Data" / "models"
DEFAULT_CKPT = PROJECT_ROOT / "output" / "best_checkpoint"

sys.path.insert(0, str(HERE))
from vlm_evaluator import QUESTIONS

ALL_MODEL_NAMES = ["VLM Fine-tuned", "CLIP Probe"]


# ---------------------------------------------------------------------------
# Test-pair loader (mirrors app.py logic)
# ---------------------------------------------------------------------------

def load_test_pairs() -> list[dict]:
    """Returns list of {label, nat_abs, tac_abs, nat_rel, tac_rel}."""
    jsonl = DATA_DIR / "test.jsonl"
    seen, pairs, label_counts = set(), [], {}
    for line in open(jsonl):
        rec     = json.loads(line)
        nat_rel = rec["natural_image"]
        tac_rel = rec["tactile_image"]
        key     = (nat_rel, tac_rel)
        if key in seen:
            continue
        seen.add(key)
        nat_abs = str(IMAGES_DIR / nat_rel)
        tac_abs = str(IMAGES_DIR / tac_rel)
        if not (Path(nat_abs).exists() and Path(tac_abs).exists()):
            print(f"  [skip] missing images: {nat_rel}")
            continue
        parts = Path(nat_rel).parts
        if len(parts) >= 3:
            base = f"{parts[0].replace('_',' ')} / {parts[1]} / {Path(parts[-1]).stem.split('_')[0]}"
        else:
            base = nat_rel
        label_counts[base] = label_counts.get(base, 0) + 1
        label = base if label_counts[base] == 1 else f"{base} ({label_counts[base]})"
        pairs.append({"label": label, "nat_abs": nat_abs, "tac_abs": tac_abs,
                      "nat_rel": nat_rel, "tac_rel": tac_rel})
    return sorted(pairs, key=lambda x: x["label"])


# ---------------------------------------------------------------------------
# Model loader factory (same as app.py)
# ---------------------------------------------------------------------------

def make_loader(name: str, model_path: str, checkpoint: str):
    from vlm_evaluator      import VLMEvaluator
    from baseline_evaluator import CLIPProbeEvaluator

    if name == "VLM Fine-tuned":
        return VLMEvaluator(model_path=model_path, lora_checkpoint=checkpoint)
    if name == "CLIP Probe":
        return CLIPProbeEvaluator(models_dir=MODELS_DIR / "clip_probe")
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",      nargs="+", default=ALL_MODEL_NAMES,
                        help="Which models to run (default: all five)")
    parser.add_argument("--checkpoint",  default=str(DEFAULT_CKPT))
    parser.add_argument("--model-path",  default="Qwen/Qwen2-VL-2B-Instruct")
    args = parser.parse_args()

    # Validate requested models
    for m in args.models:
        if m not in ALL_MODEL_NAMES:
            print(f"Unknown model '{m}'. Valid: {ALL_MODEL_NAMES}")
            sys.exit(1)

    pairs = load_test_pairs()
    print(f"Test pairs: {len(pairs)}")
    print(f"Models to run: {args.models}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    # Load existing results (for resuming partial runs)
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing pair entries from checkpoint.\n")
    else:
        results = {}

    # Ensure all pair entries exist
    for pair in pairs:
        if pair["label"] not in results:
            results[pair["label"]] = {
                "nat_rel": pair["nat_rel"],
                "tac_rel": pair["tac_rel"],
                "models":  {},
                "timings": {},
            }

    # Run each model across all pairs
    for model_name in args.models:
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Skip pairs already done for this model
        todo = [p for p in pairs
                if model_name not in results[p["label"]]["models"]]
        if not todo:
            print(f"  All pairs already done for {model_name}, skipping.\n")
            continue

        print(f"  Loading model…")
        t_load = time.time()
        evaluator = make_loader(model_name, args.model_path, args.checkpoint)
        print(f"  Model loaded in {time.time()-t_load:.1f}s\n")

        for i, pair in enumerate(todo):
            t0 = time.time()
            result = evaluator.evaluate(pair["nat_abs"], pair["tac_abs"])
            elapsed = time.time() - t0

            results[pair["label"]]["models"][model_name]  = result
            results[pair["label"]]["timings"][model_name] = round(elapsed, 2)

            flagged = result["flagged"]
            print(f"  [{i+1:3d}/{len(todo)}] {pair['label'][:50]:<50}  "
                  f"{elapsed:5.1f}s  flagged={flagged or 'none'}")

            # Checkpoint after every pair so progress isn't lost
            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f)

        print(f"\n  {model_name} done.\n")

        # Free GPU memory before loading next model
        del evaluator
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Final summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for pair in pairs:
        done = list(results[pair["label"]]["models"].keys())
        print(f"  {pair['label'][:55]:<55}  {len(done)}/{len(ALL_MODEL_NAMES)} models")

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
