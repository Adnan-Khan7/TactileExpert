#!/usr/bin/env python3
"""
Evaluate CLIP Probe on the fresh data/splits_v2 test set.

Protocol:
  1. Load data/splits_v2/{val,test}.jsonl
  2. Run CLIP Probe on the 4 options it was trained on
     (too_thick, broken_lines, missing_parts, missing_texture)
  3. Calibrate thresholds on val (F1-maximising per option)
  4. Report metrics at balanced (t=0.50) AND val-calibrated thresholds
  5. Flag extra_parts and non_conformant as "no model — pending retrain"

Usage:
  salloc --account=def-majidk --gpus-per-node=h100:1 --cpus-per-task=4 \\
         --mem=32G --time=1:00:00
  source ~/vlm_finetune/venv/bin/activate
  cd ~/vlm_finetune
  python code/baselines/eval_on_new_splits.py
  python code/baselines/eval_on_new_splits.py --device cpu   # slower, no GPU needed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    raise ImportError("pip install open-clip-torch")
try:
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
except ImportError:
    raise ImportError("pip install scikit-learn")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = DATA_ROOT
IMAGES_DIR = ROOT / "images"
SPLITS_DIR = ROOT / "data" / "splits_v2"
MODELS_DIR = ROOT / "TactileEval_Context_And_Data" / "models"
# After retraining, v2 heads take precedence; fall back to original if v2 not yet ready
MODELS_DIR_V2  = MODELS_DIR / "clip_probe_v2"
MODELS_DIR_V1  = MODELS_DIR / "clip_probe"
OUTPUT_DIR = ROOT / "output"

# ── Option / head layout ─────────────────────────────────────────────────────
# All 5 heads. Script auto-detects which checkpoints exist and skips the rest.
TASK_HEADS = {
    "QL": ["too_thick",   "broken_lines"],
    "QP": ["missing_parts"],
    "QT": ["missing_texture"],
    "QE": ["extra_parts"],
    "QN": ["non_conformant"],
}
ALL_OPTIONS = [o for opts in TASK_HEADS.values() for o in opts]

# ── CLIP architecture (must mirror step2_train_stable_evaluator.py) ──────────
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "laion2b_s32b_b82k"
FEAT_DIM  = 768
INPUT_DIM = FEAT_DIM * 3


def build_clip_head(n_out: int, hidden: int = 512) -> nn.Module:
    return nn.Sequential(
        nn.Linear(INPUT_DIM, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, n_out),
    )


# ── Data loading ─────────────────────────────────────────────────────────────
def load_split(name: str) -> list[dict]:
    path = SPLITS_DIR / f"{name}.jsonl"
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(probs, labels, threshold=0.5) -> dict:
    y_true = np.array(labels)
    y_prob = np.array(probs)
    y_pred = (y_prob >= threshold).astype(int)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    tp  = int(((y_pred==1)&(y_true==1)).sum())
    fp  = int(((y_pred==1)&(y_true==0)).sum())
    fn  = int(((y_pred==0)&(y_true==1)).sum())
    return dict(f1=round(f1,4), auc=round(auc,4),
                precision=round(pre,4), recall=round(rec,4),
                tp=tp, fp=fp, fn=fn,
                n_pos=int(y_true.sum()), n_total=len(y_true),
                threshold=round(threshold,4))


def best_threshold(probs, labels) -> float:
    """Val-F1-maximising threshold, scanned over 0.05–0.95 in steps of 0.05."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 1.0, 0.05):
        f1 = f1_score(labels, (np.array(probs) >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2)


# ── CLIP Probe scoring ────────────────────────────────────────────────────────
def score_with_clip(records: list[dict], device: torch.device) -> dict[str, dict]:
    """Return {option_id: {probs: [...], labels: [...]}} for all known options."""
    print("\n[CLIP Probe] Loading backbone...")
    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # Load heads — prefer v2 (retrained on new data), fall back to v1
    heads = {}
    skipped = []
    for dim, opts in TASK_HEADS.items():
        ckpt_v2 = MODELS_DIR_V2 / f"{dim.lower()}_evaluator.pt"
        ckpt_v1 = MODELS_DIR_V1 / f"{dim.lower()}_evaluator.pt"
        if ckpt_v2.exists():
            ckpt_path, source = ckpt_v2, "v2"
        elif ckpt_v1.exists():
            ckpt_path, source = ckpt_v1, "v1"
        else:
            print(f"  [{dim}] no checkpoint found — skipping {opts}")
            skipped.extend(opts)
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        head = build_clip_head(len(opts), ckpt.get("hidden_dim", 512))
        head.load_state_dict(ckpt["state_dict"])
        head.eval().to(device)
        heads[dim] = (head, opts)
        print(f"  [{dim}] loaded {ckpt_path.name}  ({source})")

    available_options = [o for opts in [h[1] for h in heads.values()] for o in opts]

    # Filter to records with available options only
    known_records = [r for r in records if r["option_id"] in available_options]

    # Extract CLIP features for unique pairs in this record set
    unique_pairs = {(r["natural_image"], r["tactile_image"]) for r in known_records}
    print(f"[CLIP Probe] Extracting features for {len(unique_pairs)} pairs...")
    pair_feats = {}
    for nat_rel, tac_rel in tqdm(unique_pairs, leave=False):
        nat = Image.open(IMAGES_DIR / nat_rel).convert("RGB")
        tac = Image.open(IMAGES_DIR / tac_rel).convert("RGB")
        with torch.no_grad():
            fn = preprocess(nat).unsqueeze(0).to(device)
            ft = preprocess(tac).unsqueeze(0).to(device)
            en = clip_model.encode_image(fn); en = en / en.norm(dim=-1, keepdim=True)
            et = clip_model.encode_image(ft); et = et / et.norm(dim=-1, keepdim=True)
            feat = torch.cat([en, et, en - et], dim=-1)
        pair_feats[(nat_rel, tac_rel)] = feat.cpu()

    # Score per option
    results = {opt: {"probs": [], "labels": []} for opt in available_options}
    for dim, (head, opts) in heads.items():
        for r in known_records:
            if r["option_id"] not in opts:
                continue
            feat = pair_feats[(r["natural_image"], r["tactile_image"])].to(device)
            with torch.no_grad():
                probs = torch.sigmoid(head(feat)).cpu().numpy()[0]
            idx = opts.index(r["option_id"])
            results[r["option_id"]]["probs"].append(float(probs[idx]))
            results[r["option_id"]]["labels"].append(int(r["label"]))

    return results, skipped if 'skipped' in dir() else []


# ── Printing ──────────────────────────────────────────────────────────────────
def print_results(val_results, test_results, val_thresholds, skipped):
    W = 72
    evaluated = list(test_results.keys())
    print(f"\n{'='*W}")
    print(f"  CLIP Probe — data/splits_v2   ({len(evaluated)} options evaluated)")
    print(f"{'='*W}")

    # Balanced on test
    print(f"\n  TEST — balanced threshold (t = 0.50)")
    print(f"  {'Option':<22} {'F1':>6} {'AUC':>6} {'P':>6} {'R':>6} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'n_pos':>6}")
    print(f"  {'-'*W}")
    bal_f1s = []
    for opt in evaluated:
        m = compute_metrics(test_results[opt]["probs"],
                            test_results[opt]["labels"], 0.5)
        bal_f1s.append(m["f1"])
        print(f"  {opt:<22} {m['f1']:>6.3f} {m['auc']:>6.3f} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['n_pos']:>6}")
    print(f"  {'macro F1':<22} {np.mean(bal_f1s):>6.3f}")

    # Val-calibrated on test
    print(f"\n  TEST — val-calibrated thresholds")
    print(f"  {'Option':<22} {'F1':>6} {'AUC':>6} {'P':>6} {'R':>6} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'t_cal':>6}")
    print(f"  {'-'*W}")
    cal_f1s = []
    for opt in evaluated:
        t = val_thresholds[opt]
        m = compute_metrics(test_results[opt]["probs"],
                            test_results[opt]["labels"], t)
        cal_f1s.append(m["f1"])
        print(f"  {opt:<22} {m['f1']:>6.3f} {m['auc']:>6.3f} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {t:>6.2f}")
    print(f"  {'macro F1':<22} {np.mean(cal_f1s):>6.3f}")

    if skipped:
        print(f"\n  SKIPPED (no checkpoint found — run train_clip_heads_v2.sh first):")
        for opt in skipped:
            print(f"  {opt}")

    print(f"\n{'='*W}")
    return {
        "macro_f1_balanced":   round(float(np.mean(bal_f1s)), 4),
        "macro_f1_calibrated": round(float(np.mean(cal_f1s)), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Device:      {device}")
    print(f"Splits dir:  {SPLITS_DIR}")

    val_records  = load_split("val")
    test_records = load_split("test")
    print(f"Val:   {len(val_records)} records  "
          f"({len(set(r['pair_id'] for r in val_records))} pairs)")
    print(f"Test:  {len(test_records)} records  "
          f"({len(set(r['pair_id'] for r in test_records))} pairs)")

    # Score val and test
    print("\nScoring val split...")
    val_results,  skipped_v = score_with_clip(val_records,  device)
    print("\nScoring test split...")
    test_results, skipped_t = score_with_clip(test_records, device)
    skipped = list(set(skipped_v + skipped_t))

    # Calibrate thresholds on val
    val_thresholds = {}
    for opt in test_results:
        val_thresholds[opt] = best_threshold(
            val_results[opt]["probs"], val_results[opt]["labels"])

    # Print and save
    summary = print_results(val_results, test_results, val_thresholds, skipped)

    out = {
        "splits":          "data/splits_v2",
        "macro_f1_balanced":   summary["macro_f1_balanced"],
        "macro_f1_calibrated": summary["macro_f1_calibrated"],
        "val_thresholds":  val_thresholds,
        "per_option": {},
    }
    for opt in test_results:
        t = val_thresholds[opt]
        out["per_option"][opt] = {
            "balanced":    compute_metrics(test_results[opt]["probs"],
                                           test_results[opt]["labels"], 0.5),
            "calibrated":  compute_metrics(test_results[opt]["probs"],
                                           test_results[opt]["labels"], t),
            "val_threshold": t,
        }
    for opt in skipped:
        out["per_option"][opt] = {"status": "no_checkpoint_run_train_clip_heads_v2"}

    out_path = OUTPUT_DIR / "clip_eval_new_splits.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
