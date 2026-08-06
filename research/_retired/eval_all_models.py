#!/usr/bin/env python3
"""
Evaluate all retrained models on data/splits_v2 test set.

Models (v2 — trained on new 987-pair manually-annotated dataset, 6 options):
  CLIP Probe    — 5 heads (QL/QP/QT/QE/QN), models/clip_probe_v2/
  ResNet-50     — 5 heads, models/resnet50_v2/
  ViT-B/16      — 5 heads, models/vitb16_v2/
  VLM (Qwen)    — checked but skipped if checkpoint not yet ready

Protocol: calibrate thresholds on val, evaluate on test (frozen thresholds).

Usage:
  sbatch code/baselines/eval_all_models.sh
  # or interactively on a GPU node:
  python code/baselines/eval_all_models.py
  python code/baselines/eval_all_models.py --device cpu
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
import torchvision.models as tvm
import torchvision.transforms as T
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

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = DATA_ROOT
SPLITS_DIR = ROOT / "data" / "splits_v2"
IMAGES_DIR = ROOT / "images"
MODELS_DIR = ROOT / "TactileEval_Context_And_Data" / "models"
OUTPUT_DIR = ROOT / "output"

# ── Option / head layout ──────────────────────────────────────────────────────
TASK_HEADS = {
    "QL": ["too_thick",   "broken_lines"],
    "QP": ["missing_parts"],
    "QT": ["missing_texture"],
    "QE": ["extra_parts"],
    "QN": ["non_conformant"],
}
HEAD_ORDER  = ["QL", "QP", "QT", "QE", "QN"]
ALL_OPTIONS = [o for opts in TASK_HEADS.values() for o in opts]

CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "laion2b_s32b_b82k"
FEAT_DIM        = 768
CLIP_INPUT_DIM  = FEAT_DIM * 3


# ── Data ──────────────────────────────────────────────────────────────────────
def load_split(name: str) -> list[dict]:
    with open(SPLITS_DIR / f"{name}.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Metrics ───────────────────────────────────────────────────────────────────
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
                n_pos=int(y_true.sum()), n_total=len(y_true))


def best_threshold(probs, labels) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(labels, (np.array(probs) >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2)


def calibrate_and_evaluate(opt_scores: dict, val_records, test_records) -> dict:
    """Given raw probs/labels per option, calibrate on val, report on test."""
    val_thresholds = {}
    for opt in opt_scores["val"]:
        val_thresholds[opt] = best_threshold(
            opt_scores["val"][opt]["probs"],
            opt_scores["val"][opt]["labels"])

    results = {"per_option": {}, "val_thresholds": val_thresholds}
    bal_f1s, cal_f1s = [], []
    for opt in ALL_OPTIONS:
        if opt not in opt_scores["test"]:
            results["per_option"][opt] = {"status": "no_checkpoint"}
            continue
        d = opt_scores["test"][opt]
        t = val_thresholds.get(opt, 0.5)
        results["per_option"][opt] = {
            "balanced":    compute_metrics(d["probs"], d["labels"], 0.5),
            "calibrated":  compute_metrics(d["probs"], d["labels"], t),
            "val_threshold": t,
        }
        bal_f1s.append(results["per_option"][opt]["balanced"]["f1"])
        cal_f1s.append(results["per_option"][opt]["calibrated"]["f1"])

    results["macro_f1_balanced"]   = round(float(np.mean(bal_f1s)) if bal_f1s else 0, 4)
    results["macro_f1_calibrated"] = round(float(np.mean(cal_f1s)) if cal_f1s else 0, 4)
    return results


# ── CLIP Probe ────────────────────────────────────────────────────────────────
def clip_head(n_out, hidden=512):
    return nn.Sequential(
        nn.Linear(CLIP_INPUT_DIM, hidden), nn.BatchNorm1d(hidden),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, n_out))


def score_clip(records: list[dict], device: torch.device) -> dict[str, dict]:
    clip_dir = MODELS_DIR / "clip_probe_v2"
    heads = {}
    for dim, opts in TASK_HEADS.items():
        ckpt_path = clip_dir / f"{dim.lower()}_evaluator.pt"
        if not ckpt_path.exists():
            print(f"  CLIP [{dim}] missing — skipping {opts}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        head = clip_head(len(opts), ckpt.get("hidden_dim", 512))
        head.load_state_dict(ckpt["state_dict"])
        head.eval().to(device)
        heads[dim] = (head, opts)

    available = [o for opts in [h[1] for h in heads.values()] for o in opts]
    recs = [r for r in records if r["option_id"] in available]

    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device)
    clip_model.eval()
    for p in clip_model.parameters(): p.requires_grad_(False)

    unique_pairs = {(r["natural_image"], r["tactile_image"]) for r in recs}
    pair_feats   = {}
    for nat, tac in tqdm(unique_pairs, desc="  CLIP features", leave=False):
        n = preprocess(Image.open(IMAGES_DIR/nat).convert("RGB")).unsqueeze(0).to(device)
        t = preprocess(Image.open(IMAGES_DIR/tac).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            en = clip_model.encode_image(n); en /= en.norm(dim=-1, keepdim=True)
            et = clip_model.encode_image(t); et /= et.norm(dim=-1, keepdim=True)
            pair_feats[(nat, tac)] = torch.cat([en, et, en-et], dim=-1).cpu()

    scores = {opt: {"probs": [], "labels": []} for opt in available}
    for dim, (head, opts) in heads.items():
        for r in recs:
            if r["option_id"] not in opts: continue
            feat = pair_feats[(r["natural_image"], r["tactile_image"])].to(device)
            with torch.no_grad():
                probs = torch.sigmoid(head(feat)).cpu().numpy()[0]
            scores[r["option_id"]]["probs"].append(float(probs[opts.index(r["option_id"])]))
            scores[r["option_id"]]["labels"].append(int(r["label"]))
    return scores


# ── ResNet-50 / ViT-B/16 ──────────────────────────────────────────────────────
def build_backbone(name: str) -> nn.Module:
    if name == "resnet50":
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = base.fc.in_features; base.fc = nn.Identity()
    else:
        base = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        feat_dim = base.heads.head.in_features; base.heads = nn.Identity()

    combined = feat_dim * 3
    def head(n): return nn.Sequential(
        nn.Linear(combined, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = base
            self.heads = nn.ModuleDict({d: head(len(opts))
                                        for d, opts in TASK_HEADS.items()})
        def forward(self, nat, tac):
            fn = self.backbone(nat); ft = self.backbone(tac)
            feat = torch.cat([fn, ft, fn-ft], dim=-1)
            return torch.cat([self.heads[d](feat) for d in HEAD_ORDER], dim=-1)
    return Model()


def score_backbone(backbone_name: str, records: list[dict],
                   device: torch.device) -> dict[str, dict]:
    slug     = backbone_name.replace("_", "")
    ckpt_dir = MODELS_DIR / f"{slug}_v2"
    ckpt_path = ckpt_dir / f"{backbone_name}_evaluator.pt"
    if not ckpt_path.exists():
        print(f"  {backbone_name}: no checkpoint at {ckpt_path}")
        return {}

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_backbone(backbone_name)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    pair_to_recs: dict[tuple, list] = {}
    for r in records:
        pair_to_recs.setdefault((r["natural_image"], r["tactile_image"]), []).append(r)

    scores = {opt: {"probs": [], "labels": []} for opt in ALL_OPTIONS}
    for (nat, tac), recs in tqdm(pair_to_recs.items(),
                                  desc=f"  {backbone_name}", leave=False):
        n = transform(Image.open(IMAGES_DIR/nat).convert("RGB")).unsqueeze(0).to(device)
        t = transform(Image.open(IMAGES_DIR/tac).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(n, t)).cpu().numpy()[0]
        for r in recs:
            idx = ALL_OPTIONS.index(r["option_id"])
            scores[r["option_id"]]["probs"].append(float(probs[idx]))
            scores[r["option_id"]]["labels"].append(int(r["label"]))
    return {k: v for k, v in scores.items() if v["probs"]}


# ── Print comparison table ────────────────────────────────────────────────────
def print_table(all_results: dict):
    models = list(all_results.keys())
    W = 24 + 12 * len(models)
    print(f"\n{'='*W}")
    print(f"  COMPARISON — AUC (balanced t=0.50) on splits_v2 test")
    print(f"{'='*W}")
    header = f"  {'Option':<22}" + "".join(f"  {m:<10}" for m in models)
    print(header)
    print(f"  {'-'*W}")

    for opt in ALL_OPTIONS:
        row = f"  {opt:<22}"
        for m in models:
            per = all_results[m].get("per_option", {}).get(opt, {})
            if "balanced" in per:
                row += f"  {per['balanced']['auc']:<10.3f}"
            else:
                row += f"  {'—':<10}"
        print(row)

    print(f"\n  {'macro F1 (balanced)':<22}" +
          "".join(f"  {all_results[m].get('macro_f1_balanced', 0):<10.4f}"
                  for m in models))
    print(f"  {'macro F1 (calibrated)':<22}" +
          "".join(f"  {all_results[m].get('macro_f1_calibrated', 0):<10.4f}"
                  for m in models))

    # Per-model detail
    for m, res in all_results.items():
        print(f"\n{'='*W}")
        print(f"  {m}   balanced macro F1={res.get('macro_f1_balanced',0):.4f}"
              f"   calibrated macro F1={res.get('macro_f1_calibrated',0):.4f}")
        print(f"{'='*W}")
        print(f"  {'Option':<22} {'F1':>6} {'AUC':>6} {'P':>6} {'R':>6}"
              f" {'TP':>4} {'FP':>4} {'FN':>4} {'n+':>5} {'t_cal':>6}")
        print(f"  {'-'*72}")
        for opt in ALL_OPTIONS:
            per = res.get("per_option", {}).get(opt, {})
            if "balanced" not in per:
                print(f"  {opt:<22} — (no checkpoint)")
                continue
            b = per["balanced"]
            c = per["calibrated"]
            t = per["val_threshold"]
            print(f"  {opt:<22} {b['f1']:>6.3f} {b['auc']:>6.3f} "
                  f"{b['precision']:>6.3f} {b['recall']:>6.3f}"
                  f" {b['tp']:>4} {b['fp']:>4} {b['fn']:>4}"
                  f" {b['n_pos']:>5} {t:>6.2f}  (cal F1={c['f1']:.3f})")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+",
                        choices=["clip", "resnet50", "vit_b_16"],
                        default=["clip", "resnet50", "vit_b_16"])
    args  = parser.parse_args()
    device = torch.device(args.device)

    print(f"Device: {device}  |  Splits: {SPLITS_DIR}")
    val_records  = load_split("val")
    test_records = load_split("test")
    print(f"Val:  {len(val_records)} records ({len(set(r['pair_id'] for r in val_records))} pairs)")
    print(f"Test: {len(test_records)} records ({len(set(r['pair_id'] for r in test_records))} pairs)")

    all_results = {}

    if "clip" in args.models:
        print("\n── CLIP Probe v2 ──")
        val_scores  = score_clip(val_records,  device)
        test_scores = score_clip(test_records, device)
        all_results["CLIP Probe"] = calibrate_and_evaluate(
            {"val": val_scores, "test": test_scores}, val_records, test_records)

    if "resnet50" in args.models:
        print("\n── ResNet-50 v2 ──")
        val_scores  = score_backbone("resnet50", val_records,  device)
        test_scores = score_backbone("resnet50", test_records, device)
        if test_scores:
            all_results["ResNet-50"] = calibrate_and_evaluate(
                {"val": val_scores, "test": test_scores}, val_records, test_records)

    if "vit_b_16" in args.models:
        print("\n── ViT-B/16 v2 ──")
        val_scores  = score_backbone("vit_b_16", val_records,  device)
        test_scores = score_backbone("vit_b_16", test_records, device)
        if test_scores:
            all_results["ViT-B/16"] = calibrate_and_evaluate(
                {"val": val_scores, "test": test_scores}, val_records, test_records)

    print_table(all_results)

    out_path = OUTPUT_DIR / "eval_all_models_v2.json"
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
