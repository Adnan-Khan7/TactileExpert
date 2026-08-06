#!/usr/bin/env python3
"""
Evaluate CLIP Probe, ResNet-50, and ViT-B/16 on the same VQA test split
used for VLM Fine-tuned evaluation, so all five models are comparable on
identical data.

Outputs:
  output/baseline_test_metrics.json   per-model per-option metrics
  (prints comparison table with VLM results from output/test_metrics.json)

Usage:
  python code/baselines/eval_baselines.py
  python code/baselines/eval_baselines.py --device cpu
  python code/baselines/eval_baselines.py --models clip resnet50 vit
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

try:
    from sklearn.metrics import (f1_score, roc_auc_score,
                                  precision_score, recall_score)
except ImportError:
    raise ImportError("pip install scikit-learn")

try:
    import open_clip
except ImportError:
    raise ImportError("pip install open-clip-torch")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = DATA_ROOT
IMAGES_DIR = ROOT / "images"
VQA_TEST   = ROOT / "data" / "vqa" / "test.jsonl"
MODELS_DIR = ROOT / "TactileEval_Context_And_Data" / "models"
OUTPUT_DIR = ROOT / "output"
VLM_METRICS = OUTPUT_DIR / "test_metrics.json"

# ---------------------------------------------------------------------------
# Option / dimension layout  (must match training order)
# ---------------------------------------------------------------------------
TASK_HEADS = {
    "QL": ["too_thick", "broken_lines"],
    "QP": ["missing_parts"],
    "QT": ["missing_texture"],
}
ALL_OPTIONS = [opt for opts in TASK_HEADS.values() for opt in opts]
# ALL_OPTIONS = ["too_thick", "broken_lines", "missing_parts", "missing_texture"]

# ---------------------------------------------------------------------------
# CLIP probe architecture  (mirrors step2_train_stable_evaluator.py)
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "laion2b_s32b_b82k"
FEAT_DIM  = 768
INPUT_DIM = FEAT_DIM * 3   # 2304


def build_clip_head(n_out: int, hidden: int = 512) -> nn.Module:
    return nn.Sequential(
        nn.Linear(INPUT_DIM, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, n_out),
    )


# ---------------------------------------------------------------------------
# ResNet-50 / ViT-B/16 architecture  (mirrors step3_train_baseline.py)
# ---------------------------------------------------------------------------
def get_transforms() -> T.Compose:
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std= [0.229, 0.224, 0.225]),
    ])


class TactileEvaluator(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        if backbone_name == "resnet50":
            base     = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = base.fc.in_features
            base.fc  = nn.Identity()
            self.backbone = base
        elif backbone_name == "vit_b_16":
            base     = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
            feat_dim = base.heads.head.in_features
            base.heads = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        combined = feat_dim * 3

        def head(n_out):
            return nn.Sequential(
                nn.Linear(combined, 256), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(256, n_out),
            )

        self.heads = nn.ModuleDict(
            {dim: head(len(opts)) for dim, opts in TASK_HEADS.items()}
        )

    def forward(self, nat, tac):
        f_nat  = self.backbone(nat)
        f_tac  = self.backbone(tac)
        feat   = torch.cat([f_nat, f_tac, f_nat - f_tac], dim=-1)
        return torch.cat(
            [self.heads[dim](feat) for dim in ["QL", "QP", "QT"]], dim=-1
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_vqa_test() -> list[dict]:
    with open(VQA_TEST) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(probs: list[float], labels: list[int],
                    threshold: float = 0.5) -> dict:
    y_true = np.array(labels)
    y_prob = np.array(probs)
    y_pred = (y_prob >= threshold).astype(int)

    f1  = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        auc = 0.0

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    return dict(
        f1=round(f1, 4), auc=round(auc, 4),
        precision=round(pre, 4), recall=round(rec, 4),
        tp=tp, fp=fp, fn=fn, tn=tn,
        n_pos=int(y_true.sum()), n_total=len(y_true),
        threshold=threshold,
    )


def metrics_for_results(opt_results: dict[str, dict],
                         cal_thresholds: dict[str, float] | None = None) -> dict:
    per_option = {}
    macro_f1s  = []
    for opt, data in opt_results.items():
        t_bal = 0.5
        t_cal = (cal_thresholds or {}).get(opt, 0.5)

        m_bal = compute_metrics(data["probs"], data["labels"], threshold=t_bal)
        m_cal = compute_metrics(data["probs"], data["labels"], threshold=t_cal)

        per_option[opt] = {
            "balanced":   m_bal,
            "calibrated": m_cal,
            "cal_threshold": t_cal,
        }
        macro_f1s.append(m_bal["f1"])

    return {
        "per_option":        per_option,
        "macro_f1_balanced": round(float(np.mean(macro_f1s)) if macro_f1s else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# CLIP Probe evaluation
# ---------------------------------------------------------------------------
def eval_clip_probe(records: list[dict], device: torch.device) -> tuple[dict, dict]:
    print("\n[CLIP Probe] Loading model...")
    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # Load per-dimension heads and calibrated thresholds
    heads: dict[str, tuple[nn.Module, list[str], dict[str, float]]] = {}
    for dim, opts in TASK_HEADS.items():
        ckpt_path = MODELS_DIR / "clip_probe" / f"{dim.lower()}_evaluator.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        head = build_clip_head(len(opts), ckpt.get("hidden_dim", 512))
        head.load_state_dict(ckpt["state_dict"])
        head.eval().to(device)

        cal_t = {}
        for opt in opts:
            cal_t[opt] = ckpt["test_results"]["per_option"][opt].get(
                "threshold_calibrated", 0.5)
        heads[dim] = (head, opts, cal_t)

    # Extract CLIP features for all unique pairs
    unique_pairs = {(r["natural_image"], r["tactile_image"]) for r in records}
    print(f"[CLIP Probe] Extracting features for {len(unique_pairs)} unique pairs...")
    pair_feats: dict[tuple, torch.Tensor] = {}
    for nat_rel, tac_rel in tqdm(unique_pairs, leave=False):
        nat_img = Image.open(IMAGES_DIR / nat_rel).convert("RGB")
        tac_img = Image.open(IMAGES_DIR / tac_rel).convert("RGB")
        with torch.no_grad():
            fn = preprocess(nat_img).unsqueeze(0).to(device)
            ft = preprocess(tac_img).unsqueeze(0).to(device)
            en = clip_model.encode_image(fn)
            en = en / en.norm(dim=-1, keepdim=True)
            et = clip_model.encode_image(ft)
            et = et / et.norm(dim=-1, keepdim=True)
            feat = torch.cat([en, et, en - et], dim=-1)
        pair_feats[(nat_rel, tac_rel)] = feat.cpu()

    # Run heads per record
    opt_results: dict[str, dict] = {opt: {"probs": [], "labels": []} for opt in ALL_OPTIONS}
    all_cal_t: dict[str, float] = {}
    for dim, (head, opts, cal_t) in heads.items():
        all_cal_t.update(cal_t)
        for r in records:
            if r["option_id"] not in opts:
                continue
            feat = pair_feats[(r["natural_image"], r["tactile_image"])].to(device)
            with torch.no_grad():
                probs = torch.sigmoid(head(feat)).cpu().numpy()[0]
            idx = opts.index(r["option_id"])
            opt_results[r["option_id"]]["probs"].append(float(probs[idx]))
            opt_results[r["option_id"]]["labels"].append(int(r["label"]))

    # Remove options with no records (shouldn't happen but be safe)
    opt_results = {k: v for k, v in opt_results.items() if v["probs"]}
    return opt_results, all_cal_t


# ---------------------------------------------------------------------------
# ResNet-50 / ViT-B/16 evaluation
# ---------------------------------------------------------------------------
def eval_backbone_model(backbone_name: str, records: list[dict],
                         device: torch.device) -> dict:
    ckpt_dir  = "resnet50_twostream" if backbone_name == "resnet50" else "vit_b16_twostream"
    ckpt_file = f"{backbone_name}_evaluator.pt"
    ckpt_path = MODELS_DIR / ckpt_dir / ckpt_file

    print(f"\n[{backbone_name}] Loading checkpoint from {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = TactileEvaluator(backbone_name)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    transform = get_transforms()

    # Group records by unique (nat, tac) pair
    pair_to_records: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["natural_image"], r["tactile_image"])
        pair_to_records.setdefault(key, []).append(r)

    opt_results: dict[str, dict] = {opt: {"probs": [], "labels": []} for opt in ALL_OPTIONS}

    print(f"[{backbone_name}] Evaluating {len(pair_to_records)} unique pairs...")
    for (nat_rel, tac_rel), recs in tqdm(pair_to_records.items(), leave=False):
        nat_img = Image.open(IMAGES_DIR / nat_rel).convert("RGB")
        tac_img = Image.open(IMAGES_DIR / tac_rel).convert("RGB")
        nat = transform(nat_img).unsqueeze(0).to(device)
        tac = transform(tac_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(nat, tac)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]  # (4,)

        for r in recs:
            opt = r["option_id"]
            idx = ALL_OPTIONS.index(opt)
            opt_results[opt]["probs"].append(float(probs[idx]))
            opt_results[opt]["labels"].append(int(r["label"]))

    opt_results = {k: v for k, v in opt_results.items() if v["probs"]}
    return opt_results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_model_metrics(model_name: str, metrics: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  {model_name}   (macro F1 balanced: {metrics['macro_f1_balanced']:.4f})")
    print(f"{'='*70}")
    print(f"  {'Option':<22} {'F1':>6} {'AUC':>6} {'P':>6} {'R':>6} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'n_pos':>6}  [balanced t=0.50]")
    print(f"  {'-'*80}")
    for opt, data in metrics["per_option"].items():
        m = data["balanced"]
        print(f"  {opt:<22} {m['f1']:>6.3f} {m['auc']:>6.3f} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['n_pos']:>6}")

    print(f"\n  {'Option':<22} {'F1':>6} {'AUC':>6} {'P':>6} {'R':>6} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'t_cal':>6}  [calibrated]")
    print(f"  {'-'*80}")
    for opt, data in metrics["per_option"].items():
        m   = data["calibrated"]
        t   = data["cal_threshold"]
        print(f"  {opt:<22} {m['f1']:>6.3f} {m['auc']:>6.3f} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {t:>6.2f}")


def print_comparison(all_metrics: dict) -> None:
    print(f"\n{'='*70}")
    print("  COMPARISON TABLE — AUC per option (balanced threshold)")
    print(f"{'='*70}")
    models = list(all_metrics.keys())
    print(f"  {'Option':<22} " + "  ".join(f"{m:<14}" for m in models))
    print(f"  {'-'*70}")
    for opt in ALL_OPTIONS:
        row = f"  {opt:<22} "
        for m in models:
            auc = all_metrics[m]["per_option"].get(opt, {}).get(
                "balanced", {}).get("auc", float("nan"))
            row += f"  {auc:<14.3f}"
        print(row)

    print(f"\n  {'':22} " + "  ".join(f"{m:<14}" for m in models))
    print(f"  {'macro F1 (balanced)':<22} " +
          "  ".join(f"  {all_metrics[m]['macro_f1_balanced']:<12.4f}" for m in models))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models",  nargs="+",
                        choices=["clip", "resnet50", "vit"],
                        default=["clip", "resnet50", "vit"],
                        help="Which models to evaluate (default: all)")
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"VQA test split: {VQA_TEST}")
    records = load_vqa_test()
    print(f"Loaded {len(records)} test records across "
          f"{len(set(r['option_id'] for r in records))} options\n")

    all_metrics: dict[str, dict] = {}

    # Load VLM results for comparison
    if VLM_METRICS.exists():
        vlm_raw = json.load(open(VLM_METRICS))
        vlm_metrics: dict = {"per_option": {}, "macro_f1_balanced": 0.0}
        f1s = []
        for opt, data in vlm_raw["per_option"].items():
            t_cal = data.get("threshold", 0.5)
            vlm_metrics["per_option"][opt] = {
                "balanced":      {"f1": 0.0, "auc": data["auc"],
                                   "precision": 0.0, "recall": 0.0,
                                   "tp": 0, "fp": 0, "fn": 0,
                                   "n_pos": data["n_pos"], "n_total": data["n_total"]},
                "calibrated":    {"f1": data["f1"], "auc": data["auc"],
                                   "precision": 0.0, "recall": 0.0,
                                   "tp": 0, "fp": 0, "fn": 0,
                                   "n_pos": data["n_pos"], "n_total": data["n_total"]},
                "cal_threshold": t_cal,
            }
            f1s.append(data["f1"])
        vlm_metrics["macro_f1_balanced"] = round(float(np.mean(f1s)) if f1s else 0.0, 4)
        all_metrics["VLM Fine-tuned"] = vlm_metrics
        print("Loaded VLM Fine-tuned results from output/test_metrics.json")

    # CLIP Probe
    if "clip" in args.models:
        opt_results, cal_t = eval_clip_probe(records, device)
        clip_metrics = metrics_for_results(opt_results, cal_t)
        all_metrics["CLIP Probe"] = clip_metrics
        print_model_metrics("CLIP Probe", clip_metrics)

    # ResNet-50
    if "resnet50" in args.models:
        opt_results = eval_backbone_model("resnet50", records, device)
        resnet_metrics = metrics_for_results(opt_results)
        all_metrics["ResNet-50"] = resnet_metrics
        print_model_metrics("ResNet-50", resnet_metrics)

    # ViT-B/16
    if "vit" in args.models:
        opt_results = eval_backbone_model("vit_b_16", records, device)
        vit_metrics = metrics_for_results(opt_results)
        all_metrics["ViT-B/16"] = vit_metrics
        print_model_metrics("ViT-B/16", vit_metrics)

    # Comparison table
    if len(all_metrics) > 1:
        print_comparison(all_metrics)

    # Save
    out_path = OUTPUT_DIR / "baseline_test_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
