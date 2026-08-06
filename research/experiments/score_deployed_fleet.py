"""Measure the CURRENTLY DEPLOYED fleet as a fleet, on both frozen sets.

Why this exists: no single evaluation ever covered the deployed combination.
The v5 vision models were gated by gate_vision_candidates.py, which prints and
writes no JSON; every experiments_out/*.json still holds the PRE-v5 vision
scores. So the only ensemble figure on record (111/150) is old-vision + 7B,
not what ships. This closes that gap.

  vision (CLIP / DINOv2 / ResNet-50 / ViT-B/16)
      re-scored here from ~/TactileExpert/models/ — CPU, no GPU needed
  VLM (Qwen2-VL-7B + LoRA)
      per-pair scores lifted from experiments_out/eval_7b_gate.json; the
      adapter gated there is byte-identical (md5) to the deployed one, so
      re-running 7B inference would only reproduce it at GPU cost
  Ensemble
      computed through the SHIPPING code path — evaluators.constants
      build_result(mode="balanced") + build_ensemble_result — so the table
      reports what the app would actually decide, not a re-derivation

Outputs experiments_out/eval_current_fleet.json and prints README-ready tables.

Usage:  python research/experiments/score_deployed_fleet.py
"""

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

SPLITS = DATA_ROOT / "data" / "splits_v3_unified"
IMG_ROOT = DATA_ROOT / "images"
VLM_JSON = DATA_ROOT / "experiments_out" / "eval_7b_gate.json"
OUT = DATA_ROOT / "experiments_out" / "eval_current_fleet.json"

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
VISION = ["CLIP Probe", "DINOv2 Probe", "ResNet-50", "ViT-B/16"]
VLM = "VLM Fine-tuned"
MODELS = ["CLIP Probe", VLM, "ResNet-50", "ViT-B/16", "DINOv2 Probe"]


def prf(preds, labels):
    tp = sum(p and l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum((not p) and l for p, l in zip(preds, labels))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return prec, rec


def load_split(name, splits_dir=None):
    pairs = {}
    for line in ((splits_dir or SPLITS) / f"{name}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["option_id"] not in OPTS:
            continue
        p = pairs.setdefault(r["pair_id"], {
            "nat": str(IMG_ROOT / r["natural_image"]),
            "tac": str(IMG_ROOT / r["tactile_image"]),
            "labels": {}})
        p["labels"][r["option_id"]] = r["label"]
    return pairs


def f1(preds, labels):
    tp = sum(p and l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum((not p) and l for p, l in zip(preds, labels))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def auc(scores, labels):
    """Rank-based AUC with tie correction; nan when a class is absent."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, i = [0] * len(scores), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rsum = sum(r for r, l in zip(ranks, labels) if l)
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def score_vision(pairs_by_split, models_dir=None):
    """Score the four vision judges.

    models_dir points at a fleet directory laid out like TactileExpert/models/
    (clip_probe/, dino_probe/, resnet50/, vit_b16/). Passing it lets a freshly
    trained fleet be evaluated from its staging directory, so nothing has to be
    installed over the live one before the numbers are in.
    """
    from evaluators import (CLIPProbeEvaluator, DINOProbeEvaluator,
                            ResNetEvaluator, ViTEvaluator)
    md = Path(models_dir) if models_dir else None
    makers = {
        "CLIP Probe":   lambda: CLIPProbeEvaluator(
            models_dir=md / "clip_probe" if md else None, device="cpu"),
        "DINOv2 Probe": lambda: DINOProbeEvaluator(
            models_dir=md / "dino_probe" if md else None, device="cpu"),
        "ResNet-50":    lambda: ResNetEvaluator(
            checkpoint=md / "resnet50" / "resnet50_evaluator.pt" if md else None,
            device="cpu"),
        "ViT-B/16":     lambda: ViTEvaluator(
            checkpoint=md / "vit_b16" / "vit_b_16_evaluator.pt" if md else None,
            device="cpu"),
    }
    out = {sp: {} for sp in pairs_by_split}
    for name in VISION:
        print(f"\n=== {name} ===", flush=True)
        ev = makers[name]()
        for sp, pairs in pairs_by_split.items():
            got = {}
            for n, (pid, p) in enumerate(sorted(pairs.items()), 1):
                got[pid] = ev.score_pair(p["nat"], p["tac"])
                if n % 25 == 0:
                    print(f"    {sp}: {n}/{len(pairs)}", flush=True)
            out[sp][name] = got
        del ev
        gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--splits", nargs="+", default=["test"],
                    help="splits to score (v1 has no gpt_test — that regime is retired)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fleet", default="v1 fleet")
    ap.add_argument("--models-dir", default=None,
                    help="fleet directory to score (default: the deployed "
                         "TactileExpert/models/). Point at a staging dir to "
                         "evaluate a new fleet before installing it.")
    ap.add_argument("--vlm-json", default=None,
                    help="JSON holding per-pair VLM scores (default: %s)" % VLM_JSON.name)
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir) if args.splits_dir else SPLITS
    out_path = Path(args.out) if args.out else OUT

    pairs = {sp: load_split(sp, splits_dir) for sp in args.splits}
    print(f"labels: {splits_dir}")
    for sp, p in pairs.items():
        print(f"{sp}: {len(p)} pairs x {len(OPTS)} options = {len(p)*len(OPTS)} decisions")

    if args.models_dir:
        print(f"fleet:  {args.models_dir}")
    scores = score_vision(pairs, args.models_dir)

    vlm_json = Path(args.vlm_json) if args.vlm_json else VLM_JSON
    vlm_src = json.loads(vlm_json.read_text())["results"]
    for sp in pairs:
        scores[sp][VLM] = vlm_src[sp][VLM]
        missing = set(pairs[sp]) - set(scores[sp][VLM])
        if missing:
            raise SystemExit(f"{sp}: 7B scores missing for {len(missing)} pairs")

    from evaluators.constants import build_result, build_ensemble_result, ENSEMBLE_NAME

    def ensemble_probs(sp, pid):
        mr = {m: build_result(scores[sp][m][pid], {}, "balanced") for m in MODELS}
        ens = build_ensemble_result(mr, "balanced")
        return {o: ens["options"][o] for o in OPTS}

    # ── per-option precision / recall / F1 / AUC, per split ──────────────────
    # Never a single accuracy figure: under this class imbalance the always-clean
    # constant scores ~66%, so aggregate accuracy cannot distinguish a useful
    # judge from a degenerate one. That is why the v1 dataset exists.
    metrics_by_split = {}
    for sp in pairs:
        metrics = {}
        order = sorted(pairs[sp])
        for m in MODELS:
            metrics[m] = {}
            for o in OPTS:
                s = [scores[sp][m][pid][o] for pid in order]
                y = [pairs[sp][pid]["labels"][o] for pid in order]
                pred = [x > 0.5 for x in s]
                p_, r_ = prf(pred, y)
                metrics[m][o] = {"precision": p_, "recall": r_,
                                 "f1": f1(pred, y), "auc": auc(s, y)}
            metrics[m]["macro_f1"] = sum(metrics[m][o]["f1"] for o in OPTS) / len(OPTS)
            metrics[m]["macro_auc"] = sum(metrics[m][o]["auc"] for o in OPTS) / len(OPTS)

        ens = {pid: ensemble_probs(sp, pid) for pid in order}
        metrics[ENSEMBLE_NAME] = {}
        for o in OPTS:
            y = [pairs[sp][pid]["labels"][o] for pid in order]
            pred = [ens[pid][o]["flagged"] for pid in order]
            p_, r_ = prf(pred, y)
            metrics[ENSEMBLE_NAME][o] = {
                "precision": p_, "recall": r_, "f1": f1(pred, y),
                "auc": auc([ens[pid][o]["prob_yes"] for pid in order], y),
            }
        metrics[ENSEMBLE_NAME]["macro_f1"] = sum(
            metrics[ENSEMBLE_NAME][o]["f1"] for o in OPTS) / len(OPTS)
        metrics[ENSEMBLE_NAME]["macro_auc"] = sum(
            metrics[ENSEMBLE_NAME][o]["auc"] for o in OPTS) / len(OPTS)

        # the degenerate reference every judge has to clear
        metrics["always-clean baseline"] = {}
        for o in OPTS:
            y = [pairs[sp][pid]["labels"][o] for pid in order]
            metrics["always-clean baseline"][o] = {
                "precision": float("nan"), "recall": 0.0, "f1": 0.0,
                "auc": 0.5, "positive_rate": sum(y) / len(y) if y else 0.0}
        metrics_by_split[sp] = metrics
    metrics = metrics_by_split[args.splits[0]]
    metrics[ENSEMBLE_NAME]["macro_f1"] = sum(metrics[ENSEMBLE_NAME][o]["f1"] for o in OPTS) / len(OPTS)
    metrics[ENSEMBLE_NAME]["macro_auc"] = sum(metrics[ENSEMBLE_NAME][o]["auc"] for o in OPTS) / len(OPTS)

    out_path.write_text(json.dumps({
        "fleet": args.fleet,
        "splits_dir": str(splits_dir),
        "note": "vision scored on CPU from TactileExpert/models/; "
                "VLM per-pair scores reused from the VLM gate JSON",
        "n_pairs": {sp: len(p) for sp, p in pairs.items()},
        "metrics": metrics,
        "metrics_by_split": metrics_by_split,
        "results": scores,
    }, indent=2))

    hdr = MODELS + [ENSEMBLE_NAME]
    for sp in pairs:
        M = metrics_by_split[sp]
        n_dec = len(pairs[sp]) * len(OPTS)
        n_pos = sum(int(bool(p["labels"][o])) for p in pairs[sp].values() for o in OPTS)
        print(f"\n\n=== {sp.upper()} — {len(pairs[sp])} pairs, {n_dec} decisions, "
              f"{n_pos} positive ({n_pos/n_dec:.1%}) ===")
        print(f"    always-clean baseline gets {n_dec-n_pos}/{n_dec} "
              f"({1-n_pos/n_dec:.1%}) — beat this per option, not in aggregate")
        for metric in ("precision", "recall", "f1", "auc"):
            print(f"\n  -- {metric} @ t=0.50 --")
            print("| Dimension | " + " | ".join(hdr) + " |")
            for o in OPTS:
                print(f"| {o} | " + " | ".join(f"{M[m][o][metric]:.3f}" for m in hdr) + " |")
        print("\n| **Macro F1** | " + " | ".join(f"{M[m]['macro_f1']:.3f}" for m in hdr) + " |")
        print("| **Macro AUC** | " + " | ".join(f"{M[m]['macro_auc']:.3f}" for m in hdr) + " |")

    print(f"\nsaved -> {out_path}")
    print("Next: mcnemar_gate.py --gate <this file> --vs-always-clean --all-models")


if __name__ == "__main__":
    main()
