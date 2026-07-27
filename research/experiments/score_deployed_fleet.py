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

import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

SPLITS = DATA_ROOT / "data" / "splits_v2"
IMG_ROOT = DATA_ROOT / "images"
VLM_JSON = DATA_ROOT / "experiments_out" / "eval_7b_gate.json"
OUT = DATA_ROOT / "experiments_out" / "eval_current_fleet.json"

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
VISION = ["CLIP Probe", "DINOv2 Probe", "ResNet-50", "ViT-B/16"]
VLM = "VLM Fine-tuned"
MODELS = ["CLIP Probe", VLM, "ResNet-50", "ViT-B/16", "DINOv2 Probe"]


def load_split(name):
    pairs = {}
    for line in (SPLITS / f"{name}.jsonl").read_text().splitlines():
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


def score_vision(pairs_by_split):
    from evaluators import (CLIPProbeEvaluator, DINOProbeEvaluator,
                            ResNetEvaluator, ViTEvaluator)
    makers = {
        "CLIP Probe":   lambda: CLIPProbeEvaluator(device="cpu"),
        "DINOv2 Probe": lambda: DINOProbeEvaluator(device="cpu"),
        "ResNet-50":    lambda: ResNetEvaluator(device="cpu"),
        "ViT-B/16":     lambda: ViTEvaluator(device="cpu"),
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
    pairs = {"test": load_split("test"), "gpt_test": load_split("gpt_test")}
    for sp, p in pairs.items():
        print(f"{sp}: {len(p)} pairs x {len(OPTS)} options = {len(p)*len(OPTS)} decisions")

    scores = score_vision(pairs)

    vlm_src = json.loads(VLM_JSON.read_text())["results"]
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

    # ── original test: per-option F1 + AUC ────────────────────────────────────
    metrics = {}
    order = sorted(pairs["test"])
    for m in MODELS:
        metrics[m] = {}
        for o in OPTS:
            s = [scores["test"][m][pid][o] for pid in order]
            y = [pairs["test"][pid]["labels"][o] for pid in order]
            metrics[m][o] = {"f1": f1([x > 0.5 for x in s], y), "auc": auc(s, y)}
        metrics[m]["macro_f1"] = sum(metrics[m][o]["f1"] for o in OPTS) / len(OPTS)
        metrics[m]["macro_auc"] = sum(metrics[m][o]["auc"] for o in OPTS) / len(OPTS)

    ens_test = {pid: ensemble_probs("test", pid) for pid in order}
    metrics[ENSEMBLE_NAME] = {}
    for o in OPTS:
        y = [pairs["test"][pid]["labels"][o] for pid in order]
        metrics[ENSEMBLE_NAME][o] = {
            "f1": f1([ens_test[pid][o]["flagged"] for pid in order], y),
            "auc": auc([ens_test[pid][o]["prob_yes"] for pid in order], y),
        }
    metrics[ENSEMBLE_NAME]["macro_f1"] = sum(metrics[ENSEMBLE_NAME][o]["f1"] for o in OPTS) / len(OPTS)
    metrics[ENSEMBLE_NAME]["macro_auc"] = sum(metrics[ENSEMBLE_NAME][o]["auc"] for o in OPTS) / len(OPTS)

    # ── GPT holdout: correct flag decisions ──────────────────────────────────
    hold, n_hold = {}, len(pairs["gpt_test"]) * len(OPTS)
    for m in MODELS:
        hold[m] = sum(int((scores["gpt_test"][m][pid][o] > 0.5) == bool(p["labels"][o]))
                      for pid, p in pairs["gpt_test"].items() for o in OPTS)
    hold[ENSEMBLE_NAME] = sum(
        int(ensemble_probs("gpt_test", pid)[o]["flagged"] == bool(p["labels"][o]))
        for pid, p in pairs["gpt_test"].items() for o in OPTS)
    n_pos = sum(int(bool(p["labels"][o])) for p in pairs["gpt_test"].values() for o in OPTS)
    hold["always-clean baseline"] = n_hold - n_pos

    OUT.write_text(json.dumps({
        "fleet": "CLIP/DINOv2/ResNet/ViT v5 + Qwen2-VL-7B (deployed 2026-07-27)",
        "note": "vision re-scored on CPU from TactileExpert/models/; "
                "VLM per-pair scores reused from eval_7b_gate.json (identical adapter md5)",
        "n_test_pairs": len(pairs["test"]), "n_holdout_decisions": n_hold,
        "holdout_positives": n_pos,
        "metrics": metrics, "holdout_correct": hold,
        "results": scores,
    }, indent=2))

    hdr = MODELS + [ENSEMBLE_NAME]
    print("\n\n=== ORIGINAL TEST (153 pairs) — F1 @ t=0.50 ===")
    print("| Dimension | " + " | ".join(hdr) + " |")
    for o in OPTS:
        print(f"| {o} | " + " | ".join(f"{metrics[m][o]['f1']:.3f}" for m in hdr) + " |")
    print("| **Macro F1** | " + " | ".join(f"{metrics[m]['macro_f1']:.3f}" for m in hdr) + " |")
    print("| **Macro AUC** | " + " | ".join(f"{metrics[m]['macro_auc']:.3f}" for m in hdr) + " |")

    print(f"\n=== GPT HOLDOUT ({len(pairs['gpt_test'])} pairs, {n_hold} decisions, "
          f"{n_pos} positive) — correct decisions ===")
    for k, v in sorted(hold.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24} {v}/{n_hold}  ({v/n_hold:.1%})")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
