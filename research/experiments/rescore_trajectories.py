"""Re-score every trajectory image with the DEPLOYED fleet (P0-DPO fix).

The problem this closes: trajectories.jsonl stores the scores each fleet
logged at collection time, spanning several fleet generations. Policy
extraction then applied the CURRENT fleet's calibrators to those older raw
scores, so `reward_calibrated`, the `helped` flag, and the BC-helped count
were all derived from a distribution mismatch.

rederive_fleet.py's in-memory cache does NOT cover this: it scores finals
plus the pre-edit image of *targeted* edits (309 images), whereas policy
extraction needs prev+curr of *every* edit step (239 images). The two sets
overlap only partially — 84 of the 239 are never scored there. This script
scores the union so both consumers read one fleet.

Outputs experiments_out/trajectory_scores.json:
    {"<pair_dir>/<image>.jpg": {model: {option: prob}}}
keyed by the same relative path trajectories.jsonl uses, so consumers can
look scores up directly.

Also reports out-of-fold ECE per model on the finals, which the calibration
figures in the thesis need recomputed for the current fleet.

Run as a GPU job (7B in the fleet). Then re-run extract_policy_data.py.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

TRAJ = REPO_ROOT / "generated_training_data" / "trajectories.jsonl"
TRAIN_DIR = REPO_ROOT / "generated_training_data"
OUT = DATA_ROOT / "experiments_out" / "trajectory_scores.json"

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
MODELS = ["CLIP Probe", "VLM Fine-tuned", "ResNet-50", "ViT-B/16", "DINOv2 Probe"]
EPS = 1e-6


def needed_images(trajs):
    """Union of what policy extraction and weight derivation each require."""
    need, finals = set(), []
    for t in trajs:
        ts = t["trajectory_id"].replace("traj_", "")
        pdir = TRAIN_DIR / f"pair_{ts}"
        steps = t["steps"]

        final = pdir / "tactile.jpg"
        if final.exists():
            need.add(final)
            finals.append((t, final))

        for si, s in enumerate(steps):
            # policy extraction reads prev+curr of every real edit step
            instr = (s.get("instruction") or "").strip()
            if si >= 1 and instr and not instr.startswith("[generated]"):
                for k in (si - 1, si):
                    p = pdir / f"tactile_iter{k}.jpg"
                    if p.exists():
                        need.add(p)
            # weight derivation reads the pre-edit image of targeted edits
            if si >= 1 and s.get("target_option") in OPTS:
                p = pdir / f"tactile_iter{si-1}.jpg"
                if p.exists():
                    need.add(p)
    return sorted(need), finals


def ece(probs, labels, n_bins=10):
    """Expected calibration error, equal-width bins."""
    if not probs:
        return float("nan")
    tot, err = len(probs), 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs)
               if (p > lo or (b == 0 and p >= lo)) and p <= hi]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        err += (len(idx) / tot) * abs(conf - acc)
    return err


def main():
    trajs = [json.loads(l) for l in TRAJ.read_text().splitlines()]
    imgs, finals = needed_images(trajs)
    print(f"{len(trajs)} trajectories -> {len(imgs)} distinct images to score", flush=True)

    from evaluators import (CLIPProbeEvaluator, VLMEvaluator, ResNetEvaluator,
                            ViTEvaluator, DINOProbeEvaluator)
    print("loading deployed fleet (7B VLM — GPU)…", flush=True)
    ev = {
        "CLIP Probe": CLIPProbeEvaluator(),
        "DINOv2 Probe": DINOProbeEvaluator(),
        "ResNet-50": ResNetEvaluator(),
        "ViT-B/16": ViTEvaluator(),
        "VLM Fine-tuned": VLMEvaluator(),
    }

    scores = {}
    for n, img in enumerate(imgs, 1):
        nat = img.parent / "natural.jpg"
        if not nat.exists():
            print(f"  SKIP (no natural.jpg): {img}", flush=True)
            continue
        key = str(img.relative_to(TRAIN_DIR))
        scores[key] = {m: ev[m].score_pair(str(nat), str(img)) for m in MODELS}
        if n % 20 == 0 or n == len(imgs):
            print(f"  {n}/{len(imgs)}", flush=True)

    # ── ECE on the finals, against the human checkbox labels ────────────────
    per_model = {m: {"p": [], "y": []} for m in MODELS}
    for t, final in finals:
        key = str(final.relative_to(TRAIN_DIR))
        if key not in scores:
            continue
        for m in MODELS:
            for o in OPTS:
                per_model[m]["p"].append(scores[key][m][o])
                per_model[m]["y"].append(int(t["human_labels"][o]))
    raw_ece = {m: round(ece(per_model[m]["p"], per_model[m]["y"]), 4) for m in MODELS}
    base_rate = (sum(per_model[MODELS[0]]["y"]) / len(per_model[MODELS[0]]["y"])
                 if per_model[MODELS[0]]["y"] else float("nan"))

    OUT.write_text(json.dumps({
        "fleet": "CLIP/DINOv2/ResNet/ViT v5 + Qwen2-VL-7B (deployed 2026-07-27)",
        "n_trajectories": len(trajs),
        "n_images": len(scores),
        "finals_defect_base_rate": round(base_rate, 4),
        "raw_ece_on_finals": raw_ece,
        "scores": scores,
    }, indent=2))

    print(f"\nscored {len(scores)} images -> {OUT}")
    print(f"defect base rate on finals: {base_rate:.1%}")
    print("raw ECE per model (uncalibrated, current fleet):")
    for m, v in raw_ece.items():
        print(f"  {m:<16} {v}")
    print("\nNEXT: point extract_policy_data.py at this file, then re-run it.")


if __name__ == "__main__":
    main()
