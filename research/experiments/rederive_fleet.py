"""Fit the ensemble weights and per-model probability calibrators.

Scores the trajectory finals (label = human checkbox) and the inferred pre-edit
positives (edit-step target_option, label=1) with the fleet in
~/TactileExpert/models/, then derives:

  * ENSEMBLE_WEIGHTS          -> paste into evaluators/constants.py
  * experiments_out/fleet_calibration.json  (per-model temperature or Platt,
    whichever wins on out-of-fold NLL)

Needs a GPU: the VLM is 7B.

TRAIN-SPLIT SESSIONS ONLY
-------------------------
Weights and calibrators are FITTED PARAMETERS, so they must not see the data the
fleet is later measured on. A calibrator in particular directly moves a score
across its threshold, so fitting one on a test pair and then thresholding that
same pair with it is circular.

Sessions are matched to a split by the `gpt_generated/pair_<session>/` prefix of
their pair_id. Library pairs never appear in trajectories.jsonl and so are never
at risk; the exposure is limited to the generated sessions that landed in
val/test. --allow-all-sessions lifts the restriction and says so loudly.

Note the cost: filtering drops the fitting pool from 154 sessions to the ~99 in
train. If the calibrators come out unstable, fitting on train+val and protecting
only test recovers most of that while keeping the reported figures clean.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

HOME = Path.home()
TRAJ = REPO_ROOT / "generated_training_data/trajectories.jsonl"
TRAIN_DIR = REPO_ROOT / "generated_training_data"
CAL_OUT = DATA_ROOT / "experiments_out/fleet_calibration.json"
SPLITS = DATA_ROOT / "data" / "splits_v3_unified"
SESSION_RE = re.compile(r"gpt_generated/pair_([0-9_]+)/")

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
MODELS = ["CLIP Probe", "VLM Fine-tuned", "ResNet-50", "ViT-B/16", "DINOv2 Probe"]
FLEET = "v1 fleet: CLIP/DINOv2/ResNet/ViT + Qwen2-VL-7B, trained on splits_v3_unified"
EPS = 1e-6


def logit(p): return math.log(min(max(p, EPS), 1 - EPS) / (1 - min(max(p, EPS), 1 - EPS)))
def sigmoid(z): return 1.0 / (1.0 + math.exp(-z))


def sessions_in_split(splits_dir, name):
    """Generated-session ids appearing in one split, from their pair_ids."""
    p = splits_dir / f"{name}.jsonl"
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        m = SESSION_RE.search(json.loads(line)["pair_id"])
        if m:
            out.add(m.group(1))
    return out


def build_scorers():
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from evaluators import (CLIPProbeEvaluator, VLMEvaluator, ResNetEvaluator,
                            ViTEvaluator, DINOProbeEvaluator)
    print("loading deployed fleet (7B VLM — GPU)…", flush=True)
    ev = {
        "CLIP Probe": CLIPProbeEvaluator(),
        "DINOv2 Probe": DINOProbeEvaluator(),
        "ResNet-50": ResNetEvaluator(),
        "ViT-B/16": ViTEvaluator(),
        "VLM Fine-tuned": VLMEvaluator(),   # default model_path is now 7B
    }
    return ev


def collect_records(ev, allowed=None):
    """Score finals + inferred pre-edit positives -> per model (prob, label, opt).

    `allowed` is the set of session ids that may contribute. None means no
    filtering, which leaks evaluation sessions into the fit — see the module
    docstring.
    """
    trajs = [json.loads(l) for l in TRAJ.read_text().splitlines()]
    recs = {m: {"p": [], "y": [], "opt": []} for m in MODELS}
    cache = {}   # (nat, tac) -> {model: {opt: prob}}

    def score(nat, tac):
        key = (nat, tac)
        if key not in cache:
            cache[key] = {m: ev[m].score_pair(nat, tac) for m in MODELS}
        return cache[key]

    n_final = n_inf = 0
    n_skipped = 0
    for t in trajs:
        ts = t["trajectory_id"].replace("traj_", "")
        if allowed is not None and ts not in allowed:
            n_skipped += 1
            continue
        pdir = TRAIN_DIR / f"pair_{ts}"
        nat = str(pdir / "natural.jpg")
        steps = t["steps"]
        if not (pdir / "tactile.jpg").exists():
            continue
        # final
        sc = score(nat, str(pdir / "tactile.jpg"))
        for m in MODELS:
            for o in OPTS:
                recs[m]["p"].append(sc[m][o]); recs[m]["y"].append(t["human_labels"][o]); recs[m]["opt"].append(o)
        n_final += 1
        # inferred positives: pre-edit image for each targeted edit
        for si, s in enumerate(steps):
            o = s.get("target_option", "")
            if si == 0 or o not in OPTS:
                continue
            pre = pdir / f"tactile_iter{si-1}.jpg"
            if not pre.exists():
                continue
            sc = score(nat, str(pre))
            for m in MODELS:
                recs[m]["p"].append(sc[m][o]); recs[m]["y"].append(1); recs[m]["opt"].append(o)
            n_inf += 1
        print(f"scored {ts} ({n_final} finals, {n_inf} inferred)", flush=True)
    for m in MODELS:
        recs[m]["p"] = np.array(recs[m]["p"], float); recs[m]["y"] = np.array(recs[m]["y"], float)
    if n_skipped:
        print(f"\nskipped {n_skipped} session(s) not in the train split")
    return recs, n_final, n_inf


# ── calibration (per model: better of temperature / Platt on out-of-fold NLL) ──
def nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def fit_temp(p, y):
    z = np.array([logit(x) for x in p])
    r = minimize_scalar(lambda T: nll(1/(1+np.exp(-z/T)), y), bounds=(0.05, 20), method="bounded")
    return {"kind": "temperature", "T": float(r.x)}

def fit_platt(p, y):
    z = np.array([logit(x) for x in p])
    r = minimize(lambda ab: nll(1/(1+np.exp(-(ab[0]*z+ab[1]))), y), x0=[1.0, 0.0], method="Nelder-Mead")
    return {"kind": "platt", "A": float(r.x[0]), "B": float(r.x[1])}

def apply_cal(p, c):
    z = np.array([logit(x) for x in p])
    z = c["A"]*z + c["B"] if c["kind"] == "platt" else z / c["T"]
    return 1/(1+np.exp(-z))

def cv_nll(p, y, fit, k=5, seed=42):
    idx = np.random.default_rng(seed).permutation(len(p)); folds = np.array_split(idx, k)
    out = np.empty_like(p)
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(k) if g != f])
        out[te] = apply_cal(p[te], fit(p[tr], y[tr]))
    return nll(out, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--allow-all-sessions", action="store_true",
                    help="fit on every session, including val/test — LEAKS")
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir) if args.splits_dir else SPLITS
    if args.allow_all_sessions:
        allowed = None
        print("!! --allow-all-sessions: fitting on val/test sessions too.")
        print("!! Weights and calibrators produced here are LEAKED. Do not deploy.\n")
    else:
        allowed = sessions_in_split(splits_dir, "train")
        held = sessions_in_split(splits_dir, "val") | sessions_in_split(splits_dir, "test")
        overlap = allowed & held
        print(f"C2: fitting on {len(allowed)} train session(s); "
              f"{len(held)} val/test session(s) excluded  [{splits_dir}]")
        if overlap:
            raise SystemExit(f"session(s) in both train and val/test: {sorted(overlap)[:5]}")
        if not allowed:
            raise SystemExit(f"no generated sessions found in {splits_dir}/train.jsonl")

    ev = build_scorers()
    recs, n_final, n_inf = collect_records(ev, allowed)
    print(f"\ncalibration/weight set: {n_final} finals + {n_inf} inferred positives\n")

    # ENSEMBLE_WEIGHTS = per-option per-model accuracy on the live set.
    #
    # CAUTION: accuracy is degenerate under this class imbalance. On too_thick
    # (~14% positive) a judge that never flags scores ~86% and therefore earns
    # the HIGHEST weight — the combiner rewards silence on exactly the options
    # the fleet is worst at. AUC-based weights are computed alongside for
    # comparison; they are threshold-free and cannot be gamed by abstaining.
    # The accuracy weights remain the default only because they are what the
    # deployed combiner was built on. Compare before adopting either.
    def _auc(p, y):
        pos, neg = p[y > 0.5], p[y <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        return float((pos[:, None] > neg[None, :]).mean()
                     + 0.5 * (pos[:, None] == neg[None, :]).mean())

    weights = {o: {} for o in OPTS}
    weights_auc = {o: {} for o in OPTS}
    flag_rate = {o: {} for o in OPTS}
    for m in MODELS:
        for o in OPTS:
            mask = np.array(recs[m]["opt"]) == o
            p, y = recs[m]["p"][mask], recs[m]["y"][mask]
            acc = np.mean((p > 0.5) == (y > 0.5)) if len(p) else 0.5
            weights[o][m] = round(float(acc), 3)
            weights_auc[o][m] = round(_auc(p, y), 3) if len(p) else 0.5
            flag_rate[o][m] = round(float(np.mean(p > 0.5)), 3) if len(p) else 0.0

    calibrators = {}
    for m in MODELS:
        p, y = recs[m]["p"], recs[m]["y"]
        best = min(("temp", fit_temp), ("platt", fit_platt), key=lambda kv: cv_nll(p, y, kv[1]))
        calibrators[m] = best[1](p, y)

    CAL_OUT.write_text(json.dumps({
        "method": "per-model temperature/Platt by out-of-fold NLL",
        "fleet": FLEET,
        "splits_dir": str(splits_dir),
        "fit_sessions": "train-split only" if allowed is not None else "ALL (LEAKED)",
        "n_fit_sessions": len(allowed) if allowed is not None else None,
        "calibration_set": {"finals": n_final, "inferred_positives": n_inf},
        "ensemble_weights_accuracy": weights,
        "ensemble_weights_auc": weights_auc,
        "flag_rate_at_0.50": flag_rate,
        "calibrators": calibrators}, indent=2))

    print("ENSEMBLE_WEIGHTS = {")
    for o in OPTS:
        print(f'    "{o}": {{' + ", ".join(f'"{m}": {weights[o][m]}' for m in MODELS) + "},")
    print("}\n")
    print("\n# AUC-based weights (threshold-free alternative — NOT deployed)")
    print("ENSEMBLE_WEIGHTS_AUC = {")
    for o in OPTS:
        print(f'    "{o}": {{' + ", ".join(f'"{m}": {weights_auc[o][m]}' for m in MODELS) + "},")
    print("}\n")
    print("flag rate at t=0.50 (a near-zero rate with a high accuracy weight means")
    print("that judge is being rewarded for abstaining, not for detecting):")
    for o in OPTS:
        print(f"  {o:<18} " + "  ".join(f"{m.split()[0]}={flag_rate[o][m]:.2f}" for m in MODELS))
    print()
    print(f"calibrators saved -> {CAL_OUT}")
    for m, c in calibrators.items():
        print(f"  {m:<15} {c}")


if __name__ == "__main__":
    main()
