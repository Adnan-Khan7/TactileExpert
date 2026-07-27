"""Re-derive ENSEMBLE_WEIGHTS + Platt calibrators for the NEWLY DEPLOYED fleet.

Why a fresh GPU pass: the live weights and calibrators were fit on scores the
OLD fleet logged into trajectories.jsonl. After deploying v5 vision + 7B VLM,
those are stale — so this re-scores the collected data with the *deployed*
evaluators and re-derives from scratch.

Scores (with the deployed fleet in ~/TactileExpert/models/):
  * trajectory FINALS  (label = human checkbox)  + INFERRED pre-edit positives
    (edit-step target_option, label=1) — the live-disagreement set used for
    per-option ENSEMBLE_WEIGHTS (per-model accuracy) and for calibration.

Outputs (paste weights into constants.py; calibration JSON is used offline):
  * ENSEMBLE_WEIGHTS dict (printed)
  * experiments_out/fleet_calibration.json  (per-model Platt/temperature)
Run as a GPU job (7B). Then paste weights, and re-run extract_policy_data.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

HOME = Path.home()
TRAJ = REPO_ROOT / "generated_training_data/trajectories.jsonl"
TRAIN_DIR = REPO_ROOT / "generated_training_data"
CAL_OUT = DATA_ROOT / "experiments_out/fleet_calibration.json"

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
MODELS = ["CLIP Probe", "VLM Fine-tuned", "ResNet-50", "ViT-B/16", "DINOv2 Probe"]
FLEET = "CLIP/DINOv2/ResNet/ViT v5; VLM 7B (deployed 2026-07-27)"
EPS = 1e-6


def logit(p): return math.log(min(max(p, EPS), 1 - EPS) / (1 - min(max(p, EPS), 1 - EPS)))
def sigmoid(z): return 1.0 / (1.0 + math.exp(-z))


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


def collect_records(ev):
    """Score finals + inferred pre-edit positives -> per model (prob, label, opt)."""
    trajs = [json.loads(l) for l in TRAJ.read_text().splitlines()]
    recs = {m: {"p": [], "y": [], "opt": []} for m in MODELS}
    cache = {}   # (nat, tac) -> {model: {opt: prob}}

    def score(nat, tac):
        key = (nat, tac)
        if key not in cache:
            cache[key] = {m: ev[m].score_pair(nat, tac) for m in MODELS}
        return cache[key]

    n_final = n_inf = 0
    for t in trajs:
        ts = t["trajectory_id"].replace("traj_", "")
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
    ev = build_scorers()
    recs, n_final, n_inf = collect_records(ev)
    print(f"\ncalibration/weight set: {n_final} finals + {n_inf} inferred positives\n")

    # ENSEMBLE_WEIGHTS = per-option per-model accuracy on the live set
    weights = {o: {} for o in OPTS}
    for m in MODELS:
        for o in OPTS:
            mask = np.array(recs[m]["opt"]) == o
            p, y = recs[m]["p"][mask], recs[m]["y"][mask]
            acc = np.mean((p > 0.5) == (y > 0.5)) if len(p) else 0.5
            weights[o][m] = round(float(acc), 3)

    calibrators = {}
    for m in MODELS:
        p, y = recs[m]["p"], recs[m]["y"]
        best = min(("temp", fit_temp), ("platt", fit_platt), key=lambda kv: cv_nll(p, y, kv[1]))
        calibrators[m] = best[1](p, y)

    CAL_OUT.write_text(json.dumps({
        "method": "per-model temperature/Platt by out-of-fold NLL",
        "fleet": FLEET,
        "calibration_set": {"finals": n_final, "inferred_positives": n_inf},
        "calibrators": calibrators}, indent=2))

    print("ENSEMBLE_WEIGHTS = {")
    for o in OPTS:
        print(f'    "{o}": {{' + ", ".join(f'"{m}": {weights[o][m]}' for m in MODELS) + "},")
    print("}\n")
    print(f"calibrators saved -> {CAL_OUT}")
    for m, c in calibrators.items():
        print(f"  {m:<15} {c}")


if __name__ == "__main__":
    main()
