"""Temperature-scale the evaluator fleet's probabilities on LIVE data (July 2026).

Why: the Chapter-5 RL reward is r_t = phi(q_t) - phi(q_{t+1}) with
phi(q) = mean per-option defect probability across the fleet. That reward is
only meaningful if q is calibrated — a raw sigmoid/softmax score of 0.30 must
actually mean a 30% chance the defect is present. The fleet was trained with
asymmetric / BCE losses and never calibrated, so this fits a per-model
temperature T (p_cal = sigmoid(logit(p)/T)) against collected human labels.

Design
------
* Calibration set = GPT-domain, the distribution the reward operates on:
    - FINALS: each pair's last-iteration per-model per-option score vs the
      human checkbox label (mostly negatives — real clean-final distribution).
    - INFERRED POSITIVES: for every edit step with a target_option, the
      pre-edit image's score for that option (label = 1) — same convention as
      build_gpt_v3_merge.py / derive_live_weights.py. Adds positive mass.
* One temperature per model (5 params). Per-option-per-model (25 params) is
  left as future work — too few positives per option today.
* Honest before/after via k-fold CV: T is fit on train folds and the
  calibrated probs are accumulated OUT-OF-FOLD, so the reported ECE/NLL/Brier
  reductions are not in-sample. Final deployed T is then fit on all data.

Output: per-model temperatures + before/after report, saved to
vlm_finetune/experiments_out/fleet_calibration.json. NOT wired into the live
app — trajectories keep storing raw scores; apply this transform offline when
preparing RL data (recompute rewards) or at the next deploy boundary.

Usage:  python code/experiments/derive_calibration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

HOME = Path.home()
TRAJ = REPO_ROOT / "generated_training_data/trajectories.jsonl"
OUT = DATA_ROOT / "experiments_out/fleet_calibration.json"

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
MODELS = ["CLIP Probe", "VLM Fine-tuned", "ResNet-50", "ViT-B/16", "DINOv2 Probe"]
# Calibrators are valid ONLY for the fleet that produced the raw scores —
# re-run after every deploy. Current deployed fleet (see models/*/RECIPE.txt):
FLEET = "CLIP/DINOv2/ViT v4; ResNet/VLM v3 (as of 2026-07-20)"
EPS = 1e-6
N_BINS = 10
K_FOLDS = 5
SEED = 42


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def apply_T(p, T):
    return sigmoid(logit(p) / T)


def nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def ece(p, y, n_bins=N_BINS):
    """Reliability-diagram ECE: |mean(pred) - positive-rate| per bin, weighted."""
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        total += (mask.sum() / n) * abs(conf - acc)
    return float(total)


def fit_temperature(p, y):
    """T > 0 minimising binary NLL of sigmoid(logit(p)/T). One parameter."""
    res = minimize_scalar(lambda T: nll(apply_T(p, T), y),
                          bounds=(0.05, 20.0), method="bounded")
    return {"kind": "temperature", "T": float(res.x)}


def apply_platt(p, A, B):
    return sigmoid(A * logit(p) + B)


def fit_platt(p, y):
    """Platt scaling: sigmoid(A*logit(p)+B). Two parameters — scale AND bias,
    so it can shift toward a base rate != 0.5, which temperature cannot."""
    z = logit(p)
    res = minimize(lambda ab: nll(sigmoid(ab[0] * z + ab[1]), y),
                   x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    A, B = float(res.x[0]), float(res.x[1])
    return {"kind": "platt", "A": A, "B": B}


def apply_cal(p, cal):
    if cal["kind"] == "temperature":
        return apply_T(p, cal["T"])
    return apply_platt(p, cal["A"], cal["B"])


def build_records():
    trajs = [json.loads(l) for l in TRAJ.read_text().splitlines()]
    # per model: (prob, label) lists; keep per-option tag for diagnostics
    recs = {m: {"p": [], "y": [], "opt": []} for m in MODELS}
    n_final = n_inf = 0
    for t in trajs:
        steps = t.get("steps", [])
        if not steps or "scores" not in steps[-1]:
            continue
        hl = t["human_labels"]
        final = steps[-1]["scores"]
        for m in MODELS:
            if m not in final:
                continue
            for o in OPTS:
                if o in final[m]:
                    recs[m]["p"].append(final[m][o])
                    recs[m]["y"].append(hl[o])
                    recs[m]["opt"].append(o)
        n_final += 1
        for si, s in enumerate(steps):
            o = s.get("target_option", "")
            if si == 0 or o not in OPTS:
                continue
            prev = steps[si - 1].get("scores", {})
            for m in MODELS:
                if m in prev and o in prev[m]:
                    recs[m]["p"].append(prev[m][o])
                    recs[m]["y"].append(1)
                    recs[m]["opt"].append(o)
            n_inf += 1
    for m in MODELS:
        recs[m]["p"] = np.array(recs[m]["p"], dtype=float)
        recs[m]["y"] = np.array(recs[m]["y"], dtype=float)
    return recs, n_final, n_inf


def cv_calibrated(p, y, fit_fn, rng):
    """Out-of-fold calibrated probabilities via K-fold fitting of fit_fn."""
    idx = rng.permutation(len(p))
    folds = np.array_split(idx, K_FOLDS)
    out = np.empty_like(p)
    for f in range(K_FOLDS):
        test_idx = folds[f]
        train_idx = np.concatenate([folds[g] for g in range(K_FOLDS) if g != f])
        cal = fit_fn(p[train_idx], y[train_idx])
        out[test_idx] = apply_cal(p[test_idx], cal)
    return out


def main():
    recs, n_final, n_inf = build_records()
    rng = np.random.default_rng(SEED)
    print(f"calibration set: {n_final} finals + {n_inf} inferred-positive records\n")

    print(f"{'model':<15}{'n':>4}{'pos%':>6}  {'method':<6}"
          f"{'NLL':>15}{'ECE':>15}{'Brier':>15}")
    print("-" * 76)
    calibrators = {}
    for m in MODELS:
        p, y = recs[m]["p"], recs[m]["y"]
        if len(p) == 0:
            continue
        posr = 100 * y.mean()
        raw = (nll(p, y), ece(p, y), brier(p, y))
        # honest out-of-fold comparison of both methods
        oof = {}
        for name, fn in (("temp", fit_temperature), ("platt", fit_platt)):
            pc = cv_calibrated(p, y, fn, np.random.default_rng(SEED))
            oof[name] = (nll(pc, y), ece(pc, y), brier(pc, y))
        best = min(oof, key=lambda k: oof[k][0])          # select on out-of-fold NLL
        calibrators[m] = (fit_temperature if best == "temp" else fit_platt)(p, y)
        print(f"{m:<15}{len(p):>4}{posr:>5.0f}%  {'raw':<6}"
              f"{raw[0]:>15.3f}{raw[1]:>15.3f}{raw[2]:>15.3f}")
        for name in ("temp", "platt"):
            tag = name + ("*" if name == best else "")
            print(f"{'':<15}{'':>4}{'':>6}  {tag:<6}"
                  f"{oof[name][0]:>15.3f}{oof[name][1]:>15.3f}{oof[name][2]:>15.3f}")

    # per-option diagnostic (which options are most miscalibrated, pooled fleet)
    print("\nper-option ECE (pooled across models, raw):")
    pooled = defaultdict(lambda: {"p": [], "y": []})
    for m in MODELS:
        for pr, yy, oo in zip(recs[m]["p"], recs[m]["y"], recs[m]["opt"]):
            pooled[oo]["p"].append(pr); pooled[oo]["y"].append(yy)
    for o in OPTS:
        p = np.array(pooled[o]["p"]); y = np.array(pooled[o]["y"])
        print(f"  {o:<16} n={len(p):>3}  pos={int(y.sum()):>3}  ECE={ece(p,y):.3f}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "method": "per-model probability calibration; per model the better of "
                  "temperature (sigmoid(logit(p)/T)) and Platt "
                  "(sigmoid(A*logit(p)+B)), selected on out-of-fold NLL",
        "fleet": FLEET,
        "calibration_set": {"finals": n_final, "inferred_positives": n_inf,
                            "positive_rate": round(float(recs[MODELS[0]]["y"].mean()), 3),
                            "domain": "gpt-image-1 collected trajectories"},
        "calibrators": calibrators,
    }, indent=2))
    print(f"\nsaved calibrators -> {OUT}")
    for m, c in calibrators.items():
        print(f"  {m:<15} {c}")


if __name__ == "__main__":
    main()
