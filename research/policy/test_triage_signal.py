#!/usr/bin/env python3
"""Is there learnable signal for DEFECT TRIAGE in the current dataset?

Cheap CPU screen before committing to a VLM-based triage policy. Question:
given the panel's scores for a tactile image, can we predict which defect the
annotator chose to repair next — better than the trivial baselines?

  target_option is an ACTION label, not a diagnosis label (the five options are
  not mutually exclusive; see SESSION_NOTES "ROOT CAUSE"). So this predicts the
  annotator's repair policy, which is exactly Loop 3. It is NOT an evaluation of
  the panel's detection quality — do not report it as one.

BASELINES it must beat:
  * constant  — always guess the majority class (missing_texture, 40.6%)
  * panel     — argmax over the panel's own consensus/probability

Validation is GroupKFold on trajectory_id: 155 steps come from only 82 sessions,
and steps within a session share an image lineage, so a plain split leaks.

Two feature sources, reported separately:
  logged   — scores as they were ON SCREEN (trajectories.jsonl steps[].scores).
             Explains the annotator's historical behaviour.
  deployed — current-fleet re-score (trajectory_scores.json). This is what a
             deployed policy would actually see at inference, so it is the
             number that matters for whether this is buildable.

Usage:  python research/policy/test_triage_signal.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import importlib.util
import json
from collections import Counter

import numpy as np

try:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    raise SystemExit("needs scikit-learn: source ~/vlm_finetune/venv/bin/activate")

TRAJ = REPO_ROOT / "generated_training_data/trajectories.jsonl"
RESCORED = DATA_ROOT / "experiments_out/trajectory_scores.json"

MODELS = ["CLIP Probe", "VLM Fine-tuned", "ResNet-50", "ViT-B/16", "DINOv2 Probe"]


def _constants():
    path = REPO_ROOT / "evaluators" / "constants.py"
    spec = importlib.util.spec_from_file_location("_tc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OPTS = _constants().ALL_OPTIONS


def featurise(scores, history_opts):
    """25 raw probs + per-option mean and n_agree + history flags."""
    f = []
    for m in MODELS:
        per = scores.get(m, {})
        f += [float(per.get(o, 0.0)) for o in OPTS]
    for o in OPTS:
        vals = [scores[m][o] for m in scores if o in scores[m]]
        f.append(float(np.mean(vals)) if vals else 0.0)
        f.append(sum(1 for v in vals if v > 0.50))
    f.append(len(history_opts))
    f += [1.0 if o in history_opts else 0.0 for o in OPTS]
    return f


def panel_argmax(scores):
    """The panel's own pick: most-agreed defect, ties broken by mean probability."""
    best, best_key = None, None
    for o in OPTS:
        vals = [scores[m][o] for m in scores if o in scores[m]]
        if not vals:
            continue
        key = (sum(1 for v in vals if v > 0.50), float(np.mean(vals)))
        if best_key is None or key > best_key:
            best, best_key = o, key
    return best


def build(source):
    rescored = None
    if source == "deployed":
        if not RESCORED.exists():
            return None
        rescored = json.loads(RESCORED.read_text())["scores"]

    trajs = [json.loads(l) for l in TRAJ.read_text().splitlines() if l.strip()]
    X, y, groups, panel = [], [], [], []
    misses = 0
    for t in trajs:
        history = []
        steps = t["steps"]
        for i in range(1, len(steps)):
            prev, curr = steps[i - 1], steps[i]
            instr = (curr.get("instruction") or "").strip()
            if not instr or instr.startswith("[generated]"):
                continue
            tgt = (curr.get("target_option") or "").strip()
            sc = prev.get("scores") or {}
            if rescored is not None:
                img = prev.get("tac_image", "")
                if img in rescored:
                    sc = rescored[img]
                else:
                    misses += 1
            if not sc or not tgt or tgt == "multiple":
                if tgt:
                    history.append(tgt)
                continue
            X.append(featurise(sc, history))
            y.append(OPTS.index(tgt))
            groups.append(t["trajectory_id"])
            panel.append(panel_argmax(sc))
            history.append(tgt)
    if misses:
        print(f"  [{source}] {misses} step image(s) absent from the re-score, fell back to logged")
    return np.array(X), np.array(y), np.array(groups), panel


def run(source):
    built = build(source)
    if built is None:
        print(f"\n### {source}: trajectory_scores.json absent — skipped")
        return
    X, y, groups, panel = built
    n = len(y)
    print(f"\n{'=' * 74}\n### FEATURES: {source}   n={n}  groups={len(set(groups))}  dim={X.shape[1]}\n{'=' * 74}")

    counts = Counter(y)
    maj = counts.most_common(1)[0]
    print(f"  label distribution: { {OPTS[k]: v for k, v in counts.most_common()} }")
    print(f"  constant baseline ({OPTS[maj[0]]}) : {maj[1] / n:.1%}")
    pa = np.mean([OPTS[t] == p for t, p in zip(y, panel)])
    print(f"  panel argmax baseline              : {pa:.1%}")

    cv = GroupKFold(n_splits=5)
    for name, clf in [
        ("logistic regression", make_pipeline(StandardScaler(),
                                              LogisticRegression(max_iter=2000, C=0.5,
                                                                 class_weight="balanced"))),
        ("gradient boosting", GradientBoostingClassifier(random_state=42)),
    ]:
        pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)
        acc = float(np.mean(pred == y))
        print(f"  {name:<34}: {acc:.1%}   (grouped 5-fold CV)")
        if name == "gradient boosting":
            print()
            print(classification_report(y, pred, labels=list(range(len(OPTS))),
                                        target_names=OPTS, zero_division=0))
            print("  confusion (rows=true target, cols=predicted):")
            cm = confusion_matrix(y, pred, labels=list(range(len(OPTS))))
            print("    " + "".join(f"{o[:11]:>13}" for o in OPTS))
            for o, row in zip(OPTS, cm):
                print(f"  {o:<16}" + "".join(f"{v or '':>13}" for v in row))


def main():
    print("DEFECT-TRIAGE SIGNAL SCREEN — predicts the annotator's repair ACTION.")
    print("Not an evaluation of panel detection quality.")
    for source in ("logged", "deployed"):
        run(source)
    print("\nRead: if neither model clears the constant baseline by a clear margin,")
    print("the panel's scores alone do not carry the signal and the images are")
    print("required — i.e. a VLM-based triage policy, not a head on the scores.")


if __name__ == "__main__":
    main()
