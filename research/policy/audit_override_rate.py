#!/usr/bin/env python3
"""Does the panel's diagnosis predict which defect the annotator chose to fix?

This is the measurement that decides whether the DEFECT-TRIAGE model is worth
building (see SESSION_NOTES_JULY29.md, "LOOP 3 REFRAMED"). No GPU, no API calls.

SOURCE: scores LOGGED AT COLLECTION TIME (trajectories.jsonl steps[].scores) —
what was on the annotator's screen.

  !! Do NOT use bc.jsonl's `scores_raw`. extract_policy_data.py substitutes
  trajectory_scores.json (the later deployed-fleet re-score) when it exists, so
  bc.jsonl holds scores the annotator never saw.

ERA FACTS (user, 2026-07-29):
  * pre-ensemble steps were seeded by a SINGLE model (CLIP, occasionally DINOv2),
    though all five models were scored and logged even then.
  * the ensemble live during collection contained Qwen2-VL-2B; the 7B deployed
    2026-07-27 after collection paused, so NO session ever saw the 7B.

So today's ENSEMBLE_WEIGHTS (7B-derived) cannot reconstruct the ordering the
annotator saw. Everything reported here is therefore WEIGHT-FREE: per-model flags
counted at prob > 0.50, which is exact in every era.

  !! `prefill_trusted` is USELESS as an override marker: the key is absent from
  all 55 pre-ensemble steps, and in the 102 steps that have it the value is
  'Ensemble (weighted)' 102/102. An earlier version of this script counted the
  absent keys as "picked an unflagged option" and reported a spurious 35.5%.

Usage:  python research/policy/audit_override_rate.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT  # noqa: E402

import importlib.util
import json
from collections import Counter, defaultdict

TRAJ = REPO_ROOT / "generated_training_data/trajectories.jsonl"


def _constants():
    """Load constants.py by path — evaluators/__init__.py imports torch and this
    script must run on a login node with bare python."""
    path = REPO_ROOT / "evaluators" / "constants.py"
    spec = importlib.util.spec_from_file_location("_tactile_constants", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OPTS = _constants().ALL_OPTIONS


def load_steps():
    """(era, n_agree per option, annotator's target) for every human edit step."""
    trajs = [json.loads(l) for l in TRAJ.read_text().splitlines() if l.strip()]
    rows, skipped = [], Counter()
    for t in trajs:
        era = ("ensemble" if "Ensemble (weighted)" in t.get("evaluator_flags", {})
               else "pre-ensemble")
        steps = t["steps"]
        for i in range(1, len(steps)):
            prev, curr = steps[i - 1], steps[i]
            instr = (curr.get("instruction") or "").strip()
            if not instr or instr.startswith("[generated]"):
                continue
            tgt = (curr.get("target_option") or "").strip()
            sc = prev.get("scores") or {}
            if not sc:
                skipped["no_logged_scores"] += 1
                continue
            if not tgt:
                skipped["target_missing"] += 1
                continue
            if tgt == "multiple":
                skipped["target_multiple"] += 1
                continue
            n_agree = {o: sum(1 for per in sc.values()
                              if o in per and per[o] > 0.50) for o in OPTS}
            rows.append((era, n_agree, tgt, len(sc)))
    return rows, skipped


def main():
    rows, skipped = load_steps()
    print(f"usable human edit steps: {len(rows)}   skipped: {dict(skipped)}")
    print(f"models scored per step : {dict(Counter(r[3] for r in rows))}")
    print()

    print("=" * 78)
    print("DID THE ANNOTATOR TARGET WHAT THE MODELS AGREED ON?  (weight-free, exact)")
    print("=" * 78)
    print(f"{'era':14}{'n':>5}{'>=3/5 steps':>13}{'overridden':>12}{'':>8}"
          f"{'>=4/5 steps':>13}{'overridden':>12}")
    for era in ("ensemble", "pre-ensemble", "ALL"):
        R = [r for r in rows if era == "ALL" or r[0] == era]
        s3 = [r for r in R if any(v >= 3 for v in r[1].values())]
        o3 = [r for r in s3 if r[1].get(r[2], 0) < 3]
        s4 = [r for r in R if any(v >= 4 for v in r[1].values())]
        o4 = [r for r in s4 if r[1].get(r[2], 0) < 4]
        print(f"{era:14}{len(R):>5}{len(s3):>13}{len(o3):>7}"
              f"{o3 and len(o3)/len(s3) or 0:>8.1%}"
              f"{len(s4):>13}{len(o4):>7}{o4 and len(o4)/len(s4) or 0:>8.1%}")
    z = [r for r in rows if r[1].get(r[2], 0) == 0]
    print(f"\ntarget flagged by ZERO models: {len(z)} ({len(z)/len(rows):.1%})"
          f"   [notes report 4.5% — cross-check]")
    print()

    print("=" * 78)
    print("CONFUSION: panel's strongest-consensus defect -> annotator's target")
    print("=" * 78)
    M, tot = defaultdict(Counter), Counter()
    for era, na, tgt, _ in rows:
        strong = [o for o in OPTS if na[o] >= 3]
        if not strong:
            continue
        M[max(strong, key=lambda o: na[o])][tgt] += 1
        tot[max(strong, key=lambda o: na[o])] += 1
    w = 17
    print(" " * w + "".join(f"{o[:11]:>13}" for o in OPTS) + f"{'total':>8}{'followed':>10}")
    agree = 0
    for top in OPTS:
        if not tot[top]:
            continue
        agree += M[top][top]
        print(f"{top:<{w}}" + "".join(f"{M[top][o] or '':>13}" for o in OPTS)
              + f"{tot[top]:>8}{M[top][top]/tot[top]:>10.0%}")
    n_cons = sum(tot.values())
    print()

    print("=" * 78)
    print("BASELINES — what any triage model must beat")
    print("=" * 78)
    counts = Counter(r[2] for r in rows)
    maj, maj_n = counts.most_common(1)[0]
    print(f"  panel's strongest-consensus defect == target : {agree}/{n_cons} = {agree/n_cons:.1%}")
    print(f"  constant guess '{maj}'{'':<6} : {maj_n}/{len(rows)} = {maj_n/len(rows):.1%}")
    print(f"  annotator target distribution: {dict(counts.most_common())}")
    print()
    if agree / n_cons <= maj_n / len(rows):
        print("  >>> The panel does NOT beat a constant guess. Same shape as the")
        print("      always-clean baseline result for detection, now for triage.")


if __name__ == "__main__":
    main()
