"""Extract edit-policy training data from TactileExpert trajectories (Ch5 Loop 3).

Produces, from generated_training_data/trajectories.jsonl:

  BC dataset  (bc.jsonl)  — one record per human edit step:
      state  = (natural image, PRE-edit tactile image, per-model raw +
                calibrated defect scores of the pre-edit image, target_option,
                object, edit-history-so-far)
      action = the human's edit instruction (the one that produced the next
               iteration)
      outcome = raw + CALIBRATED reward (defect-mass reduction), `helped` flag
      Behavior cloning trains on records with helped=True (calibrated reward
      > 0); the full set is emitted so filtering stays a training-time choice.

  DPO dataset (dpo.jsonl) — one record per rewritten prefill:
      chosen   = the human's instruction
      rejected = the prefill suggestion shown at the time
      A pair is emitted only when the two DIFFER after normalization
      (whitespace collapse, case-fold, object-name substitution): text equal
      after normalization means the human effectively ACCEPTED the suggestion
      (e.g. copied the unsubstituted card text), which is acceptance signal,
      not preference signal.

Rewards use the fleet calibrators from experiments_out/fleet_calibration.json
(Platt/temperature per model). Trajectories store RAW scores, so calibration
is applied here at extraction time; recompute after every fleet deploy.

Usage:  python code/policy/extract_policy_data.py
Output: experiments_out/policy_data/{bc.jsonl,dpo.jsonl,stats.json}
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import json
import math
import re
from pathlib import Path

HOME = Path.home()
TRAJ = REPO_ROOT / "generated_training_data/trajectories.jsonl"
CAL = DATA_ROOT / "experiments_out/fleet_calibration.json"
OUT_DIR = DATA_ROOT / "experiments_out/policy_data"
RESCORED = DATA_ROOT / "experiments_out/trajectory_scores.json"

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
EPS = 1e-6


def load_calibrators():
    cal = json.loads(CAL.read_text())
    print(f"calibrators for fleet: {cal.get('fleet', '?')}")
    return cal["calibrators"]


def load_rescored():
    """Per-image scores from the deployed fleet, if they have been produced.

    Without this file the only scores available are the ones each fleet logged
    at collection time, which span several fleet generations — so applying the
    current fleet's calibrators to them mixes distributions and corrupts the
    reward sign. research/experiments/rescore_trajectories.py produces the
    single-fleet replacement; prefer it whenever present.
    """
    if not RESCORED.exists():
        print("WARNING: no trajectory_scores.json — falling back to the scores "
              "logged at collection time. Rewards will be FLEET-MIXED and the "
              "`helped` counts are not trustworthy. Run "
              "research/experiments/rescore_trajectories.sh first.")
        return None
    d = json.loads(RESCORED.read_text())
    print(f"re-scored images from: {d.get('fleet', '?')}  ({d.get('n_images')} images)")
    return d["scores"]


def calibrate(p, c):
    z = math.log(max(p, EPS) / max(1 - p, EPS))
    if c["kind"] == "platt":
        z = c["A"] * z + c["B"]
    else:  # temperature
        z = z / c["T"]
    return 1.0 / (1.0 + math.exp(-z))


def defect_mass(scores, calibrators=None):
    """Mean defect probability across all models and options present.
    Raw when calibrators is None, calibrated otherwise."""
    vals = []
    for m, per_opt in scores.items():
        c = calibrators.get(m) if calibrators else None
        for o in OPTS:
            if o in per_opt:
                p = per_opt[o]
                vals.append(calibrate(p, c) if c else p)
    return sum(vals) / len(vals) if vals else None


def norm_text(s, obj=""):
    s = s.strip()
    if obj:
        s = s.replace("the object", f"the {obj}")
    s = re.sub(r"\s+", " ", s).lower().rstrip(".")
    return s


def main():
    calibrators = load_calibrators()
    rescored = load_rescored()
    trajs = [json.loads(l) for l in TRAJ.read_text().splitlines()]

    def scores_of(step):
        """Deployed-fleet scores for this step's image, else the logged ones."""
        if rescored is not None:
            img = step.get("tac_image", "")
            if img in rescored:
                return rescored[img]
        return step.get("scores", {})

    bc, dpo = [], []
    n_steps = n_prefill = n_accept_logged = n_accept_norm = 0
    n_rescored_hits = n_logged_fallback = 0

    for t in trajs:
        obj = (t.get("object") or "").strip()
        era = "ensemble" if "Ensemble (weighted)" in t.get("evaluator_flags", {}) else "pre-ensemble"
        steps = t["steps"]
        history = []
        for si in range(1, len(steps)):
            prev, curr = steps[si - 1], steps[si]
            instr = (curr.get("instruction") or "").strip()
            if not instr or instr.startswith("[generated]"):
                continue
            n_steps += 1

            s_prev, s_curr = scores_of(prev), scores_of(curr)
            if rescored is not None:
                for st in (prev, curr):
                    if st.get("tac_image", "") in rescored:
                        n_rescored_hits += 1
                    else:
                        n_logged_fallback += 1

            raw_prev = defect_mass(s_prev)
            raw_curr = defect_mass(s_curr)
            cal_prev = defect_mass(s_prev, calibrators)
            cal_curr = defect_mass(s_curr, calibrators)
            r_raw = round(raw_prev - raw_curr, 4) if None not in (raw_prev, raw_curr) else None
            r_cal = round(cal_prev - cal_curr, 4) if None not in (cal_prev, cal_curr) else None

            state = {
                "trajectory_id": t["trajectory_id"],
                "object": obj,
                "era": era,
                "nat_image": curr.get("nat_image", ""),
                "tac_image": prev.get("tac_image", ""),   # the image being edited
                "scores_raw": s_prev,
                "target_option": curr.get("target_option", ""),
                "history": list(history),
            }
            bc.append({**state,
                       "instruction": instr,
                       "reward_raw": r_raw,
                       "reward_calibrated": r_cal,
                       "helped": bool(r_cal is not None and r_cal > 0)})

            prefill = (curr.get("prefill_instruction") or "").strip()
            if prefill:
                n_prefill += 1
                if curr.get("accepted_prefill"):
                    n_accept_logged += 1
                if norm_text(instr, obj) == norm_text(prefill, obj):
                    n_accept_norm += 1     # effective acceptance — no preference signal
                else:
                    dpo.append({**state,
                                "chosen": instr,
                                "rejected": prefill,
                                "prefill_trusted": curr.get("prefill_trusted", ""),
                                "reward_calibrated": r_cal})
            history.append(instr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("bc.jsonl", bc), ("dpo.jsonl", dpo)):
        with open(OUT_DIR / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    helped = sum(r["helped"] for r in bc)
    stats = {
        "trajectories": len(trajs),
        "edit_steps": n_steps,
        "bc_records": len(bc),
        "bc_helped (calibrated reward > 0)": helped,
        "dpo_pairs": len(dpo),
        "prefills_shown": n_prefill,
        "accept_logged_flag": n_accept_logged,
        "accept_after_normalization": n_accept_norm,
        "effective_acceptance_rate": round(n_accept_norm / n_prefill, 3) if n_prefill else None,
        "score_source": ("deployed-fleet re-score" if rescored is not None
                         else "trajectory-logged (FLEET-MIXED — rewards unreliable)"),
        "step_scores_from_rescore": n_rescored_hits,
        "step_scores_from_log": n_logged_fallback,
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nsaved -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
