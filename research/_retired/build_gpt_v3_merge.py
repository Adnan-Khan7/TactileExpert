"""Merge the July 2026 trajectory-collection pairs into data/splits_v2 (v3 merge).

Sources
-------
~/TactileExpert/generated_training_data/trajectories.jsonl (pairs saved on/after
2026-07-04; the 30 pre-July pairs are already in train.jsonl from the June merge).

What it does
------------
1. Holdout: every 5th pair by save order (indices 4, 9, 14, …) goes to a NEW
   file data/splits_v2/gpt_test.jsonl (split="gpt_test"). Training scripts only
   read train/val/test.jsonl, so this file is inert to them — it is the frozen
   GPT-domain evaluation set.
2. Remaining pairs -> appended to train.jsonl:
   - final image: one row per option, labels from human_labels (the
     checkbox-verified ground truth).
   - inferred step labels: for every edit step with a target_option, the
     PREVIOUS iteration's image gets a single positive row for that option
     (the user's action asserts the defect was present before the edit).
     Positives only — nothing is inferred about other options.
3. Copies all pair images (natural, tactile, tactile_iter*) into
   images/gpt_generated/pair_<ts>/.

Delta merge (safe to re-run): pairs whose timestamps already appear in
train.jsonl / gpt_test.jsonl are skipped; only new pairs are appended.
The every-5th holdout rule is indexed over the FULL sorted collection,
so earlier train/holdout assignments never change.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import json
import shutil
from pathlib import Path

ROOT = DATA_ROOT
TRAIN_SRC = REPO_ROOT / "generated_training_data"
SPLITS = ROOT / "data" / "splits_v2"
IMG_DST = ROOT / "images" / "gpt_generated"

OPTIONS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
CUTOFF = "20260704"

# ── Sealed split (proposal Ch5 §5.4) ──────────────────────────────────────────
# The in-domain holdout BOTH gates deployment AND grows with collection, so
# repeated gating progressively selects models against it and the final reported
# figure inherits that optimisation. A share of NEWLY collected holdout pairs is
# therefore sealed: never read by any gate, scored once at write-up.
#
# Sealing is prospective ONLY. The first 30 holdout pairs have already gated
# deployment decisions; sealing cannot be applied retroactively to them, and
# pretending otherwise would be worse than not sealing at all. SEAL_AFTER_ORDINAL
# is the count of holdout pairs that existed when sealing was introduced
# (2026-07-30) — do not lower it.
SEAL_AFTER_ORDINAL = 30
SEAL_EVERY = 3          # every 3rd NEW holdout pair -> sealed (~1/3, per notes)


def row(ts, obj, tac_rel, option, label, split, synthetic=False):
    nat = f"gpt_generated/pair_{ts}/natural.jpg"
    tac = f"gpt_generated/pair_{ts}/{tac_rel}"
    return {
        "pair_id": f"{nat}::{tac}",
        "natural_image": nat,
        "tactile_image": tac,
        "category": "gpt_generated",
        "object": obj,
        "option_id": option,
        "label": int(label),
        "split": split,
        # Carried through from app.py so downstream can train with or without
        # deliberately corrupted pairs, and so their contribution is auditable.
        "synthetic_corruption": bool(synthetic),
    }


def main():
    """Delta merge: pairs already present in train.jsonl or gpt_test.jsonl are
    skipped; new pairs keep the global every-5th holdout rule (index within
    the full sorted collection), so earlier assignments never change."""
    train_path = SPLITS / "train.jsonl"
    gpt_test_path = SPLITS / "gpt_test.jsonl"
    sealed_path = SPLITS / "gpt_sealed.jsonl"
    import re
    existing = train_path.read_text()
    if gpt_test_path.exists():
        existing += gpt_test_path.read_text()
    if sealed_path.exists():
        existing += sealed_path.read_text()   # else a sealed pair re-merges into the gate set
    merged_ts = set(re.findall(r"pair_(2026\d{4}_\d{6})", existing))

    trajs = [json.loads(l) for l in (TRAIN_SRC / "trajectories.jsonl").read_text().splitlines()]
    new = [t for t in trajs if t["trajectory_id"].replace("traj_", "") >= CUTOFF]
    new.sort(key=lambda t: t["trajectory_id"])
    delta = [t for t in new if t["trajectory_id"].replace("traj_", "") not in merged_ts]
    print(f"collection: {len(new)} pairs, already merged: {len(new) - len(delta)}, delta: {len(delta)}")
    if not delta:
        print("nothing to merge")
        return

    holdout_idx = set(range(4, len(new), 5))
    # Ordinal of each holdout index within the global holdout sequence. Stable as
    # the collection grows: indices are 4, 9, 14, ... so an existing pair's
    # ordinal never changes when new pairs are appended.
    hold_ordinal = {i: k for k, i in enumerate(sorted(holdout_idx))}
    train_rows, test_rows, sealed_rows = [], [], []
    n_inferred = 0
    n_synth_diverted = n_synth = 0

    for i, t in enumerate(new):
        ts = t["trajectory_id"].replace("traj_", "")
        if ts in merged_ts:
            continue
        obj = t["object"]
        src = TRAIN_SRC / f"pair_{ts}"
        dst = IMG_DST / f"pair_{ts}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.jpg"):
            shutil.copy2(f, dst / f.name)

        # Deliberately corrupted pairs NEVER enter the holdout. The holdout must
        # keep measuring NATURALLY occurring defects — a synthetic positive there
        # would mean reporting how well the models detect defects we manufactured
        # ourselves. Non-negotiable #3 of the controlled-corruption protocol.
        #
        # Diverting (rather than reindexing) preserves the property that earlier
        # train/holdout assignments never change: holdout_idx stays indexed over
        # the full sorted collection, and a synthetic pair landing on a holdout
        # index simply falls through to train. The holdout ends up slightly
        # smaller than 1-in-5, which is the correct trade.
        synthetic = bool(t.get("synthetic_corruption", False))
        n_synth += synthetic
        is_holdout = i in holdout_idx and not synthetic
        if synthetic and i in holdout_idx:
            n_synth_diverted += 1
            print(f"  synthetic pair {ts} landed on a holdout index — diverted to train")
        # Prospective sealing: every SEAL_EVERY-th holdout pair beyond the first
        # SEAL_AFTER_ORDINAL goes to a slice no gate ever reads.
        is_sealed = False
        if is_holdout:
            k = hold_ordinal[i]
            if k >= SEAL_AFTER_ORDINAL and (k - SEAL_AFTER_ORDINAL) % SEAL_EVERY == 0:
                is_sealed = True

        if is_sealed:
            split, bucket = "gpt_sealed", sealed_rows
        elif is_holdout:
            split, bucket = "gpt_test", test_rows
        else:
            split, bucket = "train", train_rows

        for opt in OPTIONS:
            bucket.append(row(ts, obj, "tactile.jpg", opt, t["human_labels"][opt],
                              split, synthetic))

        if not is_holdout and not is_sealed:
            for si, s in enumerate(t["steps"]):
                if si == 0 or s["target_option"] not in OPTIONS:
                    continue
                train_rows.append(row(ts, obj, f"tactile_iter{si-1}.jpg",
                                      s["target_option"], 1, "train", synthetic))
                n_inferred += 1

    with open(train_path, "a") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(gpt_test_path, "a") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")
    if sealed_rows:
        with open(sealed_path, "a") as f:
            for r in sealed_rows:
                f.write(json.dumps(r) + "\n")

    held = [new[i]["object"] for i in sorted(holdout_idx)
            if new[i]["trajectory_id"].replace("traj_", "") not in merged_ts]
    print(f"train rows appended: {len(train_rows)} "
          f"({len(train_rows) - n_inferred} final-label + {n_inferred} inferred-positive)")
    print(f"gpt_test rows: {len(test_rows)} ({len(held)} pairs: {held})")
    if n_synth:
        print(f"synthetic (deliberately corrupted) pairs merged: {n_synth}"
              f"  — {n_synth_diverted} diverted off a holdout index into train")
    assert not any(r["synthetic_corruption"] for r in test_rows), \
        "INVARIANT VIOLATED: a synthetic pair reached the holdout"
    if sealed_rows:
        n_sealed_pairs = len({r["pair_id"] for r in sealed_rows})
        print(f"SEALED this merge: {n_sealed_pairs} pair(s) -> {sealed_path.name} "
              f"(never read by any gate; score once at write-up)")
    assert not (set(r["pair_id"] for r in test_rows) & set(r["pair_id"] for r in sealed_rows)), \
        "INVARIANT VIOLATED: a pair is in both the gate holdout and the sealed slice"
    print("invariant OK: holdout contains no synthetic pairs; gate and sealed sets are disjoint")
    for split_name, rows in [("train-added", train_rows), ("gpt_test", test_rows)]:
        pos = {o: sum(1 for r in rows if r["option_id"] == o and r["label"]) for o in OPTIONS}
        print(f"  {split_name} positives: {pos}")


if __name__ == "__main__":
    main()
