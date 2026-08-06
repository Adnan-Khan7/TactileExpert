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


def row(ts, obj, tac_rel, option, label, split):
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
    }


def main():
    """Delta merge: pairs already present in train.jsonl or gpt_test.jsonl are
    skipped; new pairs keep the global every-5th holdout rule (index within
    the full sorted collection), so earlier assignments never change."""
    train_path = SPLITS / "train.jsonl"
    gpt_test_path = SPLITS / "gpt_test.jsonl"
    import re
    existing = train_path.read_text()
    if gpt_test_path.exists():
        existing += gpt_test_path.read_text()
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
    train_rows, test_rows = [], []
    n_inferred = 0

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

        is_holdout = i in holdout_idx
        split = "gpt_test" if is_holdout else "train"
        bucket = test_rows if is_holdout else train_rows

        for opt in OPTIONS:
            bucket.append(row(ts, obj, "tactile.jpg", opt, t["human_labels"][opt], split))

        if not is_holdout:
            for si, s in enumerate(t["steps"]):
                if si == 0 or s["target_option"] not in OPTIONS:
                    continue
                train_rows.append(row(ts, obj, f"tactile_iter{si-1}.jpg",
                                      s["target_option"], 1, "train"))
                n_inferred += 1

    with open(train_path, "a") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(gpt_test_path, "a") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")

    held = [new[i]["object"] for i in sorted(holdout_idx)
            if new[i]["trajectory_id"].replace("traj_", "") not in merged_ts]
    print(f"train rows appended: {len(train_rows)} "
          f"({len(train_rows) - n_inferred} final-label + {n_inferred} inferred-positive)")
    print(f"gpt_test rows: {len(test_rows)} ({len(held)} pairs: {held})")
    for split_name, rows in [("train-added", train_rows), ("gpt_test", test_rows)]:
        pos = {o: sum(1 for r in rows if r["option_id"] == o and r["label"]) for o in OPTIONS}
        print(f"  {split_name} positives: {pos}")


if __name__ == "__main__":
    main()
