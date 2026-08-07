#!/usr/bin/env python3
"""Ingest collected sessions into the unified splits.

Reads  $TACTILE_COLLECT_DIR/annotations.jsonl   (app.py on_step_save output)
Writes data/splits_v3_unified/{train,val,test}.jsonl

DRY RUN BY DEFAULT. Nothing is written without --apply.

WHY THIS SCRIPT EXISTS
----------------------
Nothing connected the collection interface to the splits. `build_unified_splits.py`
re-assigns rows that already exist in data/splits_v2/ — a directory that no longer
exists on disk — and the only scripts that ever read annotations.jsonl into a split
are in research/_retired/. So every session the app collected after dataset v1 was
frozen had no path into training data at all.

THE LABEL TRAP THIS CLOSES
--------------------------
app.py used to write one fully-labelled `tactile.jpg` per session, while the
intermediate `tactile_iterN.jpg` images carried a SINGLE inferred positive — the
defect the annotator chose to target next. That is an ACTION label, not a
diagnosis, and 37.5% of them were corrected on review. Two consequences are
baked into the older code and must not be carried forward:

  * `build_unified_splits.is_final()` tests for the filename `tactile.jpg` and
    uses it as a proxy for "trustworthy enough to evaluate on". app.py no longer
    writes `tactile.jpg` at all — it was a byte-identical duplicate of the last
    iteration carrying its own second label row. So is_final() is False for
    EVERY row this script produces, and it is NOT a usable trust signal here.

  * Every row in annotations.jsonl now carries all five options, set by a human
    on the iteration in front of them. These are diagnoses, not actions. They are
    stamped label_verified=True with verification_kind="collected_step", so
    coverage stays answerable from the dataset alone.

Read trust off `verification_kind`, never off the filename.

SPLIT PLACEMENT
---------------
Everything lands in TRAIN unless --val/--test are given. Two reasons: iterations
of one session share a photograph, so a session split across train and test leaks;
and the deployed fleet's numbers are quoted against the v1 test set, which must
not move underneath them. When --val/--test are used, sessions are assigned whole
and deterministically under --seed, and a session with any pair already sitting in
another split is refused rather than moved.

UPSERT, NOT APPEND
------------------
Rows are keyed on (pair_id, option_id), matching the app's own _upsert. Re-saving
an iteration in the app corrects its row; re-ingesting corrects the split row.
Additions and corrections are counted separately, because they have different
consequences: a correction changes labels only, an addition changes the record set
and therefore invalidates every cached feature file.

Usage:
  python research/annotator/ingest_collected.py                    # dry run
  python research/annotator/ingest_collected.py --apply
  python research/annotator/ingest_collected.py --val 0.15 --test 0.15 --apply
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT, require_data_root  # noqa: E402

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime

ACTIVE = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
SESSION_RE = re.compile(r"(pair_\d{8}_\d{6})")


def session_of(rel_path: str):
    """pair_YYYYMMDD_HHMMSS from a staged relative path, else None."""
    m = SESSION_RE.search(rel_path)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-dir", default=None,
                    help="collection root (default $TACTILE_COLLECT_DIR, else "
                         "REPO_ROOT/generated_training_data_v2)")
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--category", default="gpt_generated_v2",
                    help="image subdirectory and row category (default gpt_generated_v2, "
                         "kept distinct from the frozen v1 gpt_generated)")
    ap.add_argument("--val", type=float, default=0.0, help="fraction of NEW sessions to val")
    ap.add_argument("--test", type=float, default=0.0, help="fraction of NEW sessions to test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args()

    require_data_root()
    collect = Path(args.collect_dir) if args.collect_dir else Path(
        os.environ.get("TACTILE_COLLECT_DIR", REPO_ROOT / "generated_training_data_v2"))
    splits = Path(args.splits_dir) if args.splits_dir else DATA_ROOT / "data" / "splits_v3_unified"
    images = DATA_ROOT / "images"

    ann_p = collect / "annotations.jsonl"
    if not ann_p.exists():
        raise SystemExit(f"no annotations.jsonl under {collect}")
    ann = [json.loads(l) for l in ann_p.read_text().splitlines() if l.strip()]
    print(f"collection: {collect}")
    print(f"  {len(ann)} labelled iteration(s)\n")
    if not ann:
        raise SystemExit("annotations.jsonl is empty — collect a session first")

    # ── validate before touching anything ────────────────────────────────────
    missing, incomplete = [], []
    for r in ann:
        for k in ("natural_image", "tactile_image"):
            if not (collect / r[k]).exists():
                missing.append(r[k])
        absent = [o for o in ACTIVE if o not in r]
        if absent:
            incomplete.append((r.get("pair_id"), absent))
    if missing:
        raise SystemExit(f"{len(missing)} image(s) referenced but absent under {collect}, "
                         f"e.g. {missing[0]}")
    if incomplete:
        # A row missing options is the OLD single-inferred-positive shape. Refuse
        # it rather than filling the gaps with zeros, which would assert negatives
        # no human ever stated.
        print(f"REFUSING {len(incomplete)} row(s) that do not carry all five options:")
        for pid, absent in incomplete[:5]:
            print(f"  {pid}  missing {', '.join(absent)}")
        raise SystemExit("these are action labels, not diagnoses — re-label them in the app")

    # ── session -> split ──────────────────────────────────────────────────────
    sessions = sorted({session_of(r["tactile_image"]) or r["pair_id"] for r in ann})
    order = list(sessions)
    random.Random(args.seed).shuffle(order)
    n_val = int(round(args.val * len(sessions)))
    n_test = int(round(args.test * len(sessions)))
    dest_of = {s: "train" for s in sessions}
    for s in order[:n_val]:
        dest_of[s] = "val"
    for s in order[n_val:n_val + n_test]:
        dest_of[s] = "test"

    # ── load splits, index existing rows ─────────────────────────────────────
    existing = {}
    where = {}          # pair_id -> split it already lives in
    for split in ("train", "val", "test"):
        p = splits / f"{split}.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
        existing[split] = rows
        for r in rows:
            where[r["pair_id"]] = split

    # Leak guard: a session already placed cannot be re-placed elsewhere.
    placed = defaultdict(set)
    for pid, split in where.items():
        s = session_of(pid)
        if s:
            placed[s].add(split)
    conflicts = []
    for s, dest in dest_of.items():
        prior = placed.get(s, set())
        if prior and dest not in prior:
            conflicts.append((s, sorted(prior), dest))
    if conflicts:
        print("REFUSING — these sessions already have pairs in another split:")
        for s, prior, dest in conflicts:
            print(f"  {s}: already in {'/'.join(prior)}, would go to {dest}")
        raise SystemExit("assigning them again would split a session across the "
                         "train/eval boundary and leak a shared photograph")

    # ── build rows ───────────────────────────────────────────────────────────
    added = Counter()
    updated = Counter()
    flips = defaultdict(Counter)
    staged = []                                   # (src, dest) image copies
    per_split_new = defaultdict(list)

    for r in ann:
        sess = session_of(r["tactile_image"]) or r["pair_id"]
        dest = dest_of[sess]

        nat_rel = f"{args.category}/{r['natural_image']}"
        tac_rel = f"{args.category}/{r['tactile_image']}"
        pair_id = f"{nat_rel}::{tac_rel}"
        staged.append((collect / r["natural_image"], images / nat_rel))
        staged.append((collect / r["tactile_image"], images / tac_rel))

        # An existing pair is corrected in place, in whichever split holds it.
        target = where.get(pair_id, dest)
        idx = {(x["pair_id"], x["option_id"]): x for x in existing[target]}

        for opt in ACTIVE:
            label = int(r[opt])
            prov = {"label_verified": True,
                    "verified_at": r.get("saved_at"),
                    "verification_kind": "collected_step"}
            key = (pair_id, opt)
            if key in idx:
                old = int(idx[key]["label"])
                if old != label:
                    flips[target][opt] += 1
                    updated[target] += 1
                idx[key]["label"] = label
                idx[key].update(prov)
            else:
                row = {"pair_id": pair_id,
                       "natural_image": nat_rel, "tactile_image": tac_rel,
                       "category": args.category, "object": sess,
                       "option_id": opt, "label": label, "split": target,
                       "synthetic_corruption": bool(r.get("synthetic_corruption", False)),
                       "object_label": r.get("object", ""),
                       "iteration": r.get("iteration", ""),
                       "collected_from": str(collect.name),
                       **prov}
                per_split_new[target].append(row)
                added[target] += 1

    # ── report ───────────────────────────────────────────────────────────────
    print(f"sessions: {len(sessions)}   placement: " +
          "  ".join(f"{k}={v}" for k, v in sorted(Counter(dest_of.values()).items())))
    print(f"images to stage: {len(set(staged))} -> {images / args.category}/\n")

    print("ROWS")
    for split in ("train", "val", "test"):
        if added[split] or updated[split]:
            print(f"  {split:6} +{added[split]:4} new   ~{updated[split]:4} corrected")
    if not sum(added.values()) and not sum(updated.values()):
        print("  nothing to do — every iteration is already ingested with these labels")

    if sum(flips.values(), Counter()):
        print("\nLABEL CORRECTIONS to rows already present")
        for split in ("train", "val", "test"):
            if flips[split]:
                print(f"  {split:6} " + "  ".join(f"{o}={n}" for o, n in flips[split].items()))

    print("\nRESULTING SPLITS")
    for split in ("train", "val", "test"):
        rows = existing[split] + per_split_new[split]
        if not rows:
            continue
        pos = sum(1 for x in rows if x["label"])
        pairs = len({x["pair_id"] for x in rows})
        print(f"  {split:6} rows={len(rows):5} pairs={pairs:4} "
              f"positive={pos/len(rows):.1%}  always-clean={1-pos/len(rows):.1%}")

    if sum(added.values()):
        print("\nNOTE: new rows change the record set, so every cached feature file "
              "will re-extract\n      on the next train run (the fingerprint pins record "
              "identity and order).")
    if added["val"] or added["test"] or updated["val"] or updated["test"]:
        print("\nWARNING: an evaluation split is changing. Numbers already reported "
              "against v1\n         test are no longer comparable — requote them "
              "after the next scoring run.")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    # ── write ────────────────────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = splits.parent / f"{splits.name}_backup_{stamp}"
    shutil.copytree(splits, backup)
    print(f"\nbackup -> {backup}")

    n_copied = 0
    for src, dst in sorted(set(staged)):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            n_copied += 1
    print(f"staged {n_copied} new image(s) -> {images / args.category}/")

    for split in ("train", "val", "test"):
        rows = existing[split] + per_split_new[split]
        with open(splits / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "collect_dir": str(collect),
        "category": args.category,
        "iterations_ingested": len(ann),
        "sessions": len(sessions),
        "placement": dict(Counter(dest_of.values())),
        "rows_added": dict(added),
        "rows_corrected": dict(updated),
        "corrections_per_option": {s: dict(c) for s, c in flips.items()},
        "images_staged": n_copied,
    }
    (splits / "ingest_report.json").write_text(json.dumps(report, indent=2))
    print(f"written -> {splits}/  (+ ingest_report.json)")
    print("\nNEXT: retrain. Caches re-extract if rows were added.")


if __name__ == "__main__":
    main()
