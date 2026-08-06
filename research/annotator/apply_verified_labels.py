#!/usr/bin/env python3
"""Merge verified labels from the annotator queue back into the v1 splits.

Reads  $TACTILE_DATA_ROOT/verification_queue.jsonl   (annotator.py output)
Writes data/splits_v3_unified/{train,val,test}.jsonl

DRY RUN BY DEFAULT. Nothing is written without --apply.

WHAT IT DOES
------------
Only records with annotated=true are merged; the rest are left exactly as they
are, so this is safe to run repeatedly mid-verification.

  * test_final / val_final  — overwrite the 5 option labels on existing rows.
  * partial_inferred        — these had ONE option known (label=1, from the edit
    step) and four UNKNOWN. A verified partial becomes a full 5-option row, so
    the train row count GROWS. Options the annotator left unticked are now
    genuine negatives, asserted by a human, not absences.

RETIRED OPTIONS ARE DROPPED
---------------------------
The splits still carry one `non_conformant` row per library pair (688/146/153),
left over from the pre-2026-08-03 taxonomy. It is not in ALL_OPTIONS, no
evaluator scores it, and the verification UI never asks about it — so those rows
can never be verified and must not reach a trainer. They are dropped here rather
than filtered downstream, so the written files match DATASET_V1.md row-for-row
and no consumer has to remember. The pre-strip copy survives in the backup.

WHY IT REPORTS CHANGES SEPARATELY
---------------------------------
After the retrain you will compare v1 numbers against the pre-v1 fleet. Two
different things moved: the LABELS changed (this script) and the MODEL changed
(the retrain). If you cannot separate them you cannot attribute the delta. So
every label flip is counted, per option and per split, and written to
label_change_report.json.

Usage:
  python research/annotator/apply_verified_labels.py            # dry run
  python research/annotator/apply_verified_labels.py --apply
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT  # noqa: E402

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime

ACTIVE = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=None)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args()

    queue_p = Path(args.queue) if args.queue else DATA_ROOT / "verification_queue.jsonl"
    splits = Path(args.splits_dir) if args.splits_dir else DATA_ROOT / "data" / "splits_v3_unified"

    q = [json.loads(l) for l in queue_p.read_text().splitlines() if l.strip()]
    verified = {r["pair_id"]: r for r in q if r.get("annotated")}
    print(f"queue: {len(q)} records, {len(verified)} verified\n")
    if not verified:
        raise SystemExit("nothing verified yet — run the annotator first")

    by_kind = Counter(r["queue_kind"] for r in verified.values())
    print("verified by kind:", dict(by_kind), "\n")

    changes = defaultdict(Counter)     # split -> option -> n flipped
    flips = []                         # detailed record of every flip
    added_rows = Counter()             # split -> rows added by completing partials
    dropped_rows = defaultdict(Counter)  # split -> retired option -> n dropped
    touched_pairs = Counter()

    new_splits = {}
    for split in ("train", "val", "test"):
        p = splits / f"{split}.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

        # drop retired options before anything indexes them
        for r in rows:
            if r["option_id"] not in ACTIVE:
                dropped_rows[split][r["option_id"]] += 1
        rows = [r for r in rows if r["option_id"] in ACTIVE]

        # index existing rows by (pair_id, option)
        idx = {(r["pair_id"], r["option_id"]): r for r in rows}
        pairs_here = {r["pair_id"] for r in rows}

        for pid, vrec in verified.items():
            if pid not in pairs_here:
                continue
            touched_pairs[split] += 1
            template = next(r for r in rows if r["pair_id"] == pid)
            # Verification provenance, carried on every row a human confirmed.
            # A future train-set pass appends to the same fields, so coverage is
            # always answerable from the dataset alone — no cross-referencing the
            # queue, which is a working file and will be rebuilt.
            prov = {"label_verified":    True,
                    "verified_at":       vrec.get("annotated_at"),
                    "verification_kind": vrec["queue_kind"]}
            for opt in ACTIVE:
                new_label = int(vrec.get(opt, 0))
                key = (pid, opt)
                if key in idx:
                    old = int(idx[key]["label"])
                    if old != new_label:
                        changes[split][opt] += 1
                        flips.append({"split": split, "pair_id": pid, "option": opt,
                                      "from": old, "to": new_label,
                                      "kind": vrec["queue_kind"]})
                    idx[key]["label"] = new_label
                    idx[key].update(prov)
                else:
                    # completing a partial: option had no row at all
                    row = {k: v for k, v in template.items()
                           if k not in ("option_id", "label")}
                    row["option_id"] = opt
                    row["label"] = new_label
                    row.update(prov)
                    row["added_by_verification"] = True
                    rows.append(row)
                    idx[key] = row
                    added_rows[split] += 1

        new_splits[split] = rows

    # ── report ────────────────────────────────────────────────────────────────
    n_dropped = sum(sum(c.values()) for c in dropped_rows.values())
    if n_dropped:
        retired = sorted({o for c in dropped_rows.values() for o in c})
        print(f"RETIRED ROWS DROPPED: {n_dropped}  ({', '.join(retired)})")
        for split in ("train", "val", "test"):
            if dropped_rows[split]:
                print(f"  {split:6} " +
                      "  ".join(f"{o}={n}" for o, n in sorted(dropped_rows[split].items())))
        print()

    print("PAIRS TOUCHED PER SPLIT:", dict(touched_pairs))
    print("ROWS ADDED by completing partials:", dict(added_rows), "\n")

    total_flips = sum(sum(c.values()) for c in changes.values())
    print(f"LABEL FLIPS: {total_flips}")
    for split in ("train", "val", "test"):
        if not changes[split]:
            continue
        n = sum(changes[split].values())
        print(f"  {split:6} {n:4}   " +
              "  ".join(f"{o}={changes[split][o]}" for o in ACTIVE if changes[split][o]))
    if total_flips:
        d = Counter((f["from"], f["to"]) for f in flips)
        print(f"  direction: 0->1 {d[(0,1)]}   1->0 {d[(1,0)]}")
    print()

    for split in ("train", "val", "test"):
        rows = new_splits[split]
        pos = sum(1 for r in rows if r["label"])
        print(f"{split:6} rows={len(rows):5} pairs={len({r['pair_id'] for r in rows}):4} "
              f"positive={pos/len(rows):.1%}  always-clean={1-pos/len(rows):.1%}")

    # ── verification coverage ────────────────────────────────────────────────
    # Which pairs a human has actually confirmed, per split. Unverified pairs
    # still carry their original v2 labels. Evaluation splits should read 100%;
    # train will not until the train-set pass runs.
    print("\nVERIFICATION COVERAGE (pairs)")
    coverage = {}
    for split in ("train", "val", "test"):
        rows = new_splits[split]
        by_pair = defaultdict(bool)
        for r in rows:
            by_pair[r["pair_id"]] |= bool(r.get("label_verified"))
        n_ver = sum(1 for v in by_pair.values() if v)
        n_tot = len(by_pair)
        kinds = Counter(r.get("verification_kind") for r in rows
                        if r.get("verification_kind"))
        coverage[split] = {"verified_pairs": n_ver, "total_pairs": n_tot,
                           "by_kind": dict(kinds)}
        bar = "COMPLETE" if n_ver == n_tot else f"{n_tot - n_ver} unverified"
        print(f"  {split:6} {n_ver:4}/{n_tot:4} pairs  ({n_ver/n_tot:5.1%})   {bar}")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "verified_records": len(verified),
        "verified_by_kind": dict(by_kind),
        "pairs_touched": dict(touched_pairs),
        "rows_added_by_completing_partials": dict(added_rows),
        "retired_rows_dropped": {s: dict(c) for s, c in dropped_rows.items()},
        "verification_coverage": coverage,
        "total_label_flips": total_flips,
        "flips_per_split_option": {s: dict(c) for s, c in changes.items()},
        "flips": flips,
    }

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = splits.parent / f"{splits.name}_backup_{stamp}"
    shutil.copytree(splits, backup)
    print(f"\nbackup -> {backup}")

    for split, rows in new_splits.items():
        with open(splits / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (splits / "label_change_report.json").write_text(json.dumps(report, indent=2))
    print(f"written -> {splits}/  (+ label_change_report.json)")
    print("\nNEXT: retrain on this split, then compare with mcnemar_gate.py.")
    print("When reporting the delta, separate LABEL changes (above) from MODEL changes.")


if __name__ == "__main__":
    main()
