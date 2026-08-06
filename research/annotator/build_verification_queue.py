#!/usr/bin/env python3
"""Build a label-verification queue for annotator.py (splits_v3 fresh start).

Produces a JSONL in the exact record shape annotator.py expects, so it can be
opened with:

    python research/annotator/annotator.py \\
        --annotations-file "$TACTILE_DATA_ROOT/verification_queue.jsonl" --share

ORDER (annotator.py lands on the first record with annotated=false, and walks
the file in order):

  1. PARTIAL-LABEL images FIRST. These are the "inferred positives": iteration
     images from an edit step, carrying a single option set to 1 because the
     annotator targeted that defect, and NOTHING known about the other four.
     They have never been fully labelled. Surfacing them first is the point of
     this pass — every one that gets completed converts a 1-option row into a
     5-option row.
  2. TEST finals — the set every reported number will be measured against.
  3. VAL finals — drives early stopping and threshold calibration.
  4. TRAIN finals — opt-in via --train-finals. Last because they are the most
     numerous and the least load-bearing: a wrong train label costs accuracy,
     a wrong test label costs a wrong conclusion.

Existing labels are PRE-TICKED so the pass is confirm-or-correct rather than
label-from-scratch. For partial records only the known option is ticked; the
other four are shown unticked and are genuinely unknown, not asserted negative.

Every record keeps `pair_id`, `split`, `queue_kind` and (for partials)
`known_option`, so apply_verified_labels.py can merge the result back into the
splits without ambiguity.

Pairs already carrying `label_verified` are skipped by default, so an
incremental pass never re-queues confirmed work. Pass --requeue-verified to
build a queue over everything regardless.

Usage:
  python research/annotator/build_verification_queue.py
  python research/annotator/build_verification_queue.py --no-val
  # the train-set pass, once the evaluation splits are done:
  python research/annotator/build_verification_queue.py \\
      --train-finals --no-val --no-partials --out "$TACTILE_DATA_ROOT/train_verification_queue.jsonl"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT  # noqa: E402

import argparse
import json
from collections import Counter, defaultdict

# annotator.py's OPTION_KEYS, in its order. noisy_background and non_conformant
# are deprecated (not in the deployed ALL_OPTIONS) but the app renders them, so
# the keys must exist. They are emitted as 0 and ignored on merge-back.
ANNOTATOR_KEYS = ["too_thick", "broken_lines", "missing_parts", "missing_texture",
                  "extra_parts", "noisy_background", "non_conformant"]
ACTIVE = ANNOTATOR_KEYS[:5]


def group_pairs(rows):
    """rows -> {pair_id: {meta..., labels: {option: 0/1}}} keeping option coverage."""
    out = {}
    for r in rows:
        e = out.setdefault(r["pair_id"], {
            "pair_id": r["pair_id"],
            "natural_image": r["natural_image"],
            "tactile_image": r["tactile_image"],
            "category": r.get("category", ""),
            "object": r.get("object", ""),
            "split": r.get("split", ""),
            "synthetic_corruption": bool(r.get("synthetic_corruption", False)),
            "labels": {},
        })
        e["labels"][r["option_id"]] = int(r["label"])
        # a pair counts as verified once any of its rows carries the stamp
        e["label_verified"] = e.get("label_verified", False) or bool(r.get("label_verified"))
    return out


def to_record(e, kind, known_option=None):
    rec = {
        "annotated": False,
        "category": e["category"] or ("gpt_generated" if "gpt_generated" in e["tactile_image"] else "library"),
        "object": e["object"] or "(unknown)",
        "natural_image": e["natural_image"],
        "tactile_image": e["tactile_image"],
        "pair_id": e["pair_id"],
        "split": e["split"],
        "queue_kind": kind,
        "synthetic_corruption": e["synthetic_corruption"],
    }
    if known_option:
        rec["known_option"] = known_option
        rec["options_unknown_at_queue_time"] = [o for o in ACTIVE if o != known_option]
    for k in ANNOTATOR_KEYS:
        rec[k] = int(e["labels"].get(k, 0))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default=None,
                    help="unified splits (default data/splits_v3_unified)")
    ap.add_argument("--out", default=None,
                    help="queue path (default $TACTILE_DATA_ROOT/verification_queue.jsonl)")
    ap.add_argument("--no-val", action="store_true", help="test + partials only")
    ap.add_argument("--no-partials", action="store_true")
    ap.add_argument("--train-finals", action="store_true",
                    help="also queue fully-labelled TRAIN pairs (the train-set pass)")
    ap.add_argument("--requeue-verified", action="store_true",
                    help="include pairs already stamped label_verified (default: skip)")
    args = ap.parse_args()

    splits = Path(args.splits_dir) if args.splits_dir else DATA_ROOT / "data" / "splits_v3_unified"
    out = Path(args.out) if args.out else DATA_ROOT / "verification_queue.jsonl"
    images = DATA_ROOT / "images"

    def load(name):
        p = splits / f"{name}.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    train, val, test = load("train"), load("val"), load("test")

    skipped = Counter()

    def wanted(e, split):
        """Skip pairs a human has already confirmed, unless asked not to."""
        if e.get("label_verified") and not args.requeue_verified:
            skipped[split] += 1
            return False
        return True

    train_pairs = group_pairs(train)

    # Partial-label records: fewer than the 5 active options present.
    partials = []
    if not args.no_partials:
        for pid, e in train_pairs.items():
            present = [o for o in ACTIVE if o in e["labels"]]
            if len(present) < len(ACTIVE) and wanted(e, "partial"):
                known = present[0] if len(present) == 1 else None
                partials.append(to_record(e, "partial_inferred", known))
        partials.sort(key=lambda r: r["tactile_image"])

    test_recs = [to_record(e, "test_final")
                 for e in group_pairs(test).values() if wanted(e, "test")]
    test_recs.sort(key=lambda r: (r["category"], r["tactile_image"]))
    val_recs = [] if args.no_val else [to_record(e, "val_final")
                                       for e in group_pairs(val).values() if wanted(e, "val")]
    val_recs.sort(key=lambda r: (r["category"], r["tactile_image"]))

    # Train finals: fully-labelled train pairs. Opt-in — this is the pass that
    # closes the last gap in verification coverage.
    train_recs = []
    if args.train_finals:
        for pid, e in train_pairs.items():
            present = [o for o in ACTIVE if o in e["labels"]]
            if len(present) == len(ACTIVE) and wanted(e, "train"):
                train_recs.append(to_record(e, "train_final"))
        train_recs.sort(key=lambda r: (r["category"], r["tactile_image"]))

    queue = partials + test_recs + val_recs + train_recs

    missing = [r for r in queue
               if not (images / r["natural_image"]).exists()
               or not (images / r["tactile_image"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} record(s) reference images that do not exist "
                         f"under {images}, e.g. {missing[0]['tactile_image']}")

    with open(out, "w") as f:
        for r in queue:
            f.write(json.dumps(r) + "\n")

    print(f"queue -> {out}   ({len(queue)} records, all annotated=false)\n")
    print(f"  1. partial_inferred : {len(partials):4}  <- surfaced FIRST, never fully labelled")
    print(f"  2. test_final       : {len(test_recs):4}")
    print(f"  3. val_final        : {len(val_recs):4}")
    print(f"  4. train_final      : {len(train_recs):4}"
          f"{'' if args.train_finals else '  (not requested — pass --train-finals)'}")
    print()
    if skipped:
        print(f"  skipped {sum(skipped.values())} already-verified pair(s): "
              + "  ".join(f"{k}={n}" for k, n in sorted(skipped.items()))
              + "   (--requeue-verified to include)")
        print()
    if partials:
        kc = Counter(r.get("known_option") for r in partials)
        print("  partials by the ONE option currently known (the other 4 are unknown):")
        for o, n in kc.most_common():
            print(f"     {str(o):<18} {n}")
        print()
    print("  open with:")
    print(f"    python research/annotator/annotator.py --annotations-file {out} --share")
    print("\n  NOTE: the app also renders noisy_background and non_conformant. Both are")
    print("  deprecated, are emitted as 0, and are ignored when labels are merged back.")


if __name__ == "__main__":
    main()
