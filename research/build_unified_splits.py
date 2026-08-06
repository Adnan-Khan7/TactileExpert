#!/usr/bin/env python3
"""Build ONE unified set from the library and generated corpora (splits_v3).

Replaces the two-frozen-sets design (library test.jsonl + GPT gpt_test.jsonl)
with a single train/val/test split spanning both domains. Motivation: the
dual-set gate rule has been ambiguous in code and docstring for weeks, and the
30-pair GPT holdout is so class-starved (8 positives in 150 decisions) that no
judge clears a degenerate always-clean baseline on it.

Writes to data/splits_v3_unified/ — it does NOT touch data/splits_v2/, so the
corpus every deployed checkpoint was trained on stays reproducible.

METHOD
------
Rows are RE-ASSIGNED, not rebuilt: every row already exists in splits_v2 with
verified labels, so this only rewrites the `split` field. Nothing is relabelled
and no label is inferred here.

  * LIBRARY rows keep their existing train/val/test assignment. Re-splitting
    them would invalidate every number ever reported on the library test set for
    no benefit.
  * GENERATED sessions are split --train/--val/--test by session (never by row),
    deterministically under --seed.
  * FINALS ONLY reach val/test. The 124 inferred positives are single-option,
    positive-only, and labelled by the ACTION the annotator took rather than the
    defect observed, so they are training signal, not evaluation ground truth.
    Inferred rows whose session lands in val/test are DROPPED, not moved.

TWO THINGS THIS SCRIPT CANNOT FIX, both reported in the summary
--------------------------------------------------------------
1. CONTAMINATION UNTIL RETRAIN. The current 154 generated sessions sit at
   124 train / 30 holdout (~80/20). Moving to 70/30 promotes ~16 sessions from
   train into val/test — and the DEPLOYED fleet was trained on those. Any
   evaluation of the CURRENT checkpoints against this split is therefore
   optimistic. Only numbers from models retrained AFTER this split are clean.
   Affected rows are tagged `was_trained_on: true` so the contamination is
   auditable rather than invisible.
2. The existing 30 holdout sessions are PINNED to test. They have already gated
   deployment decisions; returning any of them to train would be the worse form
   of the same leak.

Usage:
  python research/build_unified_splits.py                  # 70/15/15, seed 42
  python research/build_unified_splits.py --test 0.30 --val 0.0   # literal 70/30
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from _paths import DATA_ROOT  # noqa: E402

import argparse
import json
import random
import re
from collections import Counter, defaultdict

SRC = DATA_ROOT / "data" / "splits_v2"
OPTIONS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]


def session_of(row):
    """pair_YYYYMMDD_HHMMSS for generated rows, else None."""
    m = re.search(r"pair_(2026\d{4}_\d{6})", row.get("tactile_image", ""))
    return m.group(1) if m else None


def is_final(row):
    """Finals are .../tactile.jpg; iteration images are .../tactile_iterN.jpg."""
    return Path(row.get("tactile_image", "")).name == "tactile.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15,
                    help="share of GENERATED sessions for val (0 to disable)")
    ap.add_argument("--test", type=float, default=0.15,
                    help="share of GENERATED sessions for test (beyond the pinned holdout)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir) if args.out_dir else DATA_ROOT / "data" / "splits_v3_unified"

    rows, pinned_test = [], set()
    for split in ("train", "val", "test", "gpt_test"):
        p = SRC / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["_src_split"] = split
            rows.append(r)
            if split == "gpt_test":
                s = session_of(r)
                if s:
                    pinned_test.add(s)

    gen_rows = [r for r in rows if session_of(r)]
    lib_rows = [r for r in rows if not session_of(r)]

    # Sessions contributing a FINAL are the only ones eligible for val/test.
    final_sessions = sorted({session_of(r) for r in gen_rows if is_final(r)})
    assignable = [s for s in final_sessions if s not in pinned_test]
    random.Random(args.seed).shuffle(assignable)

    n_val = int(round(args.val * len(final_sessions)))
    n_test_extra = max(0, int(round(args.test * len(final_sessions))) - len(pinned_test))

    split_of = {s: "test" for s in pinned_test}
    for s in assignable[:n_val]:
        split_of[s] = "val"
    for s in assignable[n_val:n_val + n_test_extra]:
        split_of[s] = "test"
    for s in assignable[n_val + n_test_extra:]:
        split_of[s] = "train"

    buckets = defaultdict(list)
    n_dropped_inferred = 0
    promoted = set()

    for r in lib_rows:
        r = {k: v for k, v in r.items() if k != "_src_split"}
        buckets[r["split"]].append(r)

    for r in gen_rows:
        s = session_of(r)
        dest = split_of.get(s, "train")
        src = r.pop("_src_split")
        if dest != "train" and not is_final(r):
            n_dropped_inferred += 1          # inferred positives never evaluate
            continue
        # Was this row inside the corpus the deployed fleet trained on?
        if dest != "train" and src == "train":
            r["was_trained_on"] = True
            promoted.add(s)
        r["split"] = dest
        r.setdefault("synthetic_corruption", False)
        buckets[dest].append(r)

    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        with open(out / f"{split}.jsonl", "w") as f:
            for r in buckets[split]:
                f.write(json.dumps(r) + "\n")

    # ── report ────────────────────────────────────────────────────────────────
    print(f"unified splits -> {out}/   (seed {args.seed}, "
          f"generated val={args.val:.0%} test={args.test:.0%})\n")
    print(f"generated sessions with a final: {len(final_sessions)}   "
          f"pinned holdout: {len(pinned_test)}")
    print(f"inferred-positive rows dropped from val/test: {n_dropped_inferred}")
    print(f"sessions promoted train->val/test (DEPLOYED FLEET SAW THESE): {len(promoted)}\n")

    for split in ("train", "val", "test"):
        rs = buckets[split]
        lib = [r for r in rs if not session_of(r)]
        gen = [r for r in rs if session_of(r)]
        print(f"=== {split}  rows={len(rs)}  "
              f"pairs={len({r['pair_id'] for r in rs})}  "
              f"(library {len({r['pair_id'] for r in lib})} / "
              f"generated {len({r['pair_id'] for r in gen})}) ===")
        tot = pos = 0
        for o in OPTIONS:
            n = sum(1 for r in rs if r["option_id"] == o)
            p = sum(1 for r in rs if r["option_id"] == o and r["label"])
            tot += n
            pos += p
            print(f"   {o:<17} {p:>5}/{n:<5} ({p/n:>5.1%} positive)" if n else
                  f"   {o:<17} absent")
        if tot:
            print(f"   {'ALL':<17} {pos:>5}/{tot:<5} ({pos/tot:>5.1%} positive)")
            print(f"   always-clean baseline on this split: "
                  f"{(tot-pos)/tot:.1%}  <- the bar any judge must beat")
        print()

    stats = {
        "seed": args.seed, "val_share": args.val, "test_share": args.test,
        "pinned_holdout_sessions": len(pinned_test),
        "promoted_train_to_eval": sorted(promoted),
        "inferred_rows_dropped": n_dropped_inferred,
        "per_split": {
            s: {"rows": len(buckets[s]),
                "pairs": len({r["pair_id"] for r in buckets[s]}),
                "positives": {o: sum(1 for r in buckets[s]
                                     if r["option_id"] == o and r["label"])
                              for o in OPTIONS}}
            for s in ("train", "val", "test")},
    }
    (out / "build_stats.json").write_text(json.dumps(stats, indent=2))

    if promoted:
        print(f"⚠ {len(promoted)} promoted sessions are tagged `was_trained_on: true`.")
        print("  Numbers from the CURRENT checkpoints on this split are optimistic.")
        print("  Retrain before reporting anything from val/test.")


if __name__ == "__main__":
    main()
