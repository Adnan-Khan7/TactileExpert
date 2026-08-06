#!/usr/bin/env python3
"""Re-derive the val/test boundary so per-option positive rates match.

Reads  data/splits_v3_unified/{train,val,test}.jsonl
Writes the same three files (train is passed through untouched).

DRY RUN BY DEFAULT. Nothing is written without --apply.

SCOPE — verified evaluation pairs only
--------------------------------------
Only pairs currently in val or test are re-assigned, and only those a human has
confirmed (`label_verified`). TRAIN IS NEVER TOUCHED.

That restriction is the whole point. Re-stratifying across all 1,275 pairs would
pull unverified train pairs into the evaluation splits, which throws away the
guarantee the verification pass was run to buy: every pair a number is measured
on has been confirmed by a human. It would also demote verified pairs into
train, wasting that effort. Once the train-set verification pass completes, the
pool can widen — pass --pool all then.

METHOD
------
Pairs are bucketed by their 5-bit label signature (32 possible buckets), and
each bucket is divided between val and test in the target proportion using
largest-remainder allocation. Matching per-option rates then falls out by
construction rather than by search, and co-occurrence structure is preserved
too — a bucket is a joint pattern, not five marginals.

Ties and within-bucket ordering are resolved by a seeded shuffle, so the result
is reproducible.

WHAT THIS DELIBERATELY DOES NOT PRESERVE
----------------------------------------
Two rules from build_unified_splits.py are retired here by decision:
  * library rows keeping their splits_v2 assignment — that existed only to keep
    old library-test numbers comparable, and those numbers are being discarded;
  * the 30 GPT holdout sessions pinned to test — same reason.
Both are recorded in the report so the change is auditable.

Usage:
  python research/restratify_v1.py                 # dry run
  python research/restratify_v1.py --apply
  python research/restratify_v1.py --val-frac 0.5  # even split instead of 174/183
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_ROOT  # noqa: E402

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime

ACTIVE = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
SEED = 42


def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def group(rows):
    """rows -> {pair_id: [row, ...]}, order preserved."""
    out = defaultdict(list)
    for r in rows:
        out[r["pair_id"]].append(r)
    return out


def signature(rows):
    """The 5-bit label pattern for a pair, as a tuple."""
    lab = {r["option_id"]: int(r["label"]) for r in rows}
    return tuple(lab.get(o, 0) for o in ACTIVE)


def rates(pairs):
    """Per-option positive rate over a list of pair row-lists."""
    out = {}
    for i, o in enumerate(ACTIVE):
        sigs = [signature(rs)[i] for rs in pairs]
        out[o] = sum(sigs) / len(sigs) if sigs else 0.0
    return out


def spread(a, b):
    """Largest per-option gap between two splits."""
    return max(abs(a[o] - b[o]) for o in ACTIVE)


def report_rates(label, val_pairs, test_pairs):
    rv, rt = rates(val_pairs), rates(test_pairs)
    print(f"\n{label}")
    print(f"  {'option':<18}{'val':>10}{'test':>10}{'gap':>10}")
    for o in ACTIVE:
        print(f"  {o:<18}{rv[o]:>9.1%}{rt[o]:>10.1%}{abs(rv[o]-rt[o]):>10.1%}")
    print(f"  {'MAX GAP':<18}{'':>10}{'':>10}{spread(rv, rt):>10.1%}")
    return rv, rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--val-frac", type=float, default=None,
                    help="share of the eval pool going to val (default: keep current ratio)")
    ap.add_argument("--pool", choices=["verified", "all"], default="verified",
                    help="'verified' (default) re-assigns only confirmed eval pairs; "
                         "'all' also admits unverified ones — only sensible once the "
                         "train verification pass is done")
    ap.add_argument("--allow-partial", action="store_true",
                    help="run even while some eval pairs are unverified (default: refuse)")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args()

    splits = Path(args.splits_dir) if args.splits_dir else DATA_ROOT / "data" / "splits_v3_unified"

    train_rows = load(splits / "train.jsonl")
    val_rows = load(splits / "val.jsonl")
    test_rows = load(splits / "test.jsonl")

    val_g, test_g = group(val_rows), group(test_rows)
    print(f"train {len(train_rows):5} rows / {len(group(train_rows)):4} pairs   (untouched)")
    print(f"val   {len(val_rows):5} rows / {len(val_g):4} pairs")
    print(f"test  {len(test_rows):5} rows / {len(test_g):4} pairs")

    # ── build the pool ───────────────────────────────────────────────────────
    def is_verified(rs):
        return any(r.get("label_verified") for r in rs)

    eval_pairs = {**val_g, **test_g}
    origin = {pid: "val" for pid in val_g} | {pid: "test" for pid in test_g}

    if args.pool == "verified":
        pool = {pid: rs for pid, rs in eval_pairs.items() if is_verified(rs)}
        held = {pid: rs for pid, rs in eval_pairs.items() if not is_verified(rs)}
    else:
        pool, held = eval_pairs, {}

    print(f"\npool: {len(pool)} pairs re-assignable, {len(held)} held in place")
    if held:
        by_split = Counter(origin[pid] for pid in held)
        print(f"  unverified, staying put: {dict(by_split)}")

    if not pool:
        raise SystemExit("nothing to re-stratify — run apply_verified_labels.py --apply first")

    # Held pairs cannot move, so they drag their split's rates with them and the
    # balancing can only compensate with the pool. Running mid-verification
    # therefore produces a lopsided, weakly-balanced split that looks finished.
    if held and not args.allow_partial:
        raise SystemExit(
            f"\nREFUSING: {len(held)} evaluation pair(s) are still unverified.\n"
            "  Held pairs are frozen in place, so the balance achievable is only\n"
            "  approximate and the val/test sizes will be skewed.\n"
            "  Finish verification, re-run apply_verified_labels.py --apply, then\n"
            "  re-run this. Use --allow-partial to override for a preview.")

    # ── sanity: every pooled pair must carry all 5 options ───────────────────
    bad = [pid for pid, rs in pool.items()
           if {r["option_id"] for r in rs} != set(ACTIVE)]
    if bad:
        raise SystemExit(f"{len(bad)} pooled pair(s) are not fully labelled, e.g. {bad[0]}")

    before_val = [rs for pid, rs in eval_pairs.items() if origin[pid] == "val"]
    before_test = [rs for pid, rs in eval_pairs.items() if origin[pid] == "test"]
    rv0, rt0 = report_rates("BEFORE", before_val, before_test)

    # ── target sizes ─────────────────────────────────────────────────────────
    held_val = sum(1 for pid in held if origin[pid] == "val")
    held_test = sum(1 for pid in held if origin[pid] == "test")
    if args.val_frac is not None:
        val_frac = args.val_frac
    else:
        val_frac = len(val_g) / (len(val_g) + len(test_g))
    # Size against the FINAL split, not the pool: held pairs already occupy
    # seats, so the pool has to make up the difference for the ratio to land.
    target_val = min(max(round(len(eval_pairs) * val_frac) - held_val, 0), len(pool))
    print(f"\ntarget: val_frac={val_frac:.4f} -> final val={target_val + held_val} / "
          f"test={len(eval_pairs) - target_val - held_val}"
          f"   (pool {target_val}/{len(pool)-target_val}, held {held_val}/{held_test})")

    # ── signature-stratified largest-remainder allocation ────────────────────
    buckets = defaultdict(list)
    for pid, rs in pool.items():
        buckets[signature(rs)].append(pid)

    rng = random.Random(SEED)
    for pids in buckets.values():
        pids.sort()             # deterministic base order
        rng.shuffle(pids)

    # Fraction the POOL must run at for the FINAL ratio to land — not val_frac,
    # which ignores the seats held pairs already occupy. Identical to val_frac
    # in the normal case where nothing is held.
    pool_frac = target_val / len(pool)
    exact = {sig: len(pids) * pool_frac for sig, pids in buckets.items()}
    alloc = {sig: int(v) for sig, v in exact.items()}

    # Hand out remaining val seats to the largest fractional parts, skipping
    # buckets that are already full, until the target is met exactly.
    order = sorted(buckets, key=lambda s: (-(exact[s] - alloc[s]), s))
    i = 0
    while sum(alloc.values()) < target_val:
        sig = order[i % len(order)]
        if alloc[sig] < len(buckets[sig]):
            alloc[sig] += 1
        i += 1
        if i > len(order) * (max(len(p) for p in buckets.values()) + 1):
            break   # every bucket saturated; cannot place more
    assert sum(alloc.values()) == target_val, \
        f"allocation {sum(alloc.values())} != target {target_val}"

    new_split = {}
    for sig, pids in buckets.items():
        n_val = alloc[sig]
        for pid in pids[:n_val]:
            new_split[pid] = "val"
        for pid in pids[n_val:]:
            new_split[pid] = "test"
    for pid in held:
        new_split[pid] = origin[pid]

    after_val = [rs for pid, rs in eval_pairs.items() if new_split[pid] == "val"]
    after_test = [rs for pid, rs in eval_pairs.items() if new_split[pid] == "test"]
    rv1, rt1 = report_rates("AFTER", after_val, after_test)

    moved = sum(1 for pid in eval_pairs if new_split[pid] != origin[pid])
    print(f"\npairs moved: {moved} / {len(eval_pairs)}"
          f"   ({sum(1 for p in eval_pairs if origin[p]=='val'  and new_split[p]=='test')} val->test, "
          f"{sum(1 for p in eval_pairs if origin[p]=='test' and new_split[p]=='val')} test->val)")
    print(f"final sizes: val={len(after_val)} pairs, test={len(after_test)} pairs")
    print(f"MAX PER-OPTION GAP: {spread(rv0, rt0):.1%}  ->  {spread(rv1, rt1):.1%}")

    # retired-rule audit
    pinned = [pid for pid in eval_pairs if "gpt_generated" in pid]
    pinned_moved = sum(1 for pid in pinned
                       if origin[pid] == "test" and new_split[pid] == "val")
    print(f"\nretired rules: {pinned_moved} formerly-pinned GPT holdout pair(s) "
          f"moved test->val (rule 3 cancelled by decision)")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "pool": args.pool,
        "val_frac": val_frac,
        "pairs_pooled": len(pool),
        "pairs_held_unverified": len(held),
        "pairs_moved": moved,
        "sizes": {"val": len(after_val), "test": len(after_test)},
        "rates_before": {"val": rv0, "test": rt0, "max_gap": spread(rv0, rt0)},
        "rates_after": {"val": rv1, "test": rt1, "max_gap": spread(rv1, rt1)},
        "gpt_holdout_pairs_moved_out_of_test": pinned_moved,
        "retired_rules": ["library rows keep splits_v2 assignment",
                          "30 GPT holdout sessions pinned to test"],
        "assignments": {pid: new_split[pid] for pid in sorted(eval_pairs)},
    }

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = splits.parent / f"{splits.name}_backup_restratify_{stamp}"
    shutil.copytree(splits, backup)
    print(f"\nbackup -> {backup}")

    out = {"val": [], "test": []}
    for pid, rs in eval_pairs.items():
        dest = new_split[pid]
        for r in rs:
            r["split"] = dest
            out[dest].append(r)

    for name, rows in out.items():
        with open(splits / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (splits / "restratify_report.json").write_text(json.dumps(report, indent=2))
    print(f"written -> {splits}/  (+ restratify_report.json)   train.jsonl untouched")


if __name__ == "__main__":
    main()
