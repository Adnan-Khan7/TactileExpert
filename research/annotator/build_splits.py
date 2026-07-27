#!/usr/bin/env python3
"""
Build fresh train/val/test splits from annotations.jsonl.

Usage (from the research workspace, default ~/vlm_finetune):
    python code/annotator/build_splits.py

Input:  annotations.jsonl  (987 pairs, pair-centric, 7 options)
Output: data/splits_v2/{train,val,test}.jsonl  (option-centric, 6 options)

Options included (6):
    too_thick, broken_lines, missing_parts, missing_texture,
    extra_parts, non_conformant

Skipped:
    noisy_background — only 22 positives across 987 pairs (~2.2%), insufficient
    to evaluate reliably in a 15% test split (~3 positives).

Split: 70 / 15 / 15, stratified by category, fixed seed = 42.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT  # noqa: E402

import json
import random
from pathlib import Path
from collections import defaultdict

ANNOTATIONS_FILE = DATA_ROOT / "annotations.jsonl"
OUT_DIR          = DATA_ROOT / "data/splits_v2"

ACTIVE_OPTIONS = [
    'too_thick', 'broken_lines', 'missing_parts', 'missing_texture',
    'extra_parts', 'non_conformant',
]

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
SEED       = 42


def main():
    # ── Load annotated pairs ──────────────────────────────────────────────────
    pairs = []
    with open(ANNOTATIONS_FILE) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get('annotated'):
                    pairs.append(r)

    print(f"Loaded {len(pairs)} annotated pairs")

    # ── Stratified split by category ─────────────────────────────────────────
    by_cat = defaultdict(list)
    for p in pairs:
        by_cat[p['category']].append(p)

    rng = random.Random(SEED)
    train_pairs, val_pairs, test_pairs = [], [], []

    print("\nPer-category split:")
    for cat, cat_pairs in sorted(by_cat.items()):
        rng.shuffle(cat_pairs)
        n        = len(cat_pairs)
        n_train  = int(n * TRAIN_FRAC)
        n_val    = int(n * VAL_FRAC)
        n_test   = n - n_train - n_val
        train_pairs.extend(cat_pairs[:n_train])
        val_pairs.extend(cat_pairs[n_train:n_train + n_val])
        test_pairs.extend(cat_pairs[n_train + n_val:])
        short = cat.replace('_and_', '+').replace('_', ' ')
        print(f"  {short:42s}  train={n_train:3d}  val={n_val:3d}  test={n_test:3d}")

    print(f"\nTotal  train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")

    # ── Write option-centric records ──────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    for split_name, split_pairs in [('train', train_pairs),
                                     ('val',   val_pairs),
                                     ('test',  test_pairs)]:
        records = []
        for p in split_pairs:
            for opt in ACTIVE_OPTIONS:
                records.append({
                    "pair_id":       p['pair_id'],
                    "natural_image": p['natural_image'],
                    "tactile_image": p['tactile_image'],
                    "category":      p['category'],
                    "object":        p['object'],
                    "option_id":     opt,
                    "label":         int(p.get(opt, 0)),
                    "split":         split_name,
                })

        out_path = OUT_DIR / f"{split_name}.jsonl"
        with open(out_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')

        n_pairs = len(split_pairs)
        print(f"{split_name}  ({n_pairs} pairs → {len(records)} records)")
        for opt in ACTIVE_OPTIONS:
            opt_recs = [r for r in records if r['option_id'] == opt]
            n_pos    = sum(1 for r in opt_recs if r['label'] == 1)
            print(f"  {opt:20s}  {n_pos:4d} / {len(opt_recs):4d}  "
                  f"({100*n_pos/len(opt_recs):.1f}% positive)")
        print()

    print(f"Wrote splits → {OUT_DIR}/")
    print("Next: python code/baselines/eval_on_new_splits.py")


if __name__ == '__main__':
    main()
