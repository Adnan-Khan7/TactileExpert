#!/usr/bin/env python3
"""
Build annotations.jsonl — master pair list for the TactileEval annotation pass.

Run once before starting the annotator:
    python code/annotator/prepare_annotations.py

Output: <TACTILE_DATA_ROOT>/annotations.jsonl

Label priority (highest wins):
  1. Cleaned splits (train/val/test.jsonl)  — for too_thick, broken_lines,
                                               missing_parts, missing_texture, extra_parts
  2. records_approved.jsonl (majority vote) — same options
  3. newly_graded_records.jsonl             — same options
  4. Default 0                              — noisy_background, non_conformant always 0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT  # noqa: E402

import json
import re
from pathlib import Path

RESEARCH_BASE  = DATA_ROOT
IMAGES_BASE    = RESEARCH_BASE / "images"
OUTPUT         = RESEARCH_BASE / "annotations.jsonl"

SPLITS_DIR     = RESEARCH_BASE / "TactileEval_Context_And_Data/data/splits"
APPROVED       = Path("/home/ak1998/scratch/vlm_finetune/archive/CONTEXT"
                      "/data_approved_only_step/records_approved.jsonl")
NEWLY_GRADED   = RESEARCH_BASE / "TactileEval_Context_And_Data/data/grading/newly_graded_records.jsonl"

# Options that may exist in source files
SOURCE_OPTIONS = ['too_thick', 'broken_lines', 'missing_parts',
                  'missing_texture', 'extra_parts']
# Full set in the new schema
ALL_OPTIONS    = SOURCE_OPTIONS + ['noisy_background', 'non_conformant']

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def extract_primary_index(filename: str) -> int | None:
    """Return the first integer found in a filename (without extension)."""
    stem = Path(filename).stem
    nums = re.findall(r'\d+', stem)
    return int(nums[0]) if nums else None


def enumerate_pairs() -> list[dict]:
    """
    Walk IMAGES_BASE and yield one dict per valid (natural, tactile) pair.
    Pairing rule: tactile file N.ext <-> natural file whose first integer is N.
    When multiple naturals share the same primary index, prefer the shortest filename.
    """
    pairs = []
    for cat_dir in sorted(IMAGES_BASE.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith('.'):
            continue
        category = cat_dir.name
        for obj_dir in sorted(cat_dir.iterdir()):
            if not obj_dir.is_dir() or obj_dir.name.startswith('.'):
                continue
            obj = obj_dir.name
            nat_dir = obj_dir / "Natural"
            tac_dir = obj_dir / "Tactile"
            if not nat_dir.is_dir() or not tac_dir.is_dir():
                continue

            # Index natural images by primary integer: idx -> filename (shortest wins)
            nat_by_idx: dict[int, str] = {}
            for f in nat_dir.iterdir():
                if f.suffix.lower() not in IMAGE_EXTS or f.name.startswith('.'):
                    continue
                idx = extract_primary_index(f.name)
                if idx is None:
                    continue
                if idx not in nat_by_idx or len(f.name) < len(nat_by_idx[idx]):
                    nat_by_idx[idx] = f.name

            # For each tactile, find matching natural
            for tac_f in sorted(tac_dir.iterdir(), key=lambda x: extract_primary_index(x.name) or 0):
                if tac_f.suffix.lower() not in IMAGE_EXTS or tac_f.name.startswith('.'):
                    continue
                idx = extract_primary_index(tac_f.name)
                if idx is None:
                    continue
                nat_name = nat_by_idx.get(idx)
                if nat_name is None:
                    continue

                pairs.append({
                    "category": category,
                    "object": obj,
                    "natural_image": f"{category}/{obj}/Natural/{nat_name}",
                    "tactile_image": f"{category}/{obj}/Tactile/{tac_f.name}",
                    "tac_stem": Path(tac_f.name).stem,
                })
    return pairs


def build_label_index() -> dict[tuple, dict[str, int]]:
    """
    Build lookup: (category, object, tac_stem) -> {option: label}

    tac_stem is used as the key because tactile filenames are consistently
    just integers (1, 2, …), avoiding natural-filename variation issues.
    """
    # intermediate: key -> {option -> (label, priority)}
    raw: dict[tuple, dict[str, tuple]] = {}

    def add(cat, obj, tac_stem, option, label, priority):
        key = (cat, obj, str(tac_stem))
        if key not in raw:
            raw[key] = {}
        existing = raw[key].get(option)
        if existing is None or existing[1] > priority:
            raw[key][option] = (int(label), priority)

    def ingest(path: Path, priority: int):
        if not path.exists():
            print(f"  [skip] not found: {path}")
            return
        count = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                pid = r.get('pair_id', '')
                opt = r.get('option_id', '')
                lbl = r.get('label', 0)
                if not pid or opt not in SOURCE_OPTIONS:
                    continue
                parts = pid.split('::')
                if len(parts) != 2:
                    continue
                tac_parts = parts[1].split('/')
                if len(tac_parts) < 4:
                    continue
                cat, obj, tac_name = tac_parts[0], tac_parts[1], tac_parts[3]
                tac_stem = Path(tac_name).stem
                add(cat, obj, tac_stem, opt, lbl, priority)
                count += 1
        print(f"  loaded {count} option records from {path.name}")

    ingest(NEWLY_GRADED, 3)
    ingest(APPROVED, 2)
    for split in ['train', 'val', 'test']:
        ingest(SPLITS_DIR / f"{split}.jsonl", 1)

    # Collapse to {key: {option: label}}
    return {key: {opt: val for opt, (val, _) in opts.items()}
            for key, opts in raw.items()}


def main():
    if OUTPUT.exists():
        answer = input(f"\n{OUTPUT} already exists. Overwrite? [y/N]: ").strip().lower()
        if answer != 'y':
            print("Aborted.")
            return

    print("\nLoading existing labels...")
    label_index = build_label_index()
    print(f"  Label index covers {len(label_index)} (category, object, tactile) keys")

    print("\nEnumerating image pairs...")
    pairs = enumerate_pairs()
    print(f"  Found {len(pairs)} pairs in images directory")

    records = []
    with_labels = 0
    for p in pairs:
        key = (p['category'], p['object'], p['tac_stem'])
        src = label_index.get(key, {})
        has_label = any(src.get(opt, 0) for opt in SOURCE_OPTIONS)
        if has_label:
            with_labels += 1

        record = {
            "pair_id":         f"{p['natural_image']}::{p['tactile_image']}",
            "natural_image":   p['natural_image'],
            "tactile_image":   p['tactile_image'],
            "category":        p['category'],
            "object":          p['object'],
            "too_thick":       src.get('too_thick',       0),
            "broken_lines":    src.get('broken_lines',    0),
            "missing_parts":   src.get('missing_parts',   0),
            "missing_texture": src.get('missing_texture', 0),
            "extra_parts":     src.get('extra_parts',     0),
            "noisy_background": 0,
            "non_conformant":  0,
            "annotated":       False,
            "annotated_at":    None,
        }
        records.append(record)

    print(f"\nWriting {len(records)} records → {OUTPUT}")
    with open(OUTPUT, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    print(f"\nSummary")
    print(f"  Total pairs:              {len(records)}")
    print(f"  With existing labels:     {with_labels}")
    print(f"  Without any labels (new): {len(records) - with_labels}")
    print(f"\nReady. Run:  python code/annotator/annotator.py --share")


if __name__ == '__main__':
    main()
