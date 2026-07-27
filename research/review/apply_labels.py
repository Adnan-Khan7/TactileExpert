#!/usr/bin/env python3
"""
TactileEval — Apply Reviewed Labels  (generalized step7b)

Reads the user_label column from a reviewed spreadsheet (output of
score_pipeline.py after manual review) and applies corrections to the
target split JSONL file.

Rules
-----
  - user_label = "1" or "0"  → override the record's label in the split file
  - user_label blank          → keep current label unchanged (default_label used)
  - Backs up the original split file before any modification

Usage
-----
  python code/review/apply_labels.py --split test
  python code/review/apply_labels.py --split test --dry-run
  python code/review/apply_labels.py --split val  --reviewed-csv review/val/scored_val.xlsx

Output
------
  TactileEval_Context_And_Data/data/splits/{split}.jsonl   (updated)
  TactileEval_Context_And_Data/data/splits/{split}_backup_{timestamp}.jsonl
  review/{split}/apply_labels_report.txt
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE         = Path(__file__).resolve().parent
CODE_DIR     = HERE.parent
PROJECT_ROOT = CODE_DIR.parent

SPLITS_DIR  = PROJECT_ROOT / "TactileEval_Context_And_Data" / "data" / "splits"
REVIEW_DIR  = PROJECT_ROOT / "review"

sys.path.insert(0, str(CODE_DIR))
from models.constants import ALL_OPTIONS


def load_reviewed(path: Path) -> pd.DataFrame:
    if path.suffix == ".xlsx":
        df = pd.read_excel(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .xlsx or .csv")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split",         required=True, choices=["test", "val", "train"])
    parser.add_argument("--reviewed-csv",  default=None,
                        help="Path to reviewed xlsx/csv (default: review/{split}/scored_{split}.xlsx)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print what would change without modifying any files")
    args = parser.parse_args()

    # Locate reviewed spreadsheet
    if args.reviewed_csv:
        reviewed_path = Path(args.reviewed_csv)
    else:
        reviewed_path = REVIEW_DIR / args.split / f"scored_{args.split}.xlsx"
    if not reviewed_path.exists():
        raise FileNotFoundError(f"Reviewed file not found: {reviewed_path}")

    df = load_reviewed(reviewed_path)

    # Filter to rows with a user_label filled in
    # Excel stores 0/1 as floats (0.0/1.0); coerce to int then back to string for comparison
    df["user_label"] = pd.to_numeric(df["user_label"], errors="coerce")
    labeled = df[df["user_label"].isin([0, 1, 0.0, 1.0])].copy()
    labeled["user_label"] = labeled["user_label"].astype(int)

    print(f"Reviewed file: {reviewed_path}")
    print(f"Total rows: {len(df)}  |  Rows with user_label: {len(labeled)}")

    if labeled.empty:
        print("\nNo user_label values found — nothing to apply.")
        return

    # Build lookup: (pair_id, option) → user_label
    overrides = {(row["pair_id"], row["option"]): int(row["user_label"])
                 for _, row in labeled.iterrows()}

    # Load split JSONL
    split_path = SPLITS_DIR / f"{args.split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}")

    records = [json.loads(l) for l in open(split_path) if l.strip()]

    # Apply overrides
    changed = []
    for rec in records:
        key = (rec["pair_id"], rec["option_id"])
        if key in overrides:
            new_label = overrides[key]
            old_label = rec["label"]
            if new_label != old_label:
                changed.append({
                    "pair_id":   rec["pair_id"],
                    "option":    rec["option_id"],
                    "old_label": old_label,
                    "new_label": new_label,
                })
                if not args.dry_run:
                    rec["label"]         = new_label
                    rec["label_fixed_by"] = f"manual_review_{args.split}_{datetime.now().strftime('%Y%m%d')}"

    print(f"\nLabel changes to apply: {len(changed)}")
    for ch in changed:
        print(f"  {ch['pair_id']}  [{ch['option']}]  {ch['old_label']} → {ch['new_label']}")

    if args.dry_run:
        print("\n[dry-run] No files modified.")
        return

    if not changed:
        print("All user_label values match current labels — no changes needed.")
        return

    # Backup original
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = SPLITS_DIR / f"{args.split}_backup_{ts}.jsonl"
    shutil.copy2(split_path, backup_path)
    print(f"\nBackup → {backup_path}")

    # Write updated split
    with open(split_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Updated → {split_path}")

    # Write report
    report_dir = REVIEW_DIR / args.split
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "apply_labels_report.txt"
    lines = [
        f"apply_labels.py — {args.split} split — {datetime.now().isoformat()}",
        f"Reviewed file: {reviewed_path}",
        f"Split file: {split_path}",
        f"Backup: {backup_path}",
        f"Changes applied: {len(changed)}",
        "",
        f"{'pair_id':<45} {'option':<20} {'old':>4} {'new':>4}",
        "-" * 80,
    ]
    for ch in changed:
        lines.append(f"  {ch['pair_id']:<43} {ch['option']:<20} {ch['old_label']:>4} → {ch['new_label']:>4}")
    lines += [
        "",
        "Next steps:",
        f"  1. Run score_pipeline.py on val/train if cleaning continues",
        f"  2. Once all splits cleaned: rebuild VQA dataset with step8_build_vqa_dataset.py",
        f"  3. Retrain evaluators",
    ]
    report_path.write_text("\n".join(lines))
    print(f"Report → {report_path}")

    # Per-option summary
    print("\nChanges by option:")
    for opt in ALL_OPTIONS:
        opt_changes = [c for c in changed if c["option"] == opt]
        if opt_changes:
            flips_to_pos = sum(1 for c in opt_changes if c["new_label"] == 1)
            flips_to_neg = sum(1 for c in opt_changes if c["new_label"] == 0)
            print(f"  {opt:<22} {len(opt_changes):>3} changes  "
                  f"(→1: {flips_to_pos}, →0: {flips_to_neg})")


if __name__ == "__main__":
    main()
