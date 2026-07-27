#!/usr/bin/env python3
"""
TactileEval — Inspection Folder Preparation  (generalized step5a)

Reads the scored spreadsheet from score_pipeline.py, copies flagged image
pairs into organized folders for visual review, and writes a comparison.txt
listing each pair with its scores and turker vote fraction.

Usage
-----
  # All options, conflict rows only (default):
  python code/review/prepare_inspection.py --split test

  # Specific option:
  python code/review/prepare_inspection.py --split test --option broken_lines

  # All rows for an option (not just conflicts):
  python code/review/prepare_inspection.py --split test --option too_thick --all-rows

  # From a specific scored file (xlsx or csv):
  python code/review/prepare_inspection.py --split test --scored-csv review/test/scored_test.xlsx

Output
------
  review/{split}/inspection/{option}/
    conflict/   — pairs where conflict_flag=True, sorted by model disagreement
    confirmed/  — pairs with no conflict (reference, optional with --include-confirmed)
    manifest.csv
    comparison.txt
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE         = Path(__file__).resolve().parent
CODE_DIR     = HERE.parent
PROJECT_ROOT = CODE_DIR.parent

IMAGES_DIR  = PROJECT_ROOT / "images"
REVIEW_DIR  = PROJECT_ROOT / "review"

sys.path.insert(0, str(CODE_DIR))
from models.constants import ALL_OPTIONS


def safe_filename(pair_id: str, suffix: str, ext: str = ".jpg") -> str:
    nat_path = pair_id.split("::")[0]
    parts    = Path(nat_path).parts
    try:
        class_name = parts[-3]
        stem       = Path(parts[-1]).stem
        base       = f"{class_name}_{stem}".replace(" ", "_").replace("(", "").replace(")", "")
    except IndexError:
        base = pair_id[-30:].replace("/", "_").replace("::", "__")
    return f"{base}{suffix}{ext}"


def copy_pair(nat_rel: str, tac_rel: str, dest_dir: Path, idx: int, pair_id: str) -> tuple[str, str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    nat_src = IMAGES_DIR / nat_rel
    tac_src = IMAGES_DIR / tac_rel

    nat_ext = Path(nat_rel).suffix or ".jpg"
    tac_ext = Path(tac_rel).suffix or ".jpg"
    nat_name = f"{idx:03d}_{safe_filename(pair_id, '_nat', nat_ext)}"
    tac_name = f"{idx:03d}_{safe_filename(pair_id, '_tac', tac_ext)}"

    for src, dst_name in [(nat_src, nat_name), (tac_src, tac_name)]:
        dst = dest_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  [warn] missing: {src}")

    return nat_name, tac_name


_SCORE_COL_LABELS = [
    ("clip_score",    "clip"),
    ("resnet_score",  "resnet"),
    ("vit_score",     "vit"),
    ("vlm_ft_score",  "vlm"),
]


def _fmt_score(row: dict, col: str) -> str:
    val = row.get(col)
    if val is None or val == "" or (isinstance(val, float) and pd.isna(val)):
        return "n/a "
    return f"{float(val):.3f}"


def write_comparison(rows: list[dict], opt: str, split: str, out_path: Path,
                     nat_names: list[str], tac_names: list[str]) -> None:
    # Determine which score columns are actually present
    present_cols = [(col, lbl) for col, lbl in _SCORE_COL_LABELS
                    if col in (rows[0] if rows else {})]
    scores_legend = " | ".join(lbl for _, lbl in present_cols) + "  — raw probabilities (0-1)"
    models_line   = ", ".join(lbl for _, lbl in present_cols)

    lines = [
        f"{opt.upper().replace('_', ' ')} — INSPECTION FILE",
        f"Split: {split}  |  Option: {opt}",
        f"Models: {models_line}",
        "",
        "LEGEND:",
        "  turker_vf : fraction of workers who flagged defect (1.0 = unanimous)",
        "  current   : label currently in dataset (0 or 1)",
        "  consensus : model majority vote (0, 1, or SPLIT)",
        "  conflict  : True where model consensus disagrees with turker label",
        "  default   : proposed label (fill user_label if you disagree)",
        "",
        f"  {scores_legend}",
        "",
        "=" * 80,
        "",
    ]

    for i, (row, nat_n, tac_n) in enumerate(zip(rows, nat_names, tac_names), 1):
        scores_str = "  ".join(f"{lbl}:{_fmt_score(row, col)}" for col, lbl in present_cols)
        consensus_str = row.get("model_consensus", "N/A")
        scores_line = f"       scores — {scores_str}" if present_cols else ""
        lines += [
            f"[{i:03d}]  conflict={str(row['conflict_flag']).upper():<5}  "
            f"turker_vf={row['turker_vf']}  current={row['current_label']}  "
            f"consensus={consensus_str}  default={row['default_label']}",
        ]
        if scores_line:
            lines.append(scores_line)
        lines += [
            f"       nat:  {nat_n}",
            f"       tac:  {tac_n}",
            f"       id:   {row['pair_id']}",
            "",
        ]

    out_path.write_text("\n".join(lines))


def process_option(df: pd.DataFrame, opt: str, split: str,
                   out_dir: Path, all_rows: bool, include_confirmed: bool) -> int:
    opt_df = df[df["option"] == opt].copy()
    if opt_df.empty:
        print(f"  [skip] no records for option '{opt}'")
        return 0

    conflict_df   = opt_df[opt_df["conflict_flag"] == True]
    confirmed_df  = opt_df[opt_df["conflict_flag"] == False] if include_confirmed else pd.DataFrame()

    target_df = opt_df if all_rows else conflict_df

    if target_df.empty:
        print(f"  [{opt}] no {'rows' if all_rows else 'conflicts'} to copy")
        return 0

    opt_dir       = out_dir / opt
    conflict_dir  = opt_dir / "conflict"
    confirmed_dir = opt_dir / "confirmed"

    # Clear stale images from any previous run so old indices don't linger
    for d in (conflict_dir, confirmed_dir):
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()

    manifest_rows = []
    nat_names, tac_names = [], []

    print(f"\n  [{opt}] {len(conflict_df)} conflicts, {len(confirmed_df)} confirmed")

    # Copy conflict images
    for idx, (_, row) in enumerate(conflict_df.iterrows(), 1):
        nat_n, tac_n = copy_pair(row["nat_path"], row["tac_path"], conflict_dir, idx, row["pair_id"])
        nat_names.append(nat_n)
        tac_names.append(tac_n)
        manifest_rows.append({**row.to_dict(), "folder": "conflict",
                               "copied_nat": nat_n, "copied_tac": tac_n})

    # Optionally copy confirmed images
    if include_confirmed and not confirmed_df.empty:
        for idx, (_, row) in enumerate(confirmed_df.iterrows(), 1):
            nat_n, tac_n = copy_pair(row["nat_path"], row["tac_path"],
                                     confirmed_dir, idx, row["pair_id"])
            manifest_rows.append({**row.to_dict(), "folder": "confirmed",
                                   "copied_nat": nat_n, "copied_tac": tac_n})

    # Write comparison.txt (conflict rows only)
    conflict_rows_list = conflict_df.to_dict("records")
    write_comparison(conflict_rows_list, opt, split,
                     opt_dir / "comparison.txt", nat_names, tac_names)

    # Write manifest CSV
    manifest_path = opt_dir / "manifest.csv"
    if manifest_rows:
        keys = list(manifest_rows[0].keys())
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(manifest_rows)
    print(f"  [{opt}] wrote {len(conflict_df)} conflict pairs → {conflict_dir}")
    print(f"  [{opt}] comparison.txt + manifest.csv → {opt_dir}")

    return len(conflict_df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split",    required=True, choices=["test", "val", "train"])
    parser.add_argument("--option",   default="all",
                        help="Option to inspect: all | too_thick | broken_lines | "
                             "missing_parts | missing_texture")
    parser.add_argument("--scored-csv",  default=None,
                        help="Path to scored file — .xlsx or .csv (default: review/{split}/scored_{split}.xlsx)")
    parser.add_argument("--all-rows",    action="store_true",
                        help="Copy all rows, not just conflicts")
    parser.add_argument("--include-confirmed", action="store_true",
                        help="Also copy non-conflict rows into confirmed/ subfolder")
    parser.add_argument("--out-dir",     default=None,
                        help="Output directory (default: review/{split}/inspection/)")
    args = parser.parse_args()

    # Load scored data — prefer xlsx, accept csv
    if args.scored_csv:
        scored_path = Path(args.scored_csv)
    else:
        xlsx = REVIEW_DIR / args.split / f"scored_{args.split}.xlsx"
        csv  = REVIEW_DIR / args.split / f"scored_{args.split}.csv"
        scored_path = xlsx if xlsx.exists() else csv

    if not scored_path.exists():
        raise FileNotFoundError(
            f"Scored file not found: {scored_path}\n"
            f"Run score_pipeline.py first:\n"
            f"  python code/review/score_pipeline.py --split {args.split}")

    df = pd.read_excel(scored_path) if scored_path.suffix == ".xlsx" else pd.read_csv(scored_path)
    # Restore bool dtype (CSV/Excel may store as string)
    df["conflict_flag"] = df["conflict_flag"].map(
        lambda x: x if isinstance(x, bool) else str(x).strip().lower() == "true")

    out_dir = Path(args.out_dir) if args.out_dir else REVIEW_DIR / args.split / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    options = ALL_OPTIONS if args.option == "all" else [args.option]
    total_copied = 0
    for opt in options:
        total_copied += process_option(df, opt, args.split, out_dir,
                                       args.all_rows, args.include_confirmed)

    print(f"\nDone. {total_copied} conflict pairs copied to {out_dir}")
    print(f"Open comparison.txt alongside the images to review.")
    print(f"Fill 'user_label' in review/{args.split}/scored_{args.split}.xlsx, then run:")
    print(f"  python code/review/apply_labels.py --split {args.split}")


if __name__ == "__main__":
    main()
