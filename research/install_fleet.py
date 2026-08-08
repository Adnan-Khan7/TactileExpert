#!/usr/bin/env python3
"""Install a trained fleet into TactileExpert/models/ — what the app and the
evaluation harness load.

DRY RUN BY DEFAULT. Nothing is written without --apply.

WHY THIS SCRIPT EXISTS
----------------------
`models/` was populated by hand. Nothing recorded which label state, which seed,
or which run each checkpoint came from, and nothing checked that the ten files
agreed with each other. A fleet assembled by `cp` can silently mix label states —
four judges from one run and a VLM from another — and every number measured
afterwards is then untraceable. `eval_full_v4.py` reads these paths directly, so
an inconsistent `models/` produces a clean-looking JSON that means nothing.

WHAT IT CHECKS BEFORE COPYING ANYTHING
--------------------------------------
Each vision checkpoint records `seed` and `splits_dir` inside the torch archive.
This script reads all ten, and refuses if:

  * a file is missing
  * any checkpoint's `splits_dir` differs from the others
  * any checkpoint's `seed` differs from the others, or from --seed if given
  * the VLM adapter is absent
  * ANY checkpoint predates the last label merge (see below)

THE STALE-LABEL CHECK IS THE ONE THAT MATTERS
--------------------------------------------
`splits_dir` CANNOT distinguish label states. `apply_verified_labels.py` rewrites
`data/splits_v3_unified/` in place, so a checkpoint trained at 407-verified and one
trained at 918-verified both record the identical path. Comparing that field
detects a mix of *directories*, not a mix of *label states*, and relying on it
would give false confidence.

What does work: every merge writes `label_change_report.json` into the splits
directory. A checkpoint whose mtime precedes that file was trained on labels that
have since changed. That is checked here and is a refusal, because it is the exact
failure this script exists to prevent.

`--label-state` is therefore an operator ASSERTION recorded in the manifest, not a
verified fact. The mtime check is what verifies it.

Usage:
  # four vision judges from a per-seed sweep dir, VLM from its own run
  python research/install_fleet.py \\
      --vision  TactileEval_Context_And_Data/models_tv918 \\
      --vlm     output_7b_tv918/best_checkpoint \\
      --label-state 918-verified --seed 42
  # then re-run with --apply
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from _paths import DATA_ROOT, REPO_ROOT, require_data_root  # noqa: E402

import argparse
import json
import shutil
from datetime import datetime

# Relative path inside the fleet -> which subdirectory of --vision holds it.
VISION_FILES = [
    ("resnet50/resnet50_evaluator.pt",  "resnet50"),
    ("vit_b16/vit_b_16_evaluator.pt",   "vit_b16"),
    ("clip_probe/ql_evaluator.pt",      "clip_probe"),
    ("clip_probe/qp_evaluator.pt",      "clip_probe"),
    ("clip_probe/qt_evaluator.pt",      "clip_probe"),
    ("clip_probe/qe_evaluator.pt",      "clip_probe"),
    ("dino_probe/ql_evaluator.pt",      "dino_probe"),
    ("dino_probe/qp_evaluator.pt",      "dino_probe"),
    ("dino_probe/qt_evaluator.pt",      "dino_probe"),
    ("dino_probe/qe_evaluator.pt",      "dino_probe"),
]


def resolve(p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else (DATA_ROOT / q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", required=True,
                    help="dir holding resnet50/ vit_b16/ clip_probe/ dino_probe/")
    ap.add_argument("--vlm", required=True, help="LoRA adapter dir to install")
    ap.add_argument("--label-state", required=True,
                    help="human-readable label state, e.g. 918-verified")
    ap.add_argument("--seed", type=int, default=None,
                    help="require every vision checkpoint to carry this seed")
    ap.add_argument("--dest", default=None, help="default: REPO_ROOT/models")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args()

    require_data_root()
    import torch

    vision = resolve(args.vision)
    vlm = resolve(args.vlm)
    dest = Path(args.dest).expanduser() if args.dest else REPO_ROOT / "models"

    print(f"vision source: {vision}")
    print(f"vlm source:    {vlm}")
    print(f"destination:   {dest}\n")

    # ── when did the labels last change? ─────────────────────────────────────
    # Anything trained before this timestamp was fitted to labels that have since
    # been corrected. See the module docstring: this is the only automatic check
    # that can catch a stale-label fleet, because splits_dir cannot.
    merge_marker = DATA_ROOT / "data" / "splits_v3_unified" / "label_change_report.json"
    merge_mtime = merge_marker.stat().st_mtime if merge_marker.exists() else None
    if merge_mtime:
        print(f"last label merge: {datetime.fromtimestamp(merge_mtime):%Y-%m-%d %H:%M:%S}"
              f"  ({merge_marker.name})\n")
    else:
        print(f"WARNING: no {merge_marker.name} found — cannot check for stale labels\n")

    # ── gather and validate ──────────────────────────────────────────────────
    problems, plan, provenance = [], [], []
    for rel, sub in VISION_FILES:
        src = vision / rel
        if not src.exists():                     # also accept a flat layout
            alt = vision / Path(rel).name
            src = alt if alt.exists() else src
        if not src.exists():
            problems.append(f"missing: {src}")
            continue
        d = torch.load(src, map_location="cpu", weights_only=False)
        mtime = src.stat().st_mtime
        stale = bool(merge_mtime and mtime < merge_mtime)
        provenance.append({
            "file": rel,
            "seed": d.get("seed"),
            "splits_dir": str(d.get("splits_dir", "")),
            "independent_aug": d.get("independent_aug"),
            "best_epoch": d.get("best_epoch"),
            "best_val_f1": d.get("best_val_f1"),
            "trained_at": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            "predates_last_label_merge": stale,
        })
        if stale:
            problems.append(f"STALE LABELS: {rel} was written "
                            f"{datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M} — before "
                            f"the last merge")
        plan.append((src, dest / rel))

    adapter = vlm / "adapter_model.safetensors"
    if not adapter.exists():
        problems.append(f"missing VLM adapter: {adapter}")
    elif merge_mtime and adapter.stat().st_mtime < merge_mtime:
        problems.append(f"STALE LABELS: VLM adapter written "
                        f"{datetime.fromtimestamp(adapter.stat().st_mtime):%Y-%m-%d %H:%M}"
                        f" — before the last merge")

    print(f"{'file':34}{'seed':>6}  {'trained':<17}  splits_dir")
    for p in provenance:
        flag = "  <- STALE" if p["predates_last_label_merge"] else ""
        print(f"{p['file']:34}{str(p['seed']):>6}  {p['trained_at']:<17}  "
              f"{Path(p['splits_dir']).name}{flag}")
    print()

    seeds = {p["seed"] for p in provenance}
    splits = {p["splits_dir"] for p in provenance}
    if len(splits) > 1:
        problems.append("checkpoints disagree on splits_dir: " + ", ".join(sorted(splits)))
    if len(seeds) > 1:
        problems.append(f"MIXED SEEDS — checkpoints disagree: {sorted(seeds)}")
    if args.seed is not None and seeds and seeds != {args.seed}:
        problems.append(f"seed mismatch: checkpoints carry {sorted(seeds)}, "
                        f"--seed says {args.seed}")

    if problems:
        print("REFUSING TO INSTALL:")
        for x in problems:
            print(f"  - {x}")
        raise SystemExit("fix the above; a mixed fleet makes every later number "
                         "untraceable")

    print(f"COHERENT: {len(plan)} vision files, seed {seeds.pop()}, "
          f"splits {Path(splits.pop()).name}")
    print(f"          VLM adapter from {vlm.parent.name}/{vlm.name}\n")

    existing = dest.exists() and any(dest.rglob("*"))
    print(f"destination currently {'POPULATED — will be backed up' if existing else 'empty'}")
    for src, dst in plan:
        print(f"  {src.relative_to(vision) if vision in src.parents else src.name} -> {dst.relative_to(dest)}")
    print(f"  {vlm} -> vlm_checkpoint/")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    # ── write ────────────────────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if existing:
        backup = dest.parent / f"{dest.name}_backup_{stamp}"
        shutil.copytree(dest, backup)
        print(f"\nbackup -> {backup}")

    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    vdst = dest / "vlm_checkpoint"
    # Replace rather than merge: a leftover file from the previous adapter (a
    # stale test_metrics.json, say) would misreport what is deployed.
    if vdst.exists():
        shutil.rmtree(vdst)
    shutil.copytree(vlm, vdst)

    manifest = {
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "label_state": args.label_state,
        "vision_source": str(vision),
        "vlm_source": str(vlm),
        "vision_provenance": provenance,
    }
    (dest / "FLEET_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"installed -> {dest}/  (+ FLEET_MANIFEST.json)")
    print(f"label state: {args.label_state}")
    print("\nNEXT: score and gate.")
    print("  sbatch research/experiments/run_eval_full.sh --splits test \\")
    print("      --base-model Qwen/Qwen2-VL-7B-Instruct --out experiments_out/<name>.json")


if __name__ == "__main__":
    main()
