"""Fold the reviewed corruption batches into a single released dataset.

Produces, under a user-supplied dataset root:

    images/corrupted_v1/…            the reviewed corrupted renderings
    <splits_out>/dataset.jsonl       every row, with a `split` field — the release form
    <splits_out>/{train,val,test}.jsonl   the same rows split out, for the trainers

Every path written is RELATIVE to the images root, never absolute, so the corpus
can be released and re-rooted anywhere. The root comes from --dataset-root, or
$TACTILE_DATA_ROOT/images by default.

**A corrupted pair inherits its SOURCE pair's split. This is not a convenience.**
corrupt(X) is the same drawing as X with one property altered. Putting X in train
and corrupt(X) in test would let a judge score on a near-duplicate of something it
trained on, and every headline number would be inflated with no way to tell from
the outside. Inheriting the split keeps each drawing wholly inside one split.

Never writes over splits_v3_unified: one directory per dataset state, so the
provenance of a reported number survives.

    python research/corruption/compile_dataset.py                 # dry run
    python research/corruption/compile_dataset.py --apply
"""

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT, require_data_root  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from evaluators.constants import ALL_OPTIONS  # noqa: E402

BATCHES = [
    ("too_thick", REPO_ROOT / "verify_generated_batch"),
    ("broken_lines", REPO_ROOT / "verify_generated_batch_broken_lines"),
]
CORRUPT_SUBDIR = "corrupted_v1"


def load_source_splits(splits_dir: Path):
    """tactile rel path -> (split, natural rel path), from the existing corpus."""
    where, nat = {}, {}
    for split in ("train", "val", "test"):
        for line in (splits_dir / f"{split}.jsonl").open():
            r = json.loads(line)
            where[r["tactile_image"]] = split
            nat[r["tactile_image"]] = r["natural_image"]
    return where, nat


def existing_rows(splits_dir: Path):
    rows = []
    for split in ("train", "val", "test"):
        for line in (splits_dir / f"{split}.jsonl").open():
            r = json.loads(line)
            r["split"] = split
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=None,
                    help="images root the paths resolve against (default $TACTILE_DATA_ROOT/images)")
    ap.add_argument("--splits-in", default=None, help="default: $TACTILE_DATA_ROOT/data/splits_v3_unified")
    ap.add_argument("--splits-out", default=None, help="default: $TACTILE_DATA_ROOT/data/splits_v4_corrupted")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    require_data_root()
    images_root = Path(args.dataset_root) if args.dataset_root else DATA_ROOT / "images"
    splits_in = Path(args.splits_in) if args.splits_in else DATA_ROOT / "data" / "splits_v3_unified"
    splits_out = Path(args.splits_out) if args.splits_out else DATA_ROOT / "data" / "splits_v4_corrupted"
    if splits_out.exists() and args.apply:
        raise SystemExit(f"{splits_out} exists — one directory per dataset state; pick another --splits-out")

    where, natural_of = load_source_splits(splits_in)
    base = existing_rows(splits_in)
    base_pairs = {r["pair_id"] for r in base}
    print(f"existing corpus: {len(base_pairs)} pairs / {len(base)} rows from {splits_in.name}")

    new_rows, copies, per_split, per_corruption, skipped = [], [], collections.Counter(), collections.Counter(), []
    for corruption, batch in BATCHES:
        ann = batch / "annotations.jsonl"
        if not ann.exists():
            print(f"  ! {batch.name}: no annotations.jsonl, skipped")
            continue
        recs = [json.loads(l) for l in ann.open() if l.strip()]
        keep = [r for r in recs if r.get("annotated") and not r.get("excluded")]
        print(f"  {batch.name}: {len(recs)} reviewed -> {len(keep)} kept")
        for r in keep:
            src = r["source_tactile"]
            if src not in where:
                skipped.append((r["tactile_image"], "source not in the corpus splits"))
                continue
            split = where[src]                      # ← leakage guard: inherit the source's split
            rel = f"{CORRUPT_SUBDIR}/{r['tactile_image']}"
            nat = r.get("natural_image") or natural_of[src]
            pid = f"{nat}::{rel}"
            if pid in base_pairs:
                skipped.append((rel, "pair_id already in the corpus"))
                continue
            copies.append((batch / "images" / r["tactile_image"], images_root / rel))
            per_split[split] += 1
            per_corruption[corruption] += 1
            for opt in ALL_OPTIONS:
                new_rows.append({
                    "pair_id": pid,
                    "natural_image": nat,
                    "tactile_image": rel,
                    "category": r["category"],
                    "object": r["object"],
                    "option_id": opt,
                    "label": int(r.get(opt, 0)),
                    "split": split,
                    "label_verified": True,
                    "verified_at": r.get("annotated_at"),
                    "verification_kind": "corruption_review_v1",
                    "synthetic_corruption": True,
                    "corruption": corruption,
                    "source_tactile": src,
                })

    print(f"\nnew pairs: {sum(per_split.values())}  ({dict(per_corruption)})")
    print(f"split inherited from source: {dict(per_split)}")
    if skipped:
        print(f"skipped {len(skipped)}: {collections.Counter(s for _, s in skipped)}")

    all_rows = base + new_rows
    pairs = {r["pair_id"] for r in all_rows}
    print(f"\ncombined: {len(pairs)} pairs / {len(all_rows)} rows")
    print(f"{'split':6} {'pairs':>6} {'rows':>7}   " + "  ".join(f"{o:>16}" for o in ALL_OPTIONS))
    for split in ("train", "val", "test"):
        rs = [r for r in all_rows if r["split"] == split]
        npair = len({r["pair_id"] for r in rs})
        pos = {o: sum(r["label"] for r in rs if r["option_id"] == o) for o in ALL_OPTIONS}
        print(f"{split:6} {npair:6} {len(rs):7}   " + "  ".join(f"{pos[o]:16}" for o in ALL_OPTIONS))

    # a drawing must live in exactly one split, source and corruption alike
    fam = collections.defaultdict(set)
    for r in all_rows:
        key = r.get("source_tactile") or r["tactile_image"]
        fam[key].add(r["split"])
    bleed = {k: v for k, v in fam.items() if len(v) > 1}
    print(f"\nleakage check — a drawing appearing in more than one split: {len(bleed)}")
    for k, v in list(bleed.items())[:5]:
        print(f"   {k}: {sorted(v)}")

    if not args.apply:
        print("\nDRY RUN — nothing copied, nothing written. Re-run with --apply.")
        return
    if bleed:
        raise SystemExit("refusing to write: a drawing spans multiple splits")

    dest = images_root / CORRUPT_SUBDIR
    dest.mkdir(parents=True, exist_ok=True)
    for src, dst in copies:
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"\ncopied {len(copies)} images -> {dest}")

    splits_out.mkdir(parents=True, exist_ok=True)
    with (splits_out / "dataset.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    for split in ("train", "val", "test"):
        with (splits_out / f"{split}.jsonl").open("w") as f:
            for r in all_rows:
                if r["split"] == split:
                    f.write(json.dumps(r) + "\n")
    (splits_out / "README.md").write_text(
        "# TactileEval — dataset state v4 (corruption-augmented)\n\n"
        f"{len(pairs)} pairs / {len(all_rows)} decisions. Every path is relative to the\n"
        "images root; set it with `--dataset-root` or `$TACTILE_DATA_ROOT/images`.\n\n"
        "`dataset.jsonl` carries every row with a `split` field and is the release form.\n"
        "`train/val/test.jsonl` are the same rows split out for the trainers.\n\n"
        f"Adds {sum(per_split.values())} reviewed synthetic corruptions "
        f"({dict(per_corruption)}) under `images/{CORRUPT_SUBDIR}/`. Each corrupted pair\n"
        "inherits its SOURCE pair's split, so a drawing never spans two splits.\n"
        "Corrupted rows carry `synthetic_corruption`, `corruption` and `source_tactile`.\n"
    )
    print(f"wrote {splits_out}/dataset.jsonl + train/val/test.jsonl + README.md")


if __name__ == "__main__":
    main()
