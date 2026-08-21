"""Seed the un-annotated batch records with the source pair's existing labels.

The corruption prompt changes stroke width and nothing else, so for a corrupted
rendering the four options that are not `too_thick` should start from whatever a
human already decided about the SOURCE rendering. That leaves the reviewer one
real question per image -- is this too thick now? -- instead of five.

`too_thick` is deliberately left exactly as it is (the VLM pre-tick), because it
is the one option the corruption is expected to change and therefore the one the
human has to judge.

Records already annotated, and records excluded, are never touched: they carry a
human decision, and re-seeding them would silently overwrite it. The script
refuses to run without --apply and backs the file up before writing.

    python research/corruption/inherit_source_labels.py            # dry run
    python research/corruption/inherit_source_labels.py --apply
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT, require_data_root  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from evaluators.constants import ALL_OPTIONS  # noqa: E402




def source_labels(splits_dir: Path) -> dict:
    """tactile rel path -> {option: label} from the human-verified splits."""
    out = {}
    for split in ("train", "val", "test"):
        f = splits_dir / f"{split}.jsonl"
        if not f.exists():
            continue
        for line in f.open():
            r = json.loads(line)
            out.setdefault(r["tactile_image"], {})[r["option_id"]] = int(r["label"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default=str(REPO_ROOT / "verify_generated_batch"))
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--hold", default="too_thick", choices=ALL_OPTIONS,
                    help="the option the corruption targets. It is left exactly as it is "
                         "(the VLM pre-tick) for the human to judge; every OTHER option is "
                         "taken from the source pair's existing human label.")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    HELD = args.hold
    INHERIT = [o for o in ALL_OPTIONS if o != HELD]

    require_data_root()
    splits_dir = Path(args.splits_dir) if args.splits_dir else DATA_ROOT / "data" / "splits_v3_unified"
    ann_path = Path(args.batch) / "annotations.jsonl"
    records = [json.loads(l) for l in ann_path.open() if l.strip()]
    src = source_labels(splits_dir)
    print(f"{len(records)} records | {len(src)} source pairs in {splits_dir.name}")

    touched = skipped_done = skipped_excluded = no_source = 0
    changes = {o: 0 for o in INHERIT}
    examples = []
    for r in records:
        if r.get("excluded"):
            skipped_excluded += 1
            continue
        if r.get("annotated"):
            skipped_done += 1
            continue
        labels = src.get(r.get("source_tactile", ""))
        if not labels:
            no_source += 1
            continue
        before = {o: r.get(o, 0) for o in INHERIT}
        for o in INHERIT:
            v = int(labels.get(o, 0))
            if r.get(o, 0) != v:
                changes[o] += 1
            r[o] = v
        r["labels_inherited_from_source"] = True
        touched += 1
        if len(examples) < 5:
            examples.append((r["tactile_image"], before, {o: r[o] for o in INHERIT}, r[HELD]))

    print(f"\nleft untouched : {skipped_done} annotated, {skipped_excluded} excluded")
    print(f"re-seeded      : {touched}")
    print(f"no source labels (left as-is): {no_source}")
    print(f"\n`{HELD}` deliberately NOT changed on any record.")
    print("option values changed by inheriting the source labels:")
    for o in INHERIT:
        print(f"   {o:16} {changes[o]:4} of {touched}")
    print("\nexamples:")
    for name, b, a, tt in examples:
        print(f"   {name}")
        print(f"      was {b}")
        print(f"      now {a}   (too_thick left at {tt})")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    backup = ann_path.with_name(f"annotations.backup_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    shutil.copy2(ann_path, backup)
    tmp = ann_path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.replace(ann_path)
    print(f"\nbacked up to {backup.name}")
    print(f"wrote {ann_path}")


if __name__ == "__main__":
    main()
