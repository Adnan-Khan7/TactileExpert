#!/usr/bin/env python3
"""
Rewrite hardcoded image paths in the VQA JSONL files to point to a new images directory.

Usage:
    python rewrite_paths.py --images-dir /path/to/family_datasets

Run this once after copying the images to the cluster, before training.
The script rewrites data/vqa/{train,val,test}.jsonl in place.
"""

import argparse
import json
from pathlib import Path

OLD_PREFIX = "/home/student/khan/smc_2026/family_datasets"

def rewrite(jsonl_path: Path, new_root: str):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            for msg in rec.get("messages", []):
                for part in msg.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "image":
                        img = part["image"]
                        if img.startswith("file://"):
                            raw = img[len("file://"):]
                            if raw.startswith(OLD_PREFIX):
                                rel = raw[len(OLD_PREFIX):].lstrip("/")
                                part["image"] = f"file://{new_root.rstrip('/')}/{rel}"
            records.append(rec)

    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"  {jsonl_path.name}: {len(records)} records rewritten")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True,
                        help="Absolute path to family_datasets on the cluster")
    args = parser.parse_args()

    new_root = str(Path(args.images_dir).resolve())
    vqa_dir = Path(__file__).resolve().parent / "data" / "vqa"

    print(f"Rewriting paths → {new_root}")
    for split in ("train", "val", "test"):
        rewrite(vqa_dir / f"{split}.jsonl", new_root)
    print("Done.")


if __name__ == "__main__":
    main()
