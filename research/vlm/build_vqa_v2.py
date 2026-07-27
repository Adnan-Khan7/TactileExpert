#!/usr/bin/env python3
"""
Build VQA dataset for VLM fine-tuning from data/splits_v2 (6 options).

Key differences from step8_build_vqa_dataset.py:
  - Source: data/splits_v2/ (human-verified labels, no vote_fraction)
  - No HC filtering needed — all records already manually verified
  - Includes extra_parts and non_conformant (previously skipped)
  - Negative downsampling still applied (missing_texture is 83% positive)
  - noisy_background excluded (only 22 positives, insufficient)

Output: data/vqa_v2/{train,val,test}.jsonl + dataset_stats.json

Usage:
  python code/vlm/build_vqa_v2.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT       = DATA_ROOT
SRC_DIR    = ROOT / "data" / "splits_v2"
OUT_DIR    = ROOT / "data" / "vqa_v2"
IMAGE_ROOT = ROOT / "images"

ACTIVE_OPTIONS = [
    "too_thick", "broken_lines", "missing_parts", "missing_texture",
    "extra_parts", "non_conformant",
]

MAX_NEG_RATIO = 3   # cap negatives at 3x positives per option per split
RANDOM_SEED   = 42

# ── BANA-aligned question phrasing ────────────────────────────────────────────
QUESTIONS = {
    "too_thick": (
        "Do the outlines in this tactile graphic appear excessively thick, "
        "causing adjacent parts to merge together or fine structural details "
        "such as teeth, fingers, or narrow components to collapse into solid "
        "filled areas rather than distinct lines?"
    ),
    # v3 rewording: defines the concept as OUTLINE continuity, explicitly
    # excluding texture fills. Deploying a checkpoint trained on this text
    # requires the same text in TactileExpert/evaluators/constants.py QUESTIONS.
    "broken_lines": (
        "Consider only the structural outlines of the object, not any texture "
        "or fill patterns inside its regions. Are the outlines broken or "
        "discontinuous, so that a fingertip tracing an edge would encounter "
        "gaps, interruptions, or missing segments?"
    ),
    "missing_parts": (
        "Comparing this tactile graphic to the reference photograph, does the "
        "tactile drawing omit any clearly visible structural parts, anatomical "
        "features, or key functional components of the subject?"
    ),
    "missing_texture": (
        "Does this tactile graphic lack any texture differentiation entirely? "
        "That is, are all surface regions represented with the same uniform "
        "fill or no fill at all, making it impossible to distinguish different "
        "areas of the subject by touch?"
    ),
    "extra_parts": (
        "Comparing this tactile graphic to the reference photograph, does the "
        "tactile drawing contain invented structures, hallucinated appendages, "
        "or decorative additions that are not present in the reference?"
    ),
    "non_conformant": (
        "Comparing this tactile graphic to the reference photograph, does the "
        "tactile graphic fail to match the overall concept, subject type, or "
        "identity shown in the photograph — for example by depicting a "
        "fundamentally different object, species, or scene rather than a "
        "tactile rendering of the photographed subject?"
    ),
}

SYSTEM_PROMPT = (
    "You are a tactile graphics quality evaluator trained on BANA (Braille "
    "Authority of North America) standards. You will be shown a natural "
    "reference photograph and a tactile graphic derived from it. Answer the "
    "question with 'yes' or 'no' only."
)


def build_message(nat_rel: str, tac_rel: str, question: str, answer: str) -> list:
    # Use absolute paths so qwen_vl_utils.process_vision_info can open the files
    nat_abs = str(IMAGE_ROOT / nat_rel)
    tac_abs = str(IMAGE_ROOT / tac_rel)
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": [
            {"type": "image", "image": nat_abs},
            {"type": "image", "image": tac_abs},
            {"type": "text",  "text": question},
        ]},
        {"role": "assistant", "content": answer},
    ]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    all_stats = {}

    for split in ["train", "val", "test"]:
        src = SRC_DIR / f"{split}.jsonl"
        records_by_opt: dict[str, list] = defaultdict(list)

        with open(src) as f:
            for line in f:
                r = json.loads(line)
                if r["option_id"] in ACTIVE_OPTIONS:
                    records_by_opt[r["option_id"]].append(r)

        out_records = []
        opt_stats   = {}

        for opt in ACTIVE_OPTIONS:
            recs  = records_by_opt[opt]
            pos   = [r for r in recs if r["label"] == 1]
            neg   = [r for r in recs if r["label"] == 0]

            # Cap negatives to avoid always-no bias
            max_neg = len(pos) * MAX_NEG_RATIO
            if len(neg) > max_neg:
                neg = rng.sample(neg, max_neg)

            opt_stats[opt] = {"pos": len(pos), "neg": len(neg),
                              "total": len(pos) + len(neg)}

            for r in pos + neg:
                answer  = "yes" if r["label"] == 1 else "no"
                question = QUESTIONS[opt]
                nat_rel  = r["natural_image"]
                tac_rel  = r["tactile_image"]
                record_id = f"{split}_{opt}_{nat_rel.replace('/', '_')}"
                out_records.append({
                    "id":            record_id,
                    "split":         split,
                    "option_id":     opt,
                    "label":         r["label"],
                    "natural_image": nat_rel,
                    "tactile_image": tac_rel,
                    "category":      r.get("category", ""),
                    "object":        r.get("object", ""),
                    "question":      question,
                    "answer":        answer,
                    "messages":      build_message(nat_rel, tac_rel, question, answer),
                })

        rng.shuffle(out_records)

        out_path = OUT_DIR / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for r in out_records:
                f.write(json.dumps(r) + "\n")

        all_stats[split] = {"total": len(out_records), "per_option": opt_stats}

        print(f"{split}: {len(out_records)} records")
        for opt in ACTIVE_OPTIONS:
            s = opt_stats[opt]
            print(f"  {opt:<22}  pos={s['pos']:4d}  neg={s['neg']:4d}  "
                  f"total={s['total']:4d}")
        print()

    with open(OUT_DIR / "dataset_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"Wrote VQA dataset → {OUT_DIR}/")
    print("Next: sbatch code/vlm/train_qwen2vl_v2.sh")


if __name__ == "__main__":
    main()
