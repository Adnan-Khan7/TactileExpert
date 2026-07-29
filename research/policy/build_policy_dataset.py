#!/usr/bin/env python3
"""Build the edit-policy training sets (Ch5 Loop 3) from the extracted BC/DPO records.

Turns experiments_out/policy_data/{bc.jsonl,dpo.jsonl} into Qwen2-VL chat-format
records, the same "bake the messages into the jsonl" convention build_vqa_v2.py
uses, so the trainers stay thin.

  SFT (behavior cloning)  state -> the human's edit instruction
  DPO (preference)        state -> chosen = human, rejected = the prefill shown

State is rendered as: [natural photo, current tactile graphic] + a text block
holding the panel's per-option defect probabilities, the target defect, and the
edits already applied in this session.

SPLITTING — the one thing to get right here. 157 BC records come from only 82
trajectories and 57 DPO pairs from 36, and the DPO trajectories are a SUBSET of
the BC ones. Splitting by row, or splitting BC and DPO independently, puts the
same editing session on both sides of the line. So trajectory_id is assigned to
a split ONCE and both datasets inherit that assignment.

Known data shape (measured 2026-07-29 — see SESSION_NOTES_JULY29.md):
  * 55 of 157 BC records are era="pre-ensemble"; all 57 DPO pairs are
    "ensemble". --era filters this.
  * 82 of 157 BC records have empty history.
  * target_option carries a 6th value "multiple" (2 records) that is NOT in
    ALL_OPTIONS. Kept as a literal string; nothing here one-hots it.

Usage:
  python research/policy/build_policy_dataset.py
  python research/policy/build_policy_dataset.py --era ensemble --seed 42
Output: experiments_out/policy_data/{sft,dpo_pairs}/{train,val,test}.jsonl
        experiments_out/policy_data/build_stats.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

POLICY_DIR = DATA_ROOT / "experiments_out/policy_data"
BC_FILE    = POLICY_DIR / "bc.jsonl"
DPO_FILE   = POLICY_DIR / "dpo.jsonl"
IMAGE_ROOT = REPO_ROOT / "generated_training_data"

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]

OPTION_DISPLAY = {
    "too_thick":       "too thick",
    "broken_lines":    "broken lines",
    "missing_parts":   "missing parts",
    "missing_texture": "missing texture",
    "extra_parts":     "extra parts",
    "multiple":        "multiple defects",
}

# The policy writes instructions, it does not answer yes/no — so this is NOT the
# evaluator SYSTEM_PROMPT from evaluators/constants.py. Deliberately separate:
# reusing the judge's prompt here would train the policy to speak as a judge.
SYSTEM_PROMPT = (
    "You are an expert at repairing tactile graphics for blind and visually "
    "impaired readers. Tactile graphics are embossed line drawings explored by "
    "touch. You are shown a natural reference photograph and the current tactile "
    "graphic derived from it, together with an automated panel's defect "
    "assessment. Write ONE concise, imperative edit instruction that repairs the "
    "target defect. Describe only what to change in the tactile graphic. Never "
    "mention the reference photograph in the instruction."
)


def panel_summary(scores_raw: dict) -> str:
    """Mean per-option defect probability across the models present.

    Mirrors defect_mass() in extract_policy_data.py, but kept per-option rather
    than averaged to a scalar so the policy can see WHICH defect dominates.
    """
    lines = []
    for opt in OPTS:
        vals = [per_opt[opt] for per_opt in scores_raw.values() if opt in per_opt]
        if vals:
            lines.append(f"  {OPTION_DISPLAY[opt]}: {sum(vals) / len(vals):.2f}")
    return "\n".join(lines)


def render_state(rec: dict) -> str:
    parts = [
        f"Object: {rec.get('object') or 'unknown'}",
        "",
        "Panel defect assessment (0 = absent, 1 = certainly present):",
        panel_summary(rec.get("scores_raw") or {}),
        "",
        f"Target defect to repair: {OPTION_DISPLAY.get(rec.get('target_option', ''), rec.get('target_option') or 'unspecified')}",
    ]
    history = rec.get("history") or []
    if history:
        parts += ["", "Edits already applied in this session:"]
        parts += [f"  {i + 1}. {h}" for i, h in enumerate(history)]
    parts += ["", "Write the next edit instruction."]
    return "\n".join(parts)


def build_messages(rec: dict, answer: str | None) -> list:
    """Qwen2-VL chat messages. Absolute image paths so process_vision_info can open them."""
    nat_abs = str(IMAGE_ROOT / rec["nat_image"])
    tac_abs = str(IMAGE_ROOT / rec["tac_image"])
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": nat_abs},
            {"type": "image", "image": tac_abs},
            {"type": "text",  "text": render_state(rec)},
        ]},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def assign_splits(traj_ids: list[str], seed: int,
                  ratios=(0.70, 0.15, 0.15)) -> dict[str, str]:
    """Deterministic trajectory-level split, shared by SFT and DPO.

    Sorted before shuffling so the assignment does not depend on file order.
    """
    ids = sorted(traj_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    split_of = {}
    for i, tid in enumerate(ids):
        if i < n_train:
            split_of[tid] = "train"
        elif i < n_train + n_val:
            split_of[tid] = "val"
        else:
            split_of[tid] = "test"
    return split_of


def missing_images(recs: list[dict]) -> list[str]:
    missing = []
    for r in recs:
        for key in ("nat_image", "tac_image"):
            p = IMAGE_ROOT / r[key]
            if not p.exists():
                missing.append(str(p))
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the trajectory-level split")
    ap.add_argument("--era", choices=["all", "ensemble", "pre-ensemble"], default="all",
                    help="Filter BC records by collection era. All DPO pairs are "
                         "ensemble-era, so this only affects SFT.")
    ap.add_argument("--min-reward", type=float, default=None,
                    help="Keep only SFT records with calibrated reward above this. "
                         "Omit to keep all (recommended: filtering is a "
                         "training-time choice, not a data-build one).")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else POLICY_DIR
    for missing_input in (p for p in (BC_FILE, DPO_FILE) if not p.exists()):
        raise SystemExit(
            f"missing {missing_input} — run research/policy/extract_policy_data.py first"
        )

    bc = [json.loads(l) for l in BC_FILE.read_text().splitlines() if l.strip()]
    dpo = [json.loads(l) for l in DPO_FILE.read_text().splitlines() if l.strip()]

    # Fail loudly on absent images: a silently-skipped pair would quietly shrink
    # an already-thin set and the counts would still look plausible.
    absent = missing_images(bc) + missing_images(dpo)
    if absent:
        raise SystemExit(
            f"{len(absent)} referenced image(s) not found under {IMAGE_ROOT}, e.g.\n  "
            + "\n  ".join(absent[:5])
        )

    bc_kept = [r for r in bc if args.era == "all" or r.get("era") == args.era]
    if args.min_reward is not None:
        bc_kept = [r for r in bc_kept
                   if r.get("reward_calibrated") is not None
                   and r["reward_calibrated"] > args.min_reward]

    # One assignment over the UNION, so a trajectory contributing to both
    # datasets lands in the same split in both.
    all_trajs = {r["trajectory_id"] for r in bc} | {r["trajectory_id"] for r in dpo}
    split_of = assign_splits(list(all_trajs), args.seed)

    sft = defaultdict(list)
    for r in bc_kept:
        sft[split_of[r["trajectory_id"]]].append({
            "trajectory_id":     r["trajectory_id"],
            "object":            r.get("object", ""),
            "era":               r.get("era", ""),
            "target_option":     r.get("target_option", ""),
            "n_history":         len(r.get("history") or []),
            "reward_raw":        r.get("reward_raw"),
            "reward_calibrated": r.get("reward_calibrated"),
            "helped":            r.get("helped"),
            "instruction":       r["instruction"],
            "messages":          build_messages(r, r["instruction"]),
        })

    pairs = defaultdict(list)
    for r in dpo:
        pairs[split_of[r["trajectory_id"]]].append({
            "trajectory_id":     r["trajectory_id"],
            "object":            r.get("object", ""),
            "era":               r.get("era", ""),
            "target_option":     r.get("target_option", ""),
            "n_history":         len(r.get("history") or []),
            "reward_calibrated": r.get("reward_calibrated"),
            "prefill_trusted":   r.get("prefill_trusted", ""),
            "chosen":            r["chosen"],
            "rejected":          r["rejected"],
            # Prompt only — the trainer appends chosen/rejected itself.
            "messages":          build_messages(r, None),
        })

    stats = {
        "seed": args.seed,
        "era_filter": args.era,
        "min_reward": args.min_reward,
        "trajectories_total": len(all_trajs),
        "trajectories_per_split": dict(Counter(split_of.values())),
        "sft": {},
        "dpo": {},
    }

    for name, buckets in (("sft", sft), ("dpo_pairs", pairs)):
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            rows = buckets.get(split, [])
            with open(d / f"{split}.jsonl", "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            key = "sft" if name == "sft" else "dpo"
            entry = {
                "n": len(rows),
                "trajectories": len({r["trajectory_id"] for r in rows}),
                "target_option": dict(Counter(r["target_option"] for r in rows)),
                "era": dict(Counter(r["era"] for r in rows)),
            }
            if key == "sft":
                entry["helped"] = sum(1 for r in rows if r.get("helped"))
            else:
                entry["negative_reward"] = sum(
                    1 for r in rows
                    if r.get("reward_calibrated") is not None
                    and r["reward_calibrated"] < 0)
            stats[key][split] = entry

    # Leakage assertion — cheap, and the failure it guards is silent.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for key, buckets in (("sft", sft), ("dpo", pairs)):
            ta = {r["trajectory_id"] for r in buckets.get(a, [])}
            tb = {r["trajectory_id"] for r in buckets.get(b, [])}
            if ta & tb:
                raise SystemExit(f"LEAK: {key} {a}/{b} share trajectories {ta & tb}")
    sft_trajs = {r["trajectory_id"] for rows in sft.values() for r in rows}
    for split, rows in pairs.items():
        for r in rows:
            if r["trajectory_id"] in sft_trajs and split_of[r["trajectory_id"]] != split:
                raise SystemExit("LEAK: a trajectory landed in different SFT and DPO splits")

    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nsaved -> {out_dir}/sft/ and {out_dir}/dpo_pairs/")


if __name__ == "__main__":
    main()
