#!/usr/bin/env python3
"""Find good/bad demo examples for the June 12 meeting.

Scores every unique pair in the cleaned HC test split (data/vqa/test.jsonl)
with the CLIP Probe, selects exemplars per option (confident hits, misses,
false alarms, and an all-clear pair), then scores the shortlist with the
VLM so the demo cheat-sheet shows what BOTH panels will display.

Runs fully on CPU (login node safe). Output:
    output/demo_examples.json      all CLIP scores + shortlist with VLM scores
    ~/DEMO_EXAMPLES_JUNE12.txt     human-readable cheat sheet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = DATA_ROOT
IMAGES_DIR   = PROJECT_ROOT / "images"
VQA_TEST     = PROJECT_ROOT / "data" / "vqa" / "test.jsonl"
OUT_JSON     = PROJECT_ROOT / "output" / "demo_examples.json"
CHEAT_SHEET  = Path.home() / "DEMO_EXAMPLES_JUNE12.txt"

sys.path.insert(0, str(REPO_ROOT))
from evaluators import CLIPProbeEvaluator, VLMEvaluator  # noqa: E402
from evaluators.constants import ALL_OPTIONS, OPTION_DISPLAY  # noqa: E402

BAL_T = 0.50


def main():
    # ---- collect unique pairs with their HC labels -------------------------
    pairs: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for line in VQA_TEST.open():
        r = json.loads(line)
        pairs[(r["natural_image"], r["tactile_image"])][r["option_id"]] = int(r["label"])
    print(f"unique test pairs: {len(pairs)}", flush=True)

    # ---- CLIP pass over everything -----------------------------------------
    clip = CLIPProbeEvaluator()
    rows = []
    t0 = time.time()
    for i, ((nat, tac), labels) in enumerate(pairs.items()):
        probs = clip.score_pair(str(IMAGES_DIR / nat), str(IMAGES_DIR / tac))
        rows.append({"natural": nat, "tactile": tac,
                     "labels": labels, "clip": probs})
        if (i + 1) % 25 == 0:
            print(f"CLIP {i+1}/{len(pairs)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- select exemplars ---------------------------------------------------
    def judged(row, opt):
        return opt in row["labels"]

    shortlist: dict[str, dict] = {}   # key -> {row, why}

    def add(key, row, why):
        k = (row["natural"], row["tactile"])
        if key not in shortlist and not any(
            (v["row"]["natural"], v["row"]["tactile"]) == k
            for v in shortlist.values()
        ):
            shortlist[key] = {"row": row, "why": why}

    for opt in ALL_OPTIONS:
        pos = [r for r in rows if judged(r, opt) and r["labels"][opt] == 1]
        neg = [r for r in rows if judged(r, opt) and r["labels"][opt] == 0]

        if pos:
            best_tp = max(pos, key=lambda r: r["clip"][opt])
            if best_tp["clip"][opt] > BAL_T:
                add(f"good_{opt}", best_tp,
                    f"GOOD — CLIP correctly flags {OPTION_DISPLAY[opt]} "
                    f"(p={best_tp['clip'][opt]:.2f}, label=1)")
            worst_fn = min(pos, key=lambda r: r["clip"][opt])
            if worst_fn["clip"][opt] < BAL_T:
                add(f"miss_{opt}", worst_fn,
                    f"BAD — CLIP misses {OPTION_DISPLAY[opt]} "
                    f"(p={worst_fn['clip'][opt]:.2f}, label=1)")
        if neg:
            worst_fp = max(neg, key=lambda r: r["clip"][opt])
            if worst_fp["clip"][opt] > BAL_T:
                add(f"falsealarm_{opt}", worst_fp,
                    f"BAD — CLIP false alarm on {OPTION_DISPLAY[opt]} "
                    f"(p={worst_fp['clip'][opt]:.2f}, label=0)")

    # an all-clear pair: every judged option is 0 and CLIP agrees everywhere
    clear = [r for r in rows
             if r["labels"] and all(v == 0 for v in r["labels"].values())
             and all(p < BAL_T for p in r["clip"].values())]
    if clear:
        best_clear = min(clear, key=lambda r: max(r["clip"].values()))
        add("all_clear", best_clear,
            f"GOOD — all-clear pair (max CLIP p={max(best_clear['clip'].values()):.2f}, "
            f"all judged labels 0)")

    print(f"shortlist: {len(shortlist)} pairs", flush=True)

    # ---- VLM pass over the shortlist ----------------------------------------
    vlm = VLMEvaluator()
    for key, entry in shortlist.items():
        row = entry["row"]
        t0 = time.time()
        row["vlm"] = vlm.score_pair(str(IMAGES_DIR / row["natural"]),
                                    str(IMAGES_DIR / row["tactile"]))
        print(f"VLM {key}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- save ----------------------------------------------------------------
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"all_clip_scores": rows,
         "shortlist": {k: v for k, v in shortlist.items()}}, indent=2))
    print(f"saved -> {OUT_JSON}", flush=True)

    # ---- cheat sheet -----------------------------------------------------------
    lines = [
        "=" * 78,
        "  DEMO EXAMPLES — June 12 meeting (selected from cleaned HC test split)",
        "  Upload in Eval-only mode: natural -> left, tactile -> upload box",
        "  Image root: ~/vlm_finetune/images/",
        "=" * 78, "",
    ]
    order = ([k for k in shortlist if k.startswith("good_")]
             + ["all_clear"] * ("all_clear" in shortlist)
             + [k for k in shortlist if k.startswith(("miss_", "falsealarm_"))])
    for key in order:
        e = shortlist[key]
        r = e["row"]
        lines += [f"[{key}]  {e['why']}",
                  f"  natural: {r['natural']}",
                  f"  tactile: {r['tactile']}",
                  f"  labels (HC): {r['labels']}"]
        for model in ("clip", "vlm"):
            probs = r.get(model)
            if probs:
                lines.append("  " + model.upper().ljust(5) + " " + "  ".join(
                    f"{OPTION_DISPLAY[o]}={probs[o]:.2f}" for o in ALL_OPTIONS))
        lines.append("")
    CHEAT_SHEET.write_text("\n".join(lines))
    print(f"saved -> {CHEAT_SHEET}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
