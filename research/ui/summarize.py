#!/usr/bin/env python3
"""
Generate a written summary of precomputed results under both threshold modes.
Reads output/precomputed_results.json, writes output/results_summary.txt

Usage:
    python code/ui/summarize.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

from pathlib import Path
import json

PROJECT_ROOT   = DATA_ROOT
PRECOMPUTED    = PROJECT_ROOT / "output" / "precomputed_results.json"
OUTPUT_TXT     = PROJECT_ROOT / "output" / "results_summary.txt"

MODELS  = ["VLM Fine-tuned", "VLM Zero-shot", "CLIP Probe", "ResNet-50", "ViT-B/16"]
OPTIONS = ["too_thick", "broken_lines", "missing_parts", "missing_texture"]
BALANCED_THRESH = {opt: 0.5 for opt in OPTIONS}

data  = json.load(open(PRECOMPUTED))
pairs = list(data.items())
n     = len(pairs)

lines = []
W     = 88   # total expected pairs

def h(title):
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)

def compute_flags(result, balanced: bool):
    flagged = []
    for opt in OPTIONS:
        d      = result["options"][opt]
        thresh = BALANCED_THRESH[opt] if balanced else d["threshold"]
        if d["prob_yes"] > thresh:
            flagged.append(opt)
    return flagged


# ── Header ────────────────────────────────────────────────────────────────
lines.append("TactileEval — Precomputed Results Summary")
lines.append(f"Pairs in JSON: {n}   (expected: {W})")
lines.append(f"Models: {', '.join(MODELS)}")
lines.append("")
lines.append("Threshold modes:")
lines.append("  Calibrated = recall-optimised paper thresholds per model")
lines.append("    VLM models : too_thick=0.05, missing_texture=0.05, missing_parts=0.25, broken_lines=0.60")
lines.append("    CLIP Probe : too_thick=0.75, broken_lines=0.65, missing_parts=0.50, missing_texture=0.50")
lines.append("    ResNet/ViT : all 0.50 (no val calibration was done)")
lines.append("  Balanced    = 0.50 for all options / all models")


# ── Section 1: high-level counts ──────────────────────────────────────────
h("1. ALL-CLEAR COUNT  (pairs with zero defects flagged)")
lines.append(f"  {'Model':<22} {'Calibrated':>12} {'Balanced (0.5)':>15}")
lines.append(f"  {'-'*22} {'-'*12} {'-'*15}")
for model in MODELS:
    cal_clear = bal_clear = 0
    for _, entry in pairs:
        if model not in entry["models"]:
            continue
        result = entry["models"][model]
        if not compute_flags(result, balanced=False):
            cal_clear += 1
        if not compute_flags(result, balanced=True):
            bal_clear += 1
    lines.append(f"  {model:<22} {cal_clear:>5}/{n}        {bal_clear:>5}/{n}")


# ── Section 2: per-option flag rates ──────────────────────────────────────
for balanced, label in [(False, "Calibrated"), (True, "Balanced (0.5)")]:
    h(f"2{'b' if balanced else 'a'}. PER-OPTION FLAG COUNTS  [{label}]")
    header = f"  {'Model':<22}" + "".join(f"  {opt:>15}" for opt in OPTIONS)
    lines.append(header)
    lines.append("  " + "-" * (22 + 17 * len(OPTIONS)))
    for model in MODELS:
        counts = {opt: 0 for opt in OPTIONS}
        avail  = 0
        for _, entry in pairs:
            if model not in entry["models"]:
                continue
            avail += 1
            for opt in compute_flags(entry["models"][model], balanced):
                counts[opt] += 1
        row = f"  {model:<22}" + "".join(
            f"  {counts[opt]:>5}/{avail:<9}" for opt in OPTIONS)
        lines.append(row)


# ── Section 3: most-flagged pairs ─────────────────────────────────────────
h("3. TOP-10 MOST FLAGGED PAIRS  (total flags across all 5 models, calibrated)")
pair_scores = []
for label, entry in pairs:
    total = sum(
        len(compute_flags(entry["models"][m], balanced=False))
        for m in MODELS if m in entry["models"]
    )
    pair_scores.append((label, total))
pair_scores.sort(key=lambda x: -x[1])
for label, score in pair_scores[:10]:
    lines.append(f"  {score:>3} flags  {label}")


# ── Section 4: cleanest pairs ─────────────────────────────────────────────
h("4. CLEANEST PAIRS  (all 5 models agree all-clear, balanced thresholds)")
clean = []
for label, entry in pairs:
    if all(
        not compute_flags(entry["models"][m], balanced=True)
        for m in MODELS if m in entry["models"]
    ):
        clean.append(label)
if clean:
    for label in clean:
        lines.append(f"  {label}")
else:
    lines.append("  None — no pair is all-clear across all 5 models simultaneously.")


# ── Section 5: per-pair full breakdown ────────────────────────────────────
h("5. PER-PAIR FULL BREAKDOWN  (calibrated | balanced)")
for label, entry in pairs:
    lines.append(f"\n  {label}")
    for model in MODELS:
        if model not in entry["models"]:
            lines.append(f"    {model:<22}  [not computed]")
            continue
        result  = entry["models"][model]
        cal_fl  = compute_flags(result, balanced=False) or ["none"]
        bal_fl  = compute_flags(result, balanced=True)  or ["none"]
        t       = entry.get("timings", {}).get(model)
        tstr    = f"  {t:.2f}s" if t else ""
        lines.append(f"    {model:<22}  cal={','.join(cal_fl):<45}  bal={','.join(bal_fl)}{tstr}")


# ── Write output ───────────────────────────────────────────────────────────
OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
text = "\n".join(lines)
OUTPUT_TXT.write_text(text)
print(text)
print(f"\n\nWritten to {OUTPUT_TXT}")
