"""Did the corruption actually fire? Geometry check before spending review time.

Corruption prompts fail silently -- gpt-image-1 returns a clean image and nothing
errors. Worse, the too_thick batch showed it often *restyles* rather than
degrades (outlined giraffe spots came back filled solid, which reads as an
improvement), so "the file changed" is not evidence the defect is there.

The reference values are measured on the 1,350 human-labelled pairs, not guessed:

  too_thick     gap-to-stroke ratio. Positives sit at p50 1.94 / p10 0.63,
                negatives at 3.28 / 1.12.
                CAVEAT: p10 conflates dense texture with merged linework. A
                crisp drawing with tight spot-fills scores like a severe
                positive. Treat it as triage, never as a label.

  broken_lines  how far you must dilate before the breaks close. Positives need
                ~1.31 stroke widths, negatives 0.77; and dilating by two stroke
                widths merges positives down to 0.24 of their component count
                against 0.42 for negatives.

Usage:
    python research/corruption/check_corruption.py --batch <dir>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "morphology"))
from _paths import DATA_ROOT, require_data_root  # noqa: E402
from extract_morph_features import load_gray, morph_features  # noqa: E402

# (feature, real-positive median, real-negative median, direction that means "corrupted")
REFERENCE = {
    "too_thick": [
        ("gap_over_stroke_p10", 0.627, 1.123, "lower"),
        ("gap_over_stroke_p50", 1.937, 3.276, "lower"),
        ("frac_gaps_below_stroke", 0.314, 0.147, "higher"),
        ("ink_fraction", None, None, "higher"),
    ],
    "broken_lines": [
        ("cc_stabilise_sw", 1.308, 0.771, "higher"),
        ("cc_ratio_2.0sw", 0.244, 0.415, "lower"),
        ("cc_ratio_1.0sw", 0.342, 0.493, "lower"),
        ("components_per_len", None, None, "higher"),
        ("endpoints_per_len", None, None, "higher"),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    require_data_root()
    batch = Path(args.batch)
    rows = [json.loads(l) for l in (batch / "manifest.jsonl").open() if l.strip()]
    rows = [r for r in rows if (batch / "images" / r["output"]).exists()]
    if args.limit:
        rows = rows[: args.limit]
    kind = rows[0].get("corruption", "too_thick")
    ref = REFERENCE[kind]
    keys = [k for k, *_ in ref]
    print(f"{len(rows)} images, corruption = {kind}\n")

    before, after, names = [], [], []
    for r in rows:
        try:
            fb = morph_features(load_gray(DATA_ROOT / "images" / r["tactile_image"], 1024), 4, [0.05], 5.0)
            fa = morph_features(load_gray(batch / "images" / r["output"], 1024), 4, [0.05], 5.0)
        except Exception as exc:
            print(f"  skip {r['output']}: {type(exc).__name__}")
            continue
        before.append([fb[k] for k in keys])
        after.append([fa[k] for k in keys])
        names.append(r["output"])
    b, a = np.array(before), np.array(after)

    print(f"{'feature':26} {'before':>9} {'after':>9} | {'real pos':>9} {'real neg':>9}  moved?")
    for i, (k, pos, neg, direction) in enumerate(ref):
        mb, ma = np.nanmedian(b[:, i]), np.nanmedian(a[:, i])
        moved = (ma > mb) if direction == "higher" else (ma < mb)
        print(f"{k:26} {mb:9.3f} {ma:9.3f} | "
              f"{'        -' if pos is None else f'{pos:9.3f}'} "
              f"{'        -' if neg is None else f'{neg:9.3f}'}  {'yes' if moved else 'NO'} ({direction})")

    # per-image verdict on the single most discriminative feature
    k, pos, neg, direction = ref[0]
    ai, bi = a[:, 0], b[:, 0]
    fired = (ai > bi) if direction == "higher" else (ai < bi)
    reached = (ai >= pos) if direction == "higher" else (ai <= pos)
    print(f"\nper-image, on {k}:")
    print(f"  moved in the intended direction : {fired.sum()}/{len(ai)}")
    print(f"  reached the real-positive level : {reached.sum()}/{len(ai)}")
    idx = np.argsort(-ai if direction == "higher" else ai)
    print(f"\n  strongest {min(4, len(idx))}:")
    for i in idx[:4]:
        print(f"     {k}={ai[i]:.3f} (was {bi[i]:.3f})  {names[i]}")
    print(f"  weakest {min(4, len(idx))} — check these by eye first:")
    for i in idx[-4:]:
        print(f"     {k}={ai[i]:.3f} (was {bi[i]:.3f})  {names[i]}")


if __name__ == "__main__":
    main()
