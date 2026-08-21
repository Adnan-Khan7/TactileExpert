"""Did the corruption actually take? Measure it, per image, before annotating.

gpt-image-1 fails silently: asked to damage an image it will sometimes hand back
something clean, or near-identical. Those are not positives, and finding out by
eye across 228 images is the slow way.

The measurement is the same one that discriminates real human-labelled positives
from negatives (`research/morphology/`): the BANA ratio of gap width to stroke
width. Across the 1,350-pair corpus, labelled `too_thick` positives sit at a
median ratio of 1.94 and clean renderings at 3.28.

A corruption "took" when the ratio moved down meaningfully AND landed in or below
the positive band. Both halves matter: an image that was already too thick before
corruption has not been improved by generating a second one.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "morphology"))
from _paths import DATA_ROOT, REPO_ROOT, require_data_root  # noqa: E402
from extract_morph_features import load_gray, morph_features  # noqa: E402

POS_MEDIAN = 1.94   # labelled too_thick positives, corpus-wide
NEG_MEDIAN = 3.28   # labelled negatives


def ratio(path: Path) -> float:
    f = morph_features(load_gray(path, 1024), 4, [0.05], 5.0)
    return float(f.get("gap_over_stroke_p50", float("nan")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default=str(REPO_ROOT / "verify_generated_batch"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="write per-image results as JSON")
    args = ap.parse_args()

    require_data_root()
    batch = Path(args.batch)
    rows = [json.loads(l) for l in (batch / "manifest.jsonl").open() if l.strip()]
    rows = [r for r in rows if (batch / "images" / r["output"]).exists()]
    if args.limit:
        rows = rows[: args.limit]

    out = []
    for i, r in enumerate(rows, 1):
        try:
            before = ratio(DATA_ROOT / "images" / r["tactile_image"])
            after = ratio(batch / "images" / r["output"])
        except Exception as exc:
            print(f"  [{i}] FAILED {r['output']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        out.append({"output": r["output"], "source": r["tactile_image"],
                    "before": before, "after": after,
                    # positive = ratio went UP = strokes got THINNER = corruption
                    # did the opposite of what was asked
                    "delta": after - before,
                    "in_positive_band": bool(after <= NEG_MEDIAN),
                    "already_positive": bool(before <= POS_MEDIAN),
                    "took": bool(after < before and after <= NEG_MEDIAN)})
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    b = np.array([o["before"] for o in out])
    a = np.array([o["after"] for o in out])
    took = sum(o["took"] for o in out)
    already = sum(o["already_positive"] for o in out)
    print(f"\n{len(out)} images")
    print(f"  gap/stroke BEFORE : median {np.nanmedian(b):.2f}   (corpus clean {NEG_MEDIAN})")
    print(f"  gap/stroke AFTER  : median {np.nanmedian(a):.2f}   (corpus positives {POS_MEDIAN})")
    print(f"  ratio decreased   : {int((a < b).sum())} / {len(out)}")
    print(f"  corruption took   : {took} / {len(out)}")
    print(f"  source was ALREADY at or below the positive median: {already}"
          f"  <- review these first, the corruption had little room to work")
    thinner = [o for o in out if o["delta"] > 0]
    under = [o for o in out if o["delta"] <= 0 and not o["took"]]
    print(f"  ratio INCREASED (strokes got thinner — the opposite of the ask): {len(thinner)}")
    print(f"  thickened but not into the positive band: {len(under)}")
    weak = sorted(thinner, key=lambda o: -o["delta"])[:15]
    if weak:
        print(f"\nworst {len(weak)} reversals — gpt-image-1 CLEANED these instead of "
              f"corrupting them; not positives, candidates for Exclude:")
        print(f"   {'gap/stroke':>18}   {'Δ':>6}   image")
        for o in weak:
            print(f"   {o['before']:6.2f} -> {o['after']:6.2f}  {o['delta']:+6.2f}   {o['output']}")
    if args.out:
        p = Path(args.out)
        p.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
