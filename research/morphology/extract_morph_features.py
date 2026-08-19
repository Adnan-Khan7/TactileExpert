"""Morphological / topological features of a tactile rendering.

README §9 item 1. The two failing options are geometric properties of the
rendering alone -- neither ANNOTATION_CRITERIA entry references the photograph
-- so measure them geometrically instead of learning them from a global
embedding:

    too_thick     BANA asks whether adjacent components merge, with a minimum
                  1/8 inch separation. The distance transform of the ink mask
                  IS half the stroke width at every pixel; the distance
                  transform of the background is half the gap width. The
                  criterion is the RATIO of the two, which is dimensionless --
                  the only form that survives a corpus running 200x200 to
                  3000x3213.

    broken_lines  Skeletonize, count endpoints and connected components. A
                  continuous contour has few endpoints per unit length; a
                  broken one has many. Normalising by skeleton length is not
                  cosmetic: a raw endpoint count is a drawing-complexity
                  feature wearing a topology costume.

Scale discipline. Every feature emitted for the classifier is either a ratio of
two lengths or a count per unit skeleton length. Absolute pixel quantities are
emitted only under the `qc__` prefix, which exists to be inspected and to be
held out of the model -- image dimensions alone separate the generated corpus
from the scraped one, and provenance predicts labels (see README 4i).

The free parameter is `--outline-min-frac`: the skeleton-branch length below
which a component is treated as texture fill rather than outline. The
`broken_lines` criterion says to ignore texture fills, and stipple fills
otherwise manufacture thousands of endpoints. It MUST be chosen on train/val
and reported -- never tuned against test.

Usage:
    # smoke test on a handful of pairs, with per-image diagnostics
    python research/morphology/extract_morph_features.py --split val --limit 12 --verbose

    # full extraction, cached by pair_id
    python research/morphology/extract_morph_features.py --split all \
        --out experiments_out/morph_features_tv974.json
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, convolve, distance_transform_edt, label as cc_label

from _binarize import binarize, load_gray, otsu  # noqa: F401  (one shared definition)
from skimage.morphology import medial_axis, skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, require_data_root  # noqa: E402

NEIGHBOURS = np.ones((3, 3), dtype=np.uint8)
NEIGHBOURS[1, 1] = 0

# Foreground is 8-connected, background 4-connected -- the standard digital
# topology pairing. scipy's default is 4-connectivity for BOTH, which shatters
# every diagonal skeleton run into isolated pixels and inflates the component
# and endpoint counts by an order of magnitude.
CONN8 = np.ones((3, 3), dtype=int)


# ── helpers ─────────────────────────────────────────────────────────────────

def neighbour_count(skel: np.ndarray) -> np.ndarray:
    return convolve(skel.astype(np.uint8), NEIGHBOURS, mode="constant", cval=0)


def quantiles(values: np.ndarray, qs=(10, 50, 90)) -> dict:
    if values.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    out = np.percentile(values, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, out)}


def _safe(a, b):
    return float(a / b) if b else float("nan")


# ── feature extraction ──────────────────────────────────────────────────────

def morph_features(gray: np.ndarray, speckle_px: int, outline_fracs: list,
                   gap_band_mult: float) -> dict:
    h, w = gray.shape
    diag = float(np.hypot(h, w))
    mask, thr, inverted, ambiguity, speckle, ink_raw = binarize(gray, speckle_px)

    f = {
        "qc__height": float(h), "qc__width": float(w), "qc__diag": diag,
        "qc__otsu": float(thr), "qc__inverted": float(inverted),
        "qc__ambiguity_frac": ambiguity, "qc__speckle_components": float(speckle),
        # qc__ prefix = held out of the model. This one is the comparator, and
        # feeding the thing you are measured against to the model would be
        # circular even though it is only one column.
        "qc__ink_fraction_raw": ink_raw,
        "ink_fraction": float(mask.mean()),
    }
    if not mask.any() or mask.all():
        return f  # degenerate rendering; caller sees the NaNs

    # ── stroke width: the medial axis carries the distance, so half-width is
    #    read where the stroke actually is rather than averaged over its bulk
    axis, dist = medial_axis(mask, return_distance=True)
    stroke_w = 2.0 * dist[axis]
    sw = quantiles(stroke_w)
    med_sw = sw["p50"]
    f["stroke_w_p10_rel"] = sw["p10"] / diag
    f["stroke_w_p50_rel"] = sw["p50"] / diag
    f["stroke_w_p90_rel"] = sw["p90"] / diag
    f["stroke_w_iqr_ratio"] = _safe(sw["p90"], sw["p10"])

    # ── gap width: distance transform of the background, sampled on the
    #    background medial axis, restricted to a band near the ink. Without the
    #    band the outer page dominates and every gap statistic becomes a
    #    measure of how much white margin the renderer left.
    bg = ~mask
    dist_bg = distance_transform_edt(bg)
    band_limit = gap_band_mult * max(med_sw, 1.0)
    near = bg & (dist_bg <= band_limit)
    gap_axis = medial_axis(bg) & near
    gap_w = 2.0 * dist_bg[gap_axis]
    gp = quantiles(gap_w)
    f["gap_w_p10_rel"] = gp["p10"] / diag
    f["gap_w_p50_rel"] = gp["p50"] / diag

    # ── the BANA ratio. Dimensionless, so it is the one thing in here that is
    #    comparable across a 200px thumbnail and a 3000px scan.
    f["gap_over_stroke_p10"] = _safe(gp["p10"], med_sw)
    f["gap_over_stroke_p50"] = _safe(gp["p50"], med_sw)
    f["frac_gaps_below_stroke"] = float((gap_w < med_sw).mean()) if gap_w.size else float("nan")

    # enclosed background: holes the strokes have already closed off
    filled = binary_fill_holes(mask)
    interior_bg = filled & ~mask
    f["enclosed_bg_frac"] = _safe(interior_bg.sum(), filled.sum())
    f["fill_ratio"] = _safe(mask.sum(), filled.sum())

    # ── merge test: close at a radius scaled to the stroke, and see how much
    #    new ink the closure adds relative to the ink already there. Normalised
    #    by ink rather than by background: the background is mostly page
    #    margin, whose size says nothing about whether strokes are merging.
    #
    #    Dilation by a Euclidean disk of radius r IS (dist_bg <= r), so it costs
    #    a comparison on an array already computed. Building disk(r) as an
    #    explicit footprint is what OOM-killed the first run -- a rendering with
    #    a 200px median stroke asks for an 801x801 structuring element.
    for mult in (0.5, 1.0):
        r = max(1.0, mult * med_sw)
        f[f"merge_at_{mult}sw"] = _safe(((dist_bg > 0) & (dist_bg <= r)).sum(), mask.sum())

    # ── topology. Every count is divided by the skeleton length IN DIAGONALS,
    #    not in pixels: a count per pixel still carries 1/resolution, which is
    #    the scale leak these features exist to avoid.
    skel = skeletonize(mask)
    skel_len = float(skel.sum())
    rel_len = skel_len / diag
    nb = neighbour_count(skel)
    endpoints = int(((nb == 1) & skel).sum())
    branches = int(((nb >= 3) & skel).sum())
    _, n_cc = cc_label(mask, structure=CONN8)
    _, n_holes = cc_label(interior_bg)

    f["skel_len_rel"] = rel_len
    f["endpoints_per_len"] = _safe(endpoints, rel_len)
    f["branches_per_len"] = _safe(branches, rel_len)
    f["components_per_len"] = _safe(n_cc, rel_len)
    f["holes_per_len"] = _safe(n_holes, rel_len)
    f["endpoints_per_component"] = _safe(endpoints, n_cc)

    # ── outline vs texture fill. The broken_lines criterion says to ignore
    #    fills, and a stipple fill is thousands of short skeleton components.
    #    Keep components whose skeleton is a meaningful fraction of the longest
    #    one -- a relative cut, so it does not smuggle scale back in.
    #
    #    Every candidate threshold is emitted from one skeleton labelling, under
    #    an `@frac` suffix. This is the experiment's one free parameter: it must
    #    be selected on train/val and reported, so the fitter needs all of them
    #    side by side rather than one baked-in choice.
    slab, n_sk = cc_label(skel, structure=CONN8)
    lens = np.bincount(slab.ravel()) if n_sk else np.zeros(1, dtype=int)
    lens[0] = 0
    longest = lens.max() if n_sk else 0
    for frac in outline_fracs:
        tag = f"@{frac:g}"
        if not n_sk or longest == 0:
            for k in ("outline_frac_of_skel", "outline_components_per_len",
                      "outline_endpoints_per_len", "fill_components_per_len"):
                f[k + tag] = float("nan")
            continue
        keep = lens >= max(2, frac * longest)
        outline = keep[slab]
        o_len = float(outline.sum())
        o_rel = o_len / diag
        o_nb = neighbour_count(outline)
        o_end = int(((o_nb == 1) & outline).sum())
        f["outline_frac_of_skel" + tag] = _safe(o_len, skel_len)
        f["outline_components_per_len" + tag] = _safe(int(keep[1:].sum()), rel_len)
        f["outline_endpoints_per_len" + tag] = _safe(o_end, o_rel)
        f["fill_components_per_len" + tag] = _safe(int((~keep[1:]).sum()), rel_len)

    # ── gap-closure sweep: a real break closes at a small radius, two genuinely
    #    separate objects do not. The radius at which the component count stops
    #    falling is the break scale.
    prev = n_cc
    f["cc_ratio_r0"] = 1.0
    stabilised = None
    for mult in (0.25, 0.5, 1.0, 2.0):
        r = max(1.0, mult * med_sw)
        _, n_r = cc_label(dist_bg <= r, structure=CONN8)
        f[f"cc_ratio_{mult}sw"] = _safe(n_r, n_cc)
        if stabilised is None and prev and n_r >= 0.95 * prev:
            stabilised = mult
        prev = n_r
    f["cc_stabilise_sw"] = float(stabilised) if stabilised is not None else 4.0
    return f


# ── driver ──────────────────────────────────────────────────────────────────

def read_pairs(splits_dir: Path, splits):
    tactile, seen = {}, []
    for sp in splits:
        with open(splits_dir / f"{sp}.jsonl") as fh:
            for line in fh:
                r = json.loads(line)
                if r["pair_id"] not in tactile:
                    tactile[r["pair_id"]] = (r["tactile_image"], sp)
                    seen.append(r["pair_id"])
    return seen, tactile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--split", default="all", help="train | val | test | all")
    ap.add_argument("--cap", type=int, default=1024, help="long-side cap in px")
    ap.add_argument("--speckle-px", type=int, default=4,
                    help="drop ink components smaller than this many px (JPEG speckle)")
    ap.add_argument("--outline-min-frac", default="0.01,0.02,0.05,0.10,0.20",
                    help="comma-separated candidates; each is emitted under an @frac suffix. "
                         "SELECT ON TRAIN/VAL AND REPORT -- never on test.")
    ap.add_argument("--gap-band-mult", type=float, default=5.0,
                    help="gap stats use background within this multiple of the median stroke width")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    require_data_root()
    splits_dir = Path(args.splits_dir) if args.splits_dir else DATA_ROOT / "data" / "splits_v3_unified"
    images = DATA_ROOT / "images"
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    outline_fracs = [float(x) for x in str(args.outline_min_frac).split(",") if x.strip()]

    out_path = Path(args.out) if args.out else None
    if out_path and not out_path.is_absolute():
        out_path = DATA_ROOT / out_path
    if out_path and out_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {out_path} (pass --force)")

    pids, tactile = read_pairs(splits_dir, splits)
    if args.limit:
        pids = pids[: args.limit]
    print(f"{splits_dir.name} [{','.join(splits)}]: {len(pids)} pairs, cap={args.cap}, "
          f"outline_fracs={outline_fracs}")

    feats, failed = {}, []
    t0 = time.time()
    for i, pid in enumerate(pids, 1):
        rel, sp = tactile[pid]
        try:
            gray = load_gray(images / rel, args.cap)
            f = morph_features(gray, args.speckle_px, outline_fracs, args.gap_band_mult)
            f["qc__split"] = sp
            feats[pid] = f
        except Exception as exc:  # a corrupt image must not lose the whole run
            failed.append({"pair_id": pid, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if args.verbose:
            print(f"  {rel[:46]:46} ink={f['ink_fraction']:.3f} amb={f['qc__ambiguity_frac']:.3f} "
                  f"sw50={f.get('stroke_w_p50_rel', float('nan')):.5f} "
                  f"g/s50={f.get('gap_over_stroke_p50', float('nan')):.2f} "
                  f"ep/len={f.get('endpoints_per_len', float('nan')):.3f} "
                  f"out%={f.get(f'outline_frac_of_skel@{outline_fracs[0]:g}', float('nan')):.2f}")
        elif i % 100 == 0:
            print(f"  {i}/{len(pids)}  {time.time() - t0:.0f}s")

    print(f"\nextracted {len(feats)}/{len(pids)} in {time.time() - t0:.0f}s")
    if failed:
        print(f"FAILED {len(failed)}:")
        for r in failed[:10]:
            print("  ", r["pair_id"], r["error"])

    names = sorted({k for f in feats.values() for k in f if not k.startswith("qc__")})
    print(f"\n{len(names)} model features (qc__ excluded), NaN rate:")
    for n in names:
        vals = np.array([feats[p].get(n, np.nan) for p in feats], dtype=float)
        bad = float(np.isnan(vals).mean())
        flag = "  <-- CHECK" if bad > 0.02 else ""
        print(f"  {n:28} nan={bad:5.1%}  median={np.nanmedian(vals):.5g}{flag}")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "splits_dir": str(splits_dir), "splits": splits,
            "params": {"cap": args.cap, "speckle_px": args.speckle_px,
                       "outline_fracs": outline_fracs,
                       "gap_band_mult": args.gap_band_mult},
            "feature_names": names, "failed": failed, "features": feats,
        }, indent=2))
        print(f"\nwrote {out_path}")
    else:
        print("\n(no --out given; nothing written)")


if __name__ == "__main__":
    main()
