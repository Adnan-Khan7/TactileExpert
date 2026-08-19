"""Trivial single-scalar baselines, and the provenance composition of a split.

Two questions this answers, both cheap and both load-bearing for README §4h/§4i:

  1. How much of each option's AUC is available from ONE untrained scalar —
     the fraction of dark pixels in the tactile rendering? This is the bar a
     morphological feature set has to clear. It is not 0.5.

  2. How much of `missing_texture` is separation between image sources rather
     than defect detection? The split is near-confounded with provenance, and
     the within-source estimate is too imprecise to publish, so the composition
     is reported instead of an AUC.

Both are read-only. The paired comparison against a deployed fleet needs a
per-pair score artifact (experiments_out/eval_*_fleet.json) and the label state
it was scored at — `splits_v3_unified/` is rewritten in place, so the artifact's
own `splits_dir` cannot be trusted to identify that state.

Usage:
    python research/morphology/trivial_baselines.py \
        --split test --out experiments_out/trivial_baselines_tv974.json
    python research/morphology/trivial_baselines.py \
        --split test --eval-json experiments_out/eval_tv918_fleet.json \
        --eval-labels data/splits_v3_unified_backup_20260817_171443 \
        --out experiments_out/trivial_baselines_tv918.json
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _binarize import ink_fraction  # noqa: E402  (one shared definition)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, require_data_root  # noqa: E402

OPTIONS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]


def source_of(tactile_rel: str) -> str:
    """web = scraped corpus, gpt = pipeline-generated (v1 and v2 rounds)."""
    return "gpt" if tactile_rel.startswith("gpt_generated") else "web"


def read_split(splits_dir: Path, split: str):
    labels = collections.defaultdict(dict)
    tactile, synthetic = {}, {}
    with open(splits_dir / f"{split}.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            pid = r["pair_id"]
            labels[pid][r["option_id"]] = int(r["label"])
            tactile[pid] = r["tactile_image"]
            synthetic[pid] = bool(r.get("synthetic_corruption", False))
    return labels, tactile, synthetic


def boot_indices(y: np.ndarray, n_boot: int, seed: int):
    """Stratified pair resampling — keeps the positive count fixed across draws.

    Unstratified resampling of a 27-positive split occasionally yields zero
    positives and an undefined AUC.
    """
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    return [
        np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        for _ in range(n_boot)
    ]


def auc_ci(y, x, idx):
    point = roc_auc_score(y, x)
    draws = np.array([roc_auc_score(y[i], x[i]) for i in idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), draws


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", default=None, help="default: $TACTILE_DATA_ROOT/data/splits_v3_unified")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cap", type=int, default=1024, help="long-side cap in px; 0 disables")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-json", default=None, help="per-pair score artifact for the paired comparison")
    ap.add_argument("--eval-labels", default=None, help="splits dir holding the label state --eval-json was scored at")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = ap.parse_args()

    require_data_root()
    splits_dir = Path(args.splits_dir) if args.splits_dir else DATA_ROOT / "data" / "splits_v3_unified"
    images = DATA_ROOT / "images"
    cap = args.cap or None

    out_path = Path(args.out) if args.out else None
    if out_path and not out_path.is_absolute():
        out_path = DATA_ROOT / out_path
    if out_path and out_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {out_path} (pass --force)")

    labels, tactile, synthetic = read_split(splits_dir, args.split)
    pids = sorted(labels)
    print(f"{splits_dir.name}/{args.split}: {len(pids)} pairs, cap={cap}")

    report = {
        "splits_dir": str(splits_dir),
        "split": args.split,
        "n_pairs": len(pids),
        "cap": cap,
        "n_boot": args.n_boot,
        "seed": args.seed,
    }

    # ── 1. composition by source ────────────────────────────────────────────
    by_src = collections.defaultdict(list)
    for pid in pids:
        by_src[source_of(tactile[pid])].append(pid)
    comp = {}
    for src, group in sorted(by_src.items()):
        comp[src] = {"n": len(group)}
        for opt in OPTIONS:
            k = sum(labels[p][opt] for p in group)
            comp[src][opt] = {"pos": int(k), "neg": int(len(group) - k)}
    report["composition_by_source"] = comp
    print("\ncomposition by source")
    print(f"  {'source':7} {'n':>4}  " + "  ".join(f"{o:>17}" for o in OPTIONS))
    for src, c in comp.items():
        cells = [f"{c[o]['pos']:4}+/{c[o]['neg']:4}-" for o in OPTIONS]
        print(f"  {src:7} {c['n']:4}  " + "  ".join(f"{x:>17}" for x in cells))

    # ── 2. ink-fraction baseline ────────────────────────────────────────────
    ink = np.array([ink_fraction(images / tactile[p], cap) for p in pids])
    report["ink_fraction"] = {"mean": float(ink.mean()), "std": float(ink.std())}
    base = {}
    print(f"\nink-fraction baseline (untrained, deterministic) — {args.split}")
    print(f"  {'option':16} {'n_pos':>5} {'AUC':>6}  {'95% CI':>16} {'width':>6}")
    for opt in OPTIONS:
        y = np.array([labels[p][opt] for p in pids])
        idx = boot_indices(y, args.n_boot, args.seed)
        a, lo, hi, _ = auc_ci(y, ink, idx)
        base[opt] = {"n_pos": int(y.sum()), "auc": a, "ci95": [lo, hi]}
        print(f"  {opt:16} {y.sum():5} {a:6.3f}  [{lo:.3f}, {hi:.3f}] {hi - lo:6.3f}")
    report["ink_fraction_auc"] = base

    # natural-only arm — the synthetic corruptions are an easier target
    nat = [p for p in pids if not synthetic[p]]
    report["n_synthetic"] = len(pids) - len(nat)
    if len(nat) < len(pids):
        nat_res = {}
        print(f"\n  natural-only ({len(nat)} pairs, {len(pids) - len(nat)} synthetic_corruption excluded)")
        for opt in ("too_thick", "broken_lines"):
            y = np.array([labels[p][opt] for p in nat])
            x = np.array([ink[pids.index(p)] for p in nat])
            idx = boot_indices(y, args.n_boot, args.seed)
            a, lo, hi, _ = auc_ci(y, x, idx)
            nat_res[opt] = {"n_pos": int(y.sum()), "auc": a, "ci95": [lo, hi]}
            print(f"  {opt:16} {y.sum():5} {a:6.3f}  [{lo:.3f}, {hi:.3f}] {hi - lo:6.3f}")
        report["ink_fraction_auc_natural_only"] = nat_res

    # ── 3. paired comparison against a scored fleet ─────────────────────────
    if args.eval_json:
        if not args.eval_labels:
            raise SystemExit(
                "--eval-json needs --eval-labels: splits_v3_unified/ is rewritten in place, so the "
                "artifact's own splits_dir does not identify the label state it was scored at."
            )
        ev = json.loads(Path(args.eval_json).read_text())
        ev_labels, ev_tactile, _ = read_split(Path(args.eval_labels), args.split)
        scored = ev["results"][args.split]
        ev_pids = sorted(next(iter(scored.values())))
        missing = [p for p in ev_pids if p not in ev_labels]
        if missing:
            raise SystemExit(f"{len(missing)} scored pairs absent from --eval-labels; wrong label state?")

        # the join is only trustworthy if it reproduces the artifact's own metrics
        drift = []
        for model, per_pair in scored.items():
            for opt in OPTIONS:
                y = np.array([ev_labels[p][opt] for p in ev_pids])
                x = np.array([per_pair[p][opt] for p in ev_pids])
                stored = ev["metrics"][model][opt]["auc"]
                if abs(roc_auc_score(y, x) - stored) > 1e-6:
                    drift.append(f"{model}/{opt}")
        if drift:
            raise SystemExit(f"label join disagrees with stored metrics on {len(drift)}: {drift[:5]}")
        print(f"\njoin validated against stored metrics ({len(scored)} models x {len(OPTIONS)} options)")

        ev_ink = np.array([ink_fraction(images / ev_tactile[p], cap) for p in ev_pids])
        paired = {}
        print(f"\npaired vs ink fraction — {Path(args.eval_json).name}, {len(ev_pids)} pairs")
        for opt in ("too_thick", "broken_lines"):
            y = np.array([ev_labels[p][opt] for p in ev_pids])
            idx = boot_indices(y, args.n_boot, args.seed)
            a_ink, lo_i, hi_i, d_ink = auc_ci(y, ev_ink, idx)
            print(f"\n  {opt} ({y.sum()} pos) — ink fraction {a_ink:.3f} [{lo_i:.3f}, {hi_i:.3f}]")
            paired[opt] = {
                "ink_fraction": {"auc": a_ink, "ci95": [lo_i, hi_i], "n_pos": int(y.sum())},
                "models": {},
            }
            print(f"    {'model':20} {'AUC':>6}  {'95% CI':>16}  {'delta':>7}  {'paired 95% CI':>18}")
            for model, per_pair in scored.items():
                x = np.array([per_pair[p][opt] for p in ev_pids])
                a_m, lo_m, hi_m, d_m = auc_ci(y, x, idx)
                diff = d_m - d_ink
                dlo, dhi = np.percentile(diff, [2.5, 97.5])
                sig = bool(dlo > 0 or dhi < 0)
                paired[opt]["models"][model] = {
                    "auc": a_m, "ci95": [lo_m, hi_m],
                    "delta_vs_ink": a_m - a_ink, "delta_ci95": [float(dlo), float(dhi)],
                    "significant": sig,
                }
                print(f"    {model:20} {a_m:6.3f}  [{lo_m:.3f}, {hi_m:.3f}]  {a_m - a_ink:+7.3f}  "
                      f"[{dlo:+.3f}, {dhi:+.3f}] {'SIG' if sig else 'n.s.'}")
        report["paired_vs_ink"] = paired
        report["eval_json"] = str(args.eval_json)
        report["eval_labels"] = str(args.eval_labels)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out_path}")
    else:
        print("\n(no --out given; nothing written)")


if __name__ == "__main__":
    main()
