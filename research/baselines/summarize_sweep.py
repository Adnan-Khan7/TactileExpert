#!/usr/bin/env python3
"""Summarize a seed sweep into the standing results table, with the noise floor.

Reads the checkpoints a sweep wrote (each carries `seed` and `test_results`) and
reports mean ± 2σ per option and macro, per model. Optionally diffs two sweeps and
marks every delta as detectable or not against the 2σ of the sweeps themselves.

WHY 2σ AND NOT A POINT ESTIMATE
-------------------------------
The trainers seeded nothing before 2026-08-05, so checkpoint-to-checkpoint deltas
carried unknown run-to-run noise. On a 183-pair test set with 26 `too_thick`
positives, per-option F1 moves several points on seed alone. Two rounds of
conclusions were retracted for trusting a 3-seed variance estimate. A delta below
2σ is not a result, and this script refuses to present one as though it were.

Usage:
  # one sweep -> the standing table
  python research/baselines/summarize_sweep.py <sweep_root> --markdown

  # two sweeps -> deltas gated on 2σ
  python research/baselines/summarize_sweep.py <new_root> --vs <old_root> --markdown
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT  # noqa: E402

import argparse
import json
import re
import statistics
from collections import defaultdict

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]

# Checkpoint path fragment -> the display name used in every other report and in
# the README results table. Keep these strings identical across tools or the
# meeting-to-meeting tables stop lining up.
DISPLAY = {
    "resnet50":   "ResNet-50",
    "vit_b16":    "ViT-B/16",
    "clip_probe": "CLIP Probe",
    "dino_probe": "DINOv2 Probe",
}
VLM_DISPLAY = "VLM Fine-tuned"


def load_sweep(root: Path):
    """model -> option -> metric -> [values across seeds]  (+ seeds seen).

    Handles both sweep layouts:
      * vision judges — `*_evaluator.pt` checkpoints carrying `seed` and
        `test_results` inside the torch archive
      * the VLM — `s<seed>/test_metrics.json`, because a LoRA run writes an
        adapter plus JSON metrics rather than a self-describing checkpoint
    """
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seeds = defaultdict(set)
    per_seed_macro = defaultdict(lambda: defaultdict(dict))   # model -> seed -> opt -> metrics

    def record(model, seed, per):
        seeds[model].add(seed)
        for opt, m in per.items():
            if opt not in OPTS:
                continue
            for metric in ("f1", "auc"):
                if m.get(metric) is not None:
                    acc[model][opt][metric].append(float(m[metric]))
                    per_seed_macro[model][seed].setdefault(opt, {})[metric] = float(m[metric])

    ckpts = sorted(root.rglob("*_evaluator.pt"))
    if ckpts:
        try:
            import torch
        except ImportError:
            raise SystemExit("torch not importable — activate the venv first:\n"
                             "  source $TACTILE_DATA_ROOT/venv/bin/activate")
        for p in ckpts:
            model = next((DISPLAY[k] for k in DISPLAY if k in p.parts or k in p.name), None)
            if model is None:
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            seed = d.get("seed")
            per = (d.get("test_results") or {}).get("per_option") or {}
            if seed is not None and per:
                record(model, seed, per)

    # VLM: seed comes from the s<N> directory name, since test_metrics.json does
    # not carry it. Only directories matching that convention are read, so a
    # one-off run in an arbitrary directory is ignored rather than mis-seeded.
    for p in sorted(root.rglob("test_metrics.json")):
        seed_dir = next((q for q in p.parents if re.fullmatch(r"s\d+", q.name)), None)
        if seed_dir is None:
            continue
        per = (json.loads(p.read_text()) or {}).get("per_option") or {}
        if per:
            record(VLM_DISPLAY, int(seed_dir.name[1:]), per)

    if not seeds:
        raise SystemExit(f"no per-seed metrics found under {root} "
                         f"(looked for *_evaluator.pt and s<N>/test_metrics.json)")
    return acc, seeds, per_seed_macro


def stats(vals):
    """mean, 2σ. 2σ is undefined below two samples — say so rather than print 0."""
    if not vals:
        return None, None
    mean = statistics.mean(vals)
    two_sd = 2 * statistics.stdev(vals) if len(vals) > 1 else None
    return mean, two_sd


def macro_across_seeds(per_seed, metric):
    """Macro per seed first, then spread across seeds.

    Averaging the per-option spreads instead would understate the macro spread,
    because option errors are correlated within a seed.
    """
    vals = []
    for seed, opts in per_seed.items():
        got = [opts[o][metric] for o in OPTS if o in opts and metric in opts[o]]
        if len(got) == len(OPTS):
            vals.append(statistics.mean(got))
    return stats(vals)


def fmt(mean, two_sd, width=13):
    if mean is None:
        return f"{'—':>{width}}"
    if two_sd is None:
        return f"{mean:>{width}.3f}"
    return f"{mean:.3f} ±{two_sd:.3f}".rjust(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="sweep root (contains backbones/ clip/ dino/)")
    ap.add_argument("--vs", default=None, help="previous sweep root to diff against")
    ap.add_argument("--markdown", action="store_true", help="emit README-ready tables")
    ap.add_argument("--out", default=None, help="also write the summary as JSON")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_absolute():
        root = (DATA_ROOT / root).resolve()
    acc, seeds, per_seed = load_sweep(root)

    models = [m for m in list(DISPLAY.values()) + [VLM_DISPLAY] if m in acc]
    print(f"sweep: {root}")
    for m in models:
        print(f"  {m:14} {len(seeds[m]):2} seeds")
    print()

    summary = {"root": str(root), "seeds": {m: sorted(seeds[m]) for m in models},
               "macro": {}, "per_option": {}}

    for metric in ("auc", "f1"):
        head = f"macro {metric.upper()}"
        print(f"=== {head} and per-option {metric.upper()} (mean ± 2σ over seeds) ===")
        print(f"{'model':14}{head:>16}" + "".join(f"{o[:13]:>16}" for o in OPTS))
        for m in models:
            mac = macro_across_seeds(per_seed[m], metric)
            summary["macro"].setdefault(m, {})[metric] = {"mean": mac[0], "two_sigma": mac[1]}
            cells = []
            for o in OPTS:
                s = stats(acc[m][o].get(metric, []))
                summary["per_option"].setdefault(m, {}).setdefault(o, {})[metric] = {
                    "mean": s[0], "two_sigma": s[1]}
                cells.append(fmt(*s, width=16))
            print(f"{m:14}{fmt(*mac, width=16)}" + "".join(cells))
        print()

    # ── diff ──────────────────────────────────────────────────────────────────
    if args.vs:
        old_root = Path(args.vs).expanduser()
        if not old_root.is_absolute():
            old_root = (DATA_ROOT / old_root).resolve()
        oacc, oseeds, oper = load_sweep(old_root)
        print(f"=== DELTA vs {old_root.name} — detectable only above the pooled 2σ ===")
        print("A delta is called only if |Δ| exceeds the larger of the two 2σ values.")
        print()
        for metric in ("auc", "f1"):
            print(f"  {metric.upper()}")
            print(f"    {'model':14}{'macro':>22}" + "".join(f"{o[:13]:>22}" for o in OPTS))
            for m in models:
                if m not in oacc:
                    continue
                row = []
                nm, ns = macro_across_seeds(per_seed[m], metric)
                om, os_ = macro_across_seeds(oper[m], metric)
                for (a, asd), (b, bsd) in [((nm, ns), (om, os_))] + [
                        (stats(acc[m][o].get(metric, [])), stats(oacc[m][o].get(metric, [])))
                        for o in OPTS]:
                    if a is None or b is None:
                        row.append(f"{'—':>22}")
                        continue
                    d = a - b
                    bar = max(x for x in (asd, bsd, 0) if x is not None)
                    mark = "detected" if abs(d) > bar > 0 else "n.s."
                    row.append(f"{d:+.3f} ({mark})".rjust(22))
                print(f"    {m:14}" + "".join(row))
            print()

    if args.markdown:
        print("=" * 78)
        print("README-READY — paste under the meeting date, keep the headers identical")
        print("=" * 78)
        print()
        # Seed count is per row, not a footnote: mixing seed counts in one table is
        # exactly how a low-seed variance estimate gets quoted as though it were
        # settled. A row below the full sweep width should not be presented.
        print("| Model | Macro AUC | Macro F1 | "
              + " | ".join(f"AUC {o}" for o in OPTS) + " | Seeds |")
        print("|---|" + "---|" * (3 + len(OPTS)))
        order = sorted(models, key=lambda m: -(macro_across_seeds(per_seed[m], "auc")[0] or 0))
        width = max(len(seeds[m]) for m in models)
        for m in order:
            ma = macro_across_seeds(per_seed[m], "auc")
            mf = macro_across_seeds(per_seed[m], "f1")
            def c(v):
                mean, sd = v
                return "—" if mean is None else (f"{mean:.3f} ±{sd:.3f}" if sd else f"{mean:.3f}")
            cells = [c(stats(acc[m][o].get("auc", []))) for o in OPTS]
            n = len(seeds[m])
            flag = "" if n == width else f" / {width}"
            print(f"| {m} | {c(ma)} | {c(mf)} | " + " | ".join(cells)
                  + f" | {n}{flag} |")
        print()
        print(f"_Mean ± 2σ over seeds. Test split. Sweep: `{root.name}`._")
        short = [m for m in models if len(seeds[m]) < width]
        if short:
            print(f"_⚠ incomplete, do not quote: {', '.join(short)} "
                  f"(fewer than {width} seeds)._")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
