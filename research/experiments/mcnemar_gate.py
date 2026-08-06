#!/usr/bin/env python3
"""Paired significance testing for the v1 fleet, per option.

WHY McNEMAR AND NOT AN ACCURACY DELTA: the two systems are scored on the SAME
decisions (same pairs, same options), so their accuracies are paired, not
independent. "+13 correct out of 150" says nothing on its own — what matters is
how many decisions FLIPPED and in which direction. McNemar tests exactly the
discordant decisions and ignores the (usually large) set both got right.

  b = A wrong, B right   (B's wins)
  c = A right, B wrong   (B's losses)

Under the null "a flip is a coin toss", b ~ Binomial(b + c, 0.5). We use the
EXACT binomial test, not the chi-square approximation, because b + c here is in
the tens and the approximation is unreliable at that size.

THREE MODES
-----------
1. --vs-always-clean   Each judge against the degenerate predictor that answers
   "clean" to everything. This is the v1 headline. On the retired GPT holdout the
   always-clean judge scored 142/150 and no fleet member could clear it; the
   unified split exists so that this question has a meaningful answer.

2. --ensemble-vs-best  The weighted ensemble against its own strongest single
   member. Answers "does combining five judges beat just using the best one" —
   without which the fleet is unjustified complexity.

3. --incumbent/--candidate   Two gate runs against each other. The original
   deployment gate; still valid for any two eval_full_v4.py outputs.

EVERY MODE REPORTS PER OPTION. Raw accuracy is degenerate under this class
imbalance — that is the entire reason the v1 dataset was rebuilt.

Usage:
  python research/experiments/mcnemar_gate.py --gate experiments_out/eval_v1_fleet.json --vs-always-clean
  python research/experiments/mcnemar_gate.py --gate experiments_out/eval_v1_fleet.json --ensemble-vs-best
  python research/experiments/mcnemar_gate.py --incumbent A.json --candidate B.json --all-models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

import argparse
import json

from scipy.stats import binomtest

SPLITS = DATA_ROOT / "data" / "splits_v3_unified"
OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
THRESH = 0.50


def load_labels(split, splits_dir):
    """{pair_id: {option: 0/1}} for a frozen split."""
    pairs = {}
    for line in (splits_dir / f"{split}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["option_id"] not in OPTS:
            continue
        pairs.setdefault(r["pair_id"], {})[r["option_id"]] = r["label"]
    return pairs


def decisions(gate_json, model, split, labels):
    """{(pair_id, option): was_correct} at the deployed threshold."""
    res = gate_json["results"][split]
    if model not in res:
        raise SystemExit(f"model {model!r} not in {split}; have {sorted(res)}")
    out = {}
    for pid, per_opt in res[model].items():
        if pid not in labels:
            continue
        for o in OPTS:
            if o in per_opt and o in labels[pid]:
                out[(pid, o)] = (per_opt[o] > THRESH) == bool(labels[pid][o])
    return out


def always_clean_decisions(labels):
    """The degenerate judge: answers 'clean' (0) to every question."""
    return {(pid, o): not bool(labels[pid][o])
            for pid in labels for o in OPTS if o in labels[pid]}


def ensemble_decisions(gate_json, split, labels):
    """Weighted-ensemble decisions via the SHIPPING combiner in constants.py."""
    from evaluators.constants import build_result, build_ensemble_result
    res = gate_json["results"][split]
    members = [m for m in res]
    out = {}
    for pid in labels:
        if any(pid not in res[m] for m in members):
            continue
        mr = {m: build_result(res[m][pid], {}, "balanced") for m in members}
        ens = build_ensemble_result(mr, "balanced")
        for o in OPTS:
            if o in labels[pid] and o in ens["options"]:
                out[(pid, o)] = ens["options"][o]["flagged"] == bool(labels[pid][o])
    return out


def mcnemar(a, b_dec):
    keys = sorted(set(a) & set(b_dec))
    b = sum(1 for k in keys if not a[k] and b_dec[k])
    c = sum(1 for k in keys if a[k] and not b_dec[k])
    n_a = sum(a[k] for k in keys)
    n_b = sum(b_dec[k] for k in keys)
    if b + c == 0:
        return dict(n=len(keys), a=n_a, b_correct=n_b, b=0, c=0, p=1.0,
                    note="decision-identical")
    p = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
    return dict(n=len(keys), a=n_a, b_correct=n_b, b=b, c=c, p=p, note="")


def subset(dec, opt):
    return {k: v for k, v in dec.items() if k[1] == opt}


def prf_auc(gate_json, model, split, labels, opt):
    """Per-option precision / recall / F1 / AUC for one model."""
    res = gate_json["results"][split][model]
    y, s = [], []
    for pid in sorted(labels):
        if pid in res and opt in res[pid] and opt in labels[pid]:
            y.append(int(labels[pid][opt]))
            s.append(float(res[pid][opt]))
    if not y:
        return None
    pred = [int(v > THRESH) for v in s]
    tp = sum(1 for p, t in zip(pred, y) if p and t)
    fp = sum(1 for p, t in zip(pred, y) if p and not t)
    fn = sum(1 for p, t in zip(pred, y) if not p and t)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return dict(n=len(y), pos=sum(y), precision=prec, recall=rec, f1=f1, auc=auc)


def print_per_option(title, a_name, b_name, a_dec, b_dec, alpha):
    print(f"\n{title}")
    print(f"  {'option':<18}{a_name[:11]:>12}{b_name[:11]:>12}"
          f"{'net':>6}{'flips':>12}{'p':>10}   verdict")
    def line(label, r):
        verdict = ("identical" if r["note"] else
                   ("SIGNIFICANT" if r["p"] < alpha else "n.s."))
        flips = f"+{r['b']}/-{r['c']}"
        print(f"  {label:<18}{r['a']:>7}/{r['n']:<4}{r['b_correct']:>7}/{r['n']:<4}"
              f"{r['b_correct'] - r['a']:>+6}{flips:>12}{r['p']:>10.4f}   {verdict}")

    rows = {}
    for o in OPTS:
        rows[o] = mcnemar(subset(a_dec, o), subset(b_dec, o))
        line(o, rows[o])
    agg = mcnemar(a_dec, b_dec)
    line("— all options —", agg)
    return rows, agg


def print_prf_table(gate_json, model, split, labels):
    print(f"\n  per-option quality — {model}")
    print(f"    {'option':<18}{'n':>6}{'pos':>6}{'prec':>8}{'recall':>8}{'F1':>8}{'AUC':>8}")
    macro = {"f1": [], "auc": []}
    for o in OPTS:
        m = prf_auc(gate_json, model, split, labels, o)
        if not m:
            continue
        macro["f1"].append(m["f1"])
        macro["auc"].append(m["auc"])
        print(f"    {o:<18}{m['n']:>6}{m['pos']:>6}{m['precision']:>8.3f}"
              f"{m['recall']:>8.3f}{m['f1']:>8.3f}{m['auc']:>8.3f}")
    if macro["f1"]:
        import math
        aucs = [a for a in macro["auc"] if not math.isnan(a)]
        print(f"    {'MACRO':<18}{'':>6}{'':>6}{'':>8}{'':>8}"
              f"{sum(macro['f1'])/len(macro['f1']):>8.3f}"
              f"{(sum(aucs)/len(aucs) if aucs else float('nan')):>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", help="a single eval_full_v4.py output (modes 1 and 2)")
    ap.add_argument("--incumbent")
    ap.add_argument("--candidate")
    ap.add_argument("--vs-always-clean", action="store_true")
    ap.add_argument("--ensemble-vs-best", action="store_true")
    ap.add_argument("--model", default="VLM Fine-tuned")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="v1 has no gpt_test — the two-corpus regime was retired")
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="write results as JSON")
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir) if args.splits_dir else SPLITS
    labels = load_labels(args.split, splits_dir)
    n_dec = sum(len(v) for v in labels.values())
    n_pos = sum(sum(v.values()) for v in labels.values())
    print(f"split={args.split}  {len(labels)} pairs  {n_dec} decisions  "
          f"{n_pos} positive ({n_pos/n_dec:.1%})")
    print(f"always-clean baseline: {n_dec-n_pos}/{n_dec} correct "
          f"({1-n_pos/n_dec:.1%})   threshold={THRESH}  alpha={args.alpha}")
    print(f"labels: {splits_dir}")

    out = {"split": args.split, "splits_dir": str(splits_dir),
           "n_pairs": len(labels), "n_decisions": n_dec, "n_positive": n_pos,
           "always_clean_correct": n_dec - n_pos, "alpha": args.alpha,
           "comparisons": []}

    if args.vs_always_clean or args.ensemble_vs_best:
        if not args.gate:
            raise SystemExit("--gate is required for --vs-always-clean / --ensemble-vs-best")
        gate = json.loads(Path(args.gate).read_text())
        print(f"gate: {args.gate}")

    if args.vs_always_clean:
        base = always_clean_decisions(labels)
        models = list(gate["results"][args.split]) if args.all_models else [args.model]
        for m in models:
            rows, agg = print_per_option(
                f"McNemar — ALWAYS-CLEAN vs {m}",
                "always-clean", m, base, decisions(gate, m, args.split, labels),
                args.alpha)
            print_prf_table(gate, m, args.split, labels)
            out["comparisons"].append({"mode": "vs_always_clean", "model": m,
                                       "per_option": rows, "aggregate": agg})

    if args.ensemble_vs_best:
        ens = ensemble_decisions(gate, args.split, labels)
        members = list(gate["results"][args.split])
        scored = {m: sum(decisions(gate, m, args.split, labels).values()) for m in members}
        best = max(scored, key=scored.get)
        print(f"\nbest single member by correct decisions: {best} "
              f"({scored[best]}/{n_dec})")
        rows, agg = print_per_option(
            f"McNemar — {best} vs ENSEMBLE",
            best, "ensemble", decisions(gate, best, args.split, labels), ens,
            args.alpha)
        out["comparisons"].append({"mode": "ensemble_vs_best", "best_member": best,
                                   "member_correct": scored,
                                   "per_option": rows, "aggregate": agg})

    if args.incumbent and args.candidate:
        inc_j = json.loads(Path(args.incumbent).read_text())
        cand_j = json.loads(Path(args.candidate).read_text())
        models = ([m for m in inc_j["results"][args.split]
                   if m in cand_j["results"][args.split]]
                  if args.all_models else [args.model])
        print(f"\nincumbent: {args.incumbent}\ncandidate: {args.candidate}")
        for m in models:
            rows, agg = print_per_option(
                f"McNemar — incumbent vs candidate — {m}",
                "incumbent", "candidate",
                decisions(inc_j, m, args.split, labels),
                decisions(cand_j, m, args.split, labels), args.alpha)
            out["comparisons"].append({"mode": "incumbent_vs_candidate", "model": m,
                                       "per_option": rows, "aggregate": agg})

    if not out["comparisons"]:
        raise SystemExit("nothing to do — pass --vs-always-clean, --ensemble-vs-best, "
                         "or both --incumbent and --candidate")

    print("\n  b = first system wrong / second right (wins)")
    print("  c = first system right / second wrong (losses)")
    print("  A large net gain with a non-significant p means the flips are")
    print("  consistent with chance — do NOT gate on the net alone.")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
