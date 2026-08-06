#!/usr/bin/env python3
"""Paired significance test between the two backbone augmentation arms.

  arm A  models_v1          random crop + flip sampled INDEPENDENTLY per image
  arm B  models_v1_pairaug  sampled ONCE per pair (the fix)

Both arms are scored on the SAME test decisions, so the comparison is paired and
McNemar applies. A macro-F1 delta between two single runs cannot separate the
augmentation change from seed noise; what the flips show is whether the two
models actually disagree, and in which direction.

Usage:
  python research/experiments/compare_aug_arms.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT  # noqa: E402

import importlib.util
import json

import torch
from PIL import Image
from scipy.stats import binomtest

R = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bb", R / "baselines/train_backbones_v2.py")
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)

SPL = DATA_ROOT / "data" / "splits_v3_unified"
IMG = DATA_ROOT / "images"
MODELS = DATA_ROOT / "TactileEval_Context_And_Data"
OPTS = bb.ALL_OPTIONS


def load_test():
    pairs = {}
    for line in (SPL / "test.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        p = pairs.setdefault(r["pair_id"], {"nat": r["natural_image"],
                                            "tac": r["tactile_image"], "lab": {}})
        p["lab"][r["option_id"]] = int(r["label"])
    return pairs


def score(arm, backbone, fn, pairs, order, tf):
    ck = MODELS / arm / backbone / fn
    c = torch.load(ck, map_location="cpu", weights_only=False)
    model = bb.TactileEvaluator(c["backbone"])
    model.load_state_dict(c["state_dict"])
    model.eval()
    out = {}
    with torch.no_grad():
        for k, pid in enumerate(order, 1):
            p = pairs[pid]
            n, t = bb.paired_augment(
                Image.open(IMG / p["nat"]).convert("RGB"),
                Image.open(IMG / p["tac"]).convert("RGB"), train=False)
            pr = torch.sigmoid(model(tf(n).unsqueeze(0), tf(t).unsqueeze(0)))[0]
            out[pid] = {o: float(pr[i]) for i, o in enumerate(OPTS)}
            if k % 60 == 0:
                print(f"    {arm}/{backbone}: {k}/{len(order)}", flush=True)
    return out


def mcnemar(a, b, opt, pairs, order):
    keys = [pid for pid in order if opt in pairs[pid]["lab"]]
    ca = [(a[pid][opt] > 0.5) == bool(pairs[pid]["lab"][opt]) for pid in keys]
    cb = [(b[pid][opt] > 0.5) == bool(pairs[pid]["lab"][opt]) for pid in keys]
    wins = sum(1 for x, y in zip(ca, cb) if not x and y)
    loss = sum(1 for x, y in zip(ca, cb) if x and not y)
    p = 1.0 if wins + loss == 0 else binomtest(min(wins, loss), wins + loss, 0.5).pvalue
    return sum(ca), sum(cb), wins, loss, p, len(keys)


def main():
    pairs = load_test()
    order = sorted(pairs)
    tf = bb.get_transforms(False)
    print(f"{len(order)} test pairs x {len(OPTS)} options, CPU\n")

    results = {}
    for backbone, fn, name in (("resnet50", "resnet50_evaluator.pt", "ResNet-50"),
                               ("vit_b16", "vit_b_16_evaluator.pt", "ViT-B/16")):
        A = score("models_v1", backbone, fn, pairs, order, tf)
        B = score("models_v1_pairaug", backbone, fn, pairs, order, tf)
        print(f"\n=== {name} — McNemar exact, A (independent aug) vs B (joint aug) ===")
        print(f"  {'option':<17}{'A ok':>7}{'B ok':>7}{'net':>6}{'flips':>11}{'p':>9}   verdict")
        rows, tb, tc = [], 0, 0
        for o in OPTS:
            na, nb, w, l, p, n = mcnemar(A, B, o, pairs, order)
            rows.append((o, na, nb, w, l, p, n)); tb += w; tc += l
            print(f"  {o:<17}{na:>7}{nb:>7}{nb-na:>+6}{f'+{w}/-{l}':>11}{p:>9.4f}"
                  f"   {'SIGNIFICANT' if p < 0.05 else 'n.s.'}")
        pp = binomtest(min(tb, tc), tb + tc, 0.5).pvalue if tb + tc else 1.0
        ta = sum(r[1] for r in rows); tbb = sum(r[2] for r in rows)
        tn = sum(r[6] for r in rows)
        print(f"  {'- pooled -':<17}{ta:>7}{tbb:>7}{tbb-ta:>+6}{f'+{tb}/-{tc}':>11}{pp:>9.4f}"
              f"   {'SIGNIFICANT' if pp < 0.05 else 'n.s.'}   (n={tn})")
        results[name] = {"per_option": {r[0]: {"A": r[1], "B": r[2], "wins": r[3],
                                               "losses": r[4], "p": r[5], "n": r[6]}
                                        for r in rows},
                         "pooled": {"A": ta, "B": tbb, "wins": tb, "losses": tc,
                                    "p": pp, "n": tn}}

    out = DATA_ROOT / "experiments_out" / "aug_arm_comparison.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
