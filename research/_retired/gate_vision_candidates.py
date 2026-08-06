"""Deploy-gate the four v5 vision candidates against the deployed incumbents.

Trains: candidates were fine-tuned on data/splits_v2/train.jsonl (5,022 option
rows = 4,128 library + 894 GPT-domain), validated on val.jsonl (146 library
pairs) for early stopping + threshold calibration.

Evaluates on TWO frozen, training-excluded sets:
  * ORIGINAL test  = test.jsonl      153 library pairs x 5 options = 765 decisions
  * GPT holdout    = gpt_test.jsonl   30 GPT pairs   x 5 options = 150 decisions

Metric: each model scores a (natural, tactile) pair -> per-option probability;
flag = prob > 0.50. Original test reported as per-option and macro F1; holdout
reported as correct flags / total (flag == human label). Deployed incumbents
are read from the re-baseline JSON (identical scoring); candidates are scored
here with the same threshold and F1 code. CPU only.

Gate rule (as stated in the thesis): deploy a candidate only if it does not
regress the incumbent on the original test AND improves the GPT holdout.

Usage:  python code/experiments/gate_vision_candidates.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(REPO_ROOT))

HOME = Path.home()
SPLITS = DATA_ROOT / "data/splits_v2"
IMG = DATA_ROOT / "images"
STAGE = DATA_ROOT / "TactileEval_Context_And_Data/models"
FLEET_JSON = DATA_ROOT / "experiments_out/eval_deployed_fleet.json"

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
DISP = {"too_thick": "Too Thick", "broken_lines": "Broken Lines",
        "missing_parts": "Missing Parts", "missing_texture": "Missing Texture",
        "extra_parts": "Extra Parts"}


def load_split(name):
    pairs = {}
    for line in (SPLITS / f"{name}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["option_id"] not in OPTS:
            continue
        p = pairs.setdefault(r["pair_id"], {
            "nat": str(IMG / r["natural_image"]),
            "tac": str(IMG / r["tactile_image"]), "labels": {}})
        p["labels"][r["option_id"]] = r["label"]
    return pairs


def f1(preds, labels):
    tp = sum(p and l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum((not p) and l for p, l in zip(preds, labels))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def per_option_f1(scores, pairs):
    order = sorted(pairs)
    out = {}
    for o in OPTS:
        preds = [scores[pid][o] > 0.5 for pid in order]
        labels = [pairs[pid]["labels"][o] for pid in order]
        out[o] = f1(preds, labels)
    out["macro"] = sum(out[o] for o in OPTS) / len(OPTS)
    return out


def holdout_correct(scores, pairs):
    per_opt = {o: 0 for o in OPTS}
    for pid, p in pairs.items():
        for o in OPTS:
            per_opt[o] += int((scores[pid][o] > 0.5) == bool(p["labels"][o]))
    total = sum(per_opt.values())
    return total, per_opt


def main():
    test = load_split("test")
    gpt = load_split("gpt_test")
    n_test = len(test) * len(OPTS)
    n_gpt = len(gpt) * len(OPTS)
    print(f"original test: {len(test)} pairs x {len(OPTS)} options = {n_test} decisions")
    print(f"GPT holdout:   {len(gpt)} pairs x {len(OPTS)} options = {n_gpt} decisions\n")

    deployed = json.loads(FLEET_JSON.read_text())["results"]  # [split][model][pid][opt]

    from evaluators.clip_probe_eval import CLIPProbeEvaluator
    from evaluators.dino_probe_eval import DINOProbeEvaluator
    from evaluators.backbone_eval import ResNetEvaluator, ViTEvaluator
    makers = {
        "CLIP Probe":   lambda: CLIPProbeEvaluator(models_dir=STAGE / "clip_probe_v2", device="cpu"),
        "DINOv2 Probe": lambda: DINOProbeEvaluator(models_dir=STAGE / "dino_probe_v2", device="cpu"),
        "ResNet-50":    lambda: ResNetEvaluator(checkpoint=STAGE / "resnet50_v2/resnet50_evaluator.pt", device="cpu"),
        "ViT-B/16":     lambda: ViTEvaluator(checkpoint=STAGE / "vitb16_v2/vit_b_16_evaluator.pt", device="cpu"),
    }

    verdicts = []
    for model, make in makers.items():
        print(f"===== {model} =====", flush=True)
        ev = make()
        cand = {"test": {}, "gpt_test": {}}
        for split, pairs in (("test", test), ("gpt_test", gpt)):
            for pid, p in pairs.items():
                cand[split][pid] = ev.score_pair(p["nat"], p["tac"])
        del ev
        gc.collect()

        dep_test_f1 = per_option_f1(deployed["test"][model], test)
        can_test_f1 = per_option_f1(cand["test"], test)
        dep_g_tot, dep_g_opt = holdout_correct(deployed["gpt_test"][model], gpt)
        can_g_tot, can_g_opt = holdout_correct(cand["gpt_test"], gpt)

        print("  original-test F1 (t=0.50):")
        print(f"    {'option':<16}{'deployed':>10}{'candidate':>12}{'delta':>9}")
        for o in OPTS:
            d, c = dep_test_f1[o], can_test_f1[o]
            print(f"    {DISP[o]:<16}{d:>10.3f}{c:>12.3f}{c-d:>+9.3f}")
        dM, cM = dep_test_f1["macro"], can_test_f1["macro"]
        print(f"    {'MACRO':<16}{dM:>10.3f}{cM:>12.3f}{cM-dM:>+9.3f}")
        print(f"  GPT holdout correct flags:  deployed {dep_g_tot}/{n_gpt} "
              f"({dep_g_tot/n_gpt:.1%})  ->  candidate {can_g_tot}/{n_gpt} "
              f"({can_g_tot/n_gpt:.1%})  delta {can_g_tot-dep_g_tot:+d}")

        no_orig_regress = cM >= dM - 0.005          # tolerance: within noise
        holdout_gain = can_g_tot > dep_g_tot
        verdict = "DEPLOY" if (no_orig_regress and holdout_gain) else "REJECT"
        why = []
        why.append("orig-test macro " + ("ok" if no_orig_regress else f"regressed {cM-dM:+.3f}"))
        why.append("holdout " + ("improved" if holdout_gain else "not improved"))
        print(f"  VERDICT: {verdict}  ({'; '.join(why)})\n", flush=True)
        verdicts.append((model, verdict, dM, cM, dep_g_tot, can_g_tot))

    print("===== SUMMARY =====")
    print(f"  {'model':<14}{'verdict':>9}{'orig macro (dep>cand)':>26}{'holdout (dep>cand)':>22}")
    for model, verdict, dM, cM, dg, cg in verdicts:
        print(f"  {model:<14}{verdict:>9}{f'{dM:.3f} > {cM:.3f}':>26}{f'{dg} > {cg}':>22}")


if __name__ == "__main__":
    main()
