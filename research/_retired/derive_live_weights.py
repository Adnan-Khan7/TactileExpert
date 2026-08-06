"""Derive ENSEMBLE_WEIGHTS from LIVE disagreement records (July 14, 2026).

Replaces the original-test-F1 weights: those were too flat for the GPT-image-1
domain (July 9 finding: VLM v3 alone beat the weighted ensemble on live data).

Method
------
Per model, per option, accuracy over two kinds of live records from
ensemble-era trajectory pairs (evaluator_flags contains the Ensemble entry):
  1. FINALS: the model's stored flag (balanced t=0.50) vs the human checkbox
     label on the saved final image. Finals are mostly clean, so this mainly
     measures false-positive behaviour (specificity).
  2. INFERRED POSITIVES: for every edit step with a target_option, the
     PRE-edit iteration is defect-positive for that option; the model's stored
     score > 0.5 counts as correct. This adds sensitivity measurement.
Accuracy (not F1) because per-option positives in finals are near zero.

Validation: re-scores the weighted-vote ensemble on the frozen original test
and gpt_test using the per-model scores stored in eval_deployed_fleet.json —
new weights must not be worse than the old ones on either set.

Usage:  python code/experiments/derive_live_weights.py
Paste the printed dict into TactileExpert/evaluators/constants.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import json
from collections import defaultdict
from pathlib import Path

HOME = Path.home()
TRAJ = REPO_ROOT / "generated_training_data/trajectories.jsonl"
FLEET_EVAL = DATA_ROOT / "experiments_out/eval_deployed_fleet.json"
SPLITS = DATA_ROOT / "data/splits_v2"

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
MODELS = ["CLIP Probe", "VLM Fine-tuned", "ResNet-50", "ViT-B/16", "DINOv2 Probe"]
ENSEMBLE = "Ensemble (weighted)"


def derive():
    trajs = [json.loads(l) for l in TRAJ.read_text().splitlines()]
    era = [t for t in trajs if ENSEMBLE in t.get("evaluator_flags", {})]
    correct = defaultdict(lambda: defaultdict(int))
    total = defaultdict(lambda: defaultdict(int))
    n_inferred = 0
    for t in era:
        hl = t["human_labels"]
        for m in MODELS:
            fl = t["evaluator_flags"].get(m, {})
            for o in OPTS:
                if o in fl:
                    total[m][o] += 1
                    correct[m][o] += (fl[o] == hl[o])
        steps = t["steps"]
        for si, s in enumerate(steps):
            o = s.get("target_option", "")
            if si == 0 or o not in OPTS:
                continue
            prev = steps[si - 1].get("scores", {})
            hit = False
            for m in MODELS:
                if m in prev and o in prev[m]:
                    total[m][o] += 1
                    correct[m][o] += (prev[m][o] > 0.5)
                    hit = True
            n_inferred += hit
    print(f"ensemble-era pairs: {len(era)}   inferred-positive records: {n_inferred}")
    return {o: {m: round(correct[m][o] / total[m][o], 3) for m in MODELS} for o in OPTS}


def load_split(name):
    pairs = {}
    for line in (SPLITS / f"{name}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["option_id"] in OPTS:
            pairs.setdefault(r["pair_id"], {})[r["option_id"]] = r["label"]
    return pairs


def f1(preds, labels):
    tp = sum(p and l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum((not p) and l for p, l in zip(preds, labels))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def validate(weights):
    res = json.loads(FLEET_EVAL.read_text())["results"]
    test = load_split("test")
    macro = []
    for o in OPTS:
        tw = sum(weights[o][m] for m in MODELS)
        preds = [sum(weights[o][m] for m in MODELS if res["test"][m][pid][o] > 0.5) / tw > 0.5
                 for pid in sorted(test)]
        macro.append(f1(preds, [test[pid][o] for pid in sorted(test)]))
    gt = load_split("gpt_test")
    c = sum((sum(weights[o][m] for m in MODELS if res["gpt_test"][m][pid][o] > 0.5)
             / sum(weights[o][m] for m in MODELS) > 0.5) == bool(gt[pid][o])
            for pid in sorted(gt) for o in OPTS)
    return sum(macro) / len(macro), f"{c}/{len(gt) * len(OPTS)}"


def main():
    weights = derive()
    m, h = validate(weights)
    print(f"validation — original test macro F1: {m:.4f}   gpt holdout: {h}")
    print("\nENSEMBLE_WEIGHTS = {")
    for o in OPTS:
        entries = ", ".join(f'"{mo}": {weights[o][mo]}' for mo in MODELS)
        print(f'    "{o}": {{{entries}}},')
    print("}")


if __name__ == "__main__":
    main()
