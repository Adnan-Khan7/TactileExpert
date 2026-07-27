"""Full evaluation of the freshly trained (v4) fleet for the README tables.

Scores all 5 new checkpoints on:
  - the original 153-pair test set  -> per-option F1 (t=0.50) + AUC
  - the frozen GPT-domain holdout   -> correct flag decisions
Then derives ENSEMBLE_WEIGHTS from the new per-option original-test F1s,
applies the weighted-vote ensemble to both sets, and prints:
  - README-ready markdown tables (F1, AUC, holdout) incl. Ensemble column
  - a paste-ready ENSEMBLE_WEIGHTS dict for evaluators/constants.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(REPO_ROOT))

HOME = Path.home()
SPLITS = DATA_ROOT / "data/splits_v2"
IMG_ROOT = DATA_ROOT / "images"
NEW = DATA_ROOT / "TactileEval_Context_And_Data/models"
# Default = the DEPLOYED VLM. Override with --vlm-ckpt to gate a candidate
# checkpoint against the deployed probes/backbones before deploying it.
VLM_CKPT = REPO_ROOT / "models/vlm_checkpoint"
OUT = DATA_ROOT / "experiments_out/eval_deployed_fleet.json"

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
DISPLAY = {"too_thick": "Too Thick", "broken_lines": "Broken Lines",
           "missing_parts": "Missing Parts", "missing_texture": "Missing Texture",
           "extra_parts": "Extra Parts"}
MODELS = ["CLIP Probe", "VLM Fine-tuned", "ResNet-50", "ViT-B/16", "DINOv2 Probe"]


def load_split(name):
    pairs = {}
    for line in (SPLITS / f"{name}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["option_id"] not in OPTS:
            continue
        pid = r["pair_id"]
        p = pairs.setdefault(pid, {
            "nat": str(IMG_ROOT / r["natural_image"]),
            "tac": str(IMG_ROOT / r["tactile_image"]),
            "labels": {},
        })
        p["labels"][r["option_id"]] = r["label"]
    return pairs


def f1(preds, labels):
    tp = sum(p and l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum((not p) and l for p, l in zip(preds, labels))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def score_all(pairs_by_split, base_model="Qwen/Qwen2-VL-2B-Instruct"):
    from evaluators.clip_probe_eval import CLIPProbeEvaluator
    from evaluators.dino_probe_eval import DINOProbeEvaluator
    from evaluators.backbone_eval import ResNetEvaluator, ViTEvaluator

    makers = {
        "CLIP Probe":   lambda: CLIPProbeEvaluator(models_dir=REPO_ROOT / "models/clip_probe"),
        "DINOv2 Probe": lambda: DINOProbeEvaluator(models_dir=REPO_ROOT / "models/dino_probe"),
        "ResNet-50":    lambda: ResNetEvaluator(checkpoint=REPO_ROOT / "models/resnet50/resnet50_evaluator.pt"),
        "ViT-B/16":     lambda: ViTEvaluator(checkpoint=REPO_ROOT / "models/vit_b16/vit_b_16_evaluator.pt"),
    }
    results = {s: {m: {} for m in MODELS} for s in pairs_by_split}
    for model, make in makers.items():
        ev = make()
        for split, pairs in pairs_by_split.items():
            for pid, p in pairs.items():
                results[split][model][pid] = ev.score_pair(p["nat"], p["tac"])
        del ev
        gc.collect(); torch.cuda.empty_cache()
        print(f"scored {model}", flush=True)

    # VLM (v4 checkpoint, v3 question wording from constants — aligned)
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel
    from evaluators.constants import QUESTIONS, SYSTEM_PROMPT

    print(f"VLM base model: {base_model}", flush=True)
    processor = AutoProcessor.from_pretrained(base_model, max_pixels=200704)
    tok = processor.tokenizer
    yes_ids = [tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("Yes")]
    no_ids = [tok.convert_tokens_to_ids("no"), tok.convert_tokens_to_ids("No")]
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": "cuda"})
    model = PeftModel.from_pretrained(base, str(VLM_CKPT)).merge_and_unload().eval()

    @torch.no_grad()
    def vscore(nat, tac, q):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{nat}"},
                {"type": "image", "image": f"file://{tac}"},
                {"type": "text", "text": q},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        logits = model(**inputs).logits[0, -1, :]
        y = max(logits[i].item() for i in yes_ids)
        n = max(logits[i].item() for i in no_ids)
        return torch.softmax(torch.tensor([y, n]), dim=0)[0].item()

    for split, pairs in pairs_by_split.items():
        for pid, p in pairs.items():
            results[split]["VLM Fine-tuned"][pid] = {
                o: round(vscore(p["nat"], p["tac"], QUESTIONS[o]), 4) for o in OPTS}
        print(f"scored VLM on {split}", flush=True)
    del model, base
    gc.collect(); torch.cuda.empty_cache()
    return results


def main():
    global VLM_CKPT, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-ckpt", type=str, default=None,
                        help="LoRA checkpoint dir to evaluate instead of the "
                             "deployed models/vlm_checkpoint (deploy gate)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: eval_deployed_fleet.json)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                        help="VLM base for the LoRA adapter (use Qwen/Qwen2-VL-7B-Instruct "
                             "when gating a 7B checkpoint)")
    args = parser.parse_args()
    if args.vlm_ckpt:
        VLM_CKPT = Path(args.vlm_ckpt)
        print(f"VLM checkpoint override: {VLM_CKPT}", flush=True)
    if args.out:
        OUT = Path(args.out)

    pairs_by_split = {"test": load_split("test"), "gpt_test": load_split("gpt_test")}
    for s, p in pairs_by_split.items():
        print(f"{s}: {len(p)} pairs", flush=True)

    results = score_all(pairs_by_split, base_model=args.base_model)

    # per-model per-option F1/AUC on original test → ensemble weights
    test = pairs_by_split["test"]
    order = sorted(test)
    metrics = {m: {} for m in MODELS}
    for m in MODELS:
        for o in OPTS:
            scores = [results["test"][m][pid][o] for pid in order]
            labels = [test[pid]["labels"][o] for pid in order]
            metrics[m][o] = {"f1": f1([s > 0.5 for s in scores], labels),
                             "auc": auc(scores, labels)}
    weights = {o: {m: round(metrics[m][o]["f1"], 3) for m in MODELS} for o in OPTS}

    # ensemble on both splits (vote at t=0.50, weighted by new F1s)
    ens = {}
    for split, pairs in pairs_by_split.items():
        ens[split] = {}
        for pid in pairs:
            ens[split][pid] = {}
            for o in OPTS:
                tw = sum(weights[o][m] for m in MODELS)
                vote = sum(weights[o][m] for m in MODELS
                           if results[split][m][pid][o] > 0.5) / tw
                mean = sum(weights[o][m] * results[split][m][pid][o] for m in MODELS) / tw
                ens[split][pid][o] = {"flag": vote > 0.5, "mean": mean}

    # ── README tables ───────────────────────────────────────────────────────
    def cell(v): return f"{v:.3f}"
    print("\n### F1 (original test, t=0.50)\n")
    hdr = "| Dimension | " + " | ".join(["CLIP Probe", "VLM", "ResNet-50", "ViT-B/16", "DINOv2 Probe", "Ensemble"]) + " |"
    print(hdr); print("|---|" + ":---:|" * 6)
    macro = {m: [] for m in MODELS}; macro_e = []
    for o in OPTS:
        e_f1 = f1([ens["test"][pid][o]["flag"] for pid in order],
                  [test[pid]["labels"][o] for pid in order])
        macro_e.append(e_f1)
        row = [cell(metrics[m][o]["f1"]) for m in MODELS] + [cell(e_f1)]
        for m in MODELS: macro[m].append(metrics[m][o]["f1"])
        print(f"| {DISPLAY[o]} | " + " | ".join(row) + " |")
    print(f"| **Macro F1** | " + " | ".join(cell(sum(macro[m]) / 5) for m in MODELS)
          + f" | {cell(sum(macro_e) / 5)} |")

    print("\n### AUC (original test)\n")
    print(hdr); print("|---|" + ":---:|" * 6)
    macro_auc = {m: [] for m in MODELS}; macro_ea = []
    for o in OPTS:
        e_auc = auc([ens["test"][pid][o]["mean"] for pid in order],
                    [test[pid]["labels"][o] for pid in order])
        macro_ea.append(e_auc)
        row = [cell(metrics[m][o]["auc"]) for m in MODELS] + [cell(e_auc)]
        for m in MODELS: macro_auc[m].append(metrics[m][o]["auc"])
        print(f"| {DISPLAY[o]} | " + " | ".join(row) + " |")
    print(f"| **Macro AUC** | " + " | ".join(cell(sum(macro_auc[m]) / 5) for m in MODELS)
          + f" | {cell(sum(macro_ea) / 5)} |")

    print("\n### GPT-domain holdout (correct flags)\n")
    gt = pairs_by_split["gpt_test"]; gorder = sorted(gt)
    n = len(gorder) * len(OPTS)
    cells = []
    for m in MODELS:
        c = sum((results["gpt_test"][m][pid][o] > 0.5) == bool(gt[pid]["labels"][o])
                for pid in gorder for o in OPTS)
        cells.append(f"{c}/{n}")
    ce = sum(ens["gpt_test"][pid][o]["flag"] == bool(gt[pid]["labels"][o])
             for pid in gorder for o in OPTS)
    cells.append(f"{ce}/{n}")
    print("| | " + " | ".join(["CLIP Probe", "VLM", "ResNet-50", "ViT-B/16", "DINOv2 Probe", "Ensemble"]) + " |")
    print("|---|" + ":---:|" * 6)
    print("| Correct / total | " + " | ".join(cells) + " |")

    print("\n### ENSEMBLE_WEIGHTS (paste into constants.py)\n")
    print("ENSEMBLE_WEIGHTS = {")
    for o in OPTS:
        entries = ", ".join(f'"{m}": {weights[o][m]}' for m in MODELS)
        print(f'    "{o}": {{{entries}}},')
    print("}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"metrics": {m: {o: metrics[m][o] for o in OPTS} for m in MODELS},
                               "weights": weights,
                               "results": results,
                               "holdout_correct": dict(zip(MODELS + ["Ensemble"], cells))},
                              indent=2, default=float))
    print(f"\nsaved → {OUT}")


if __name__ == "__main__":
    main()
