"""Before/after evaluation of all 5 evaluators on the frozen gpt_test holdout.

OLD = July-4 checkpoints deployed in ~/TactileExpert/models/
NEW = v3 checkpoints from the July-8 retrain (TactileEval_Context_And_Data/models/*_v2,
      VLM from vlm_finetune/output_v2/best_checkpoint)

The v3 VLM was trained with a reworded broken_lines question; each VLM
checkpoint is therefore scored with ITS OWN question text.

Output: printed per-option flag counts (t=0.50) + per-pair probability table,
JSON dump to experiments_out/gpt_test_eval.json.
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
SPLITS = DATA_ROOT / "data/splits_v2/gpt_test.jsonl"
IMG_ROOT = DATA_ROOT / "images"
NEW_MODELS = DATA_ROOT / "TactileEval_Context_And_Data/models"
OLD_MODELS = REPO_ROOT / "models"
OUT = DATA_ROOT / "experiments_out/gpt_test_eval.json"

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]

BROKEN_LINES_Q_V3 = (
    "Consider only the structural outlines of the object, not any texture "
    "or fill patterns inside its regions. Are the outlines broken or "
    "discontinuous, so that a fingertip tracing an edge would encounter "
    "gaps, interruptions, or missing segments?"
)


def load_pairs():
    pairs = {}
    for line in SPLITS.read_text().splitlines():
        r = json.loads(line)
        ts = r["natural_image"].split("/")[1]  # pair_<ts>
        p = pairs.setdefault(ts, {
            "object": r["object"],
            "nat": str(IMG_ROOT / r["natural_image"]),
            "tac": str(IMG_ROOT / r["tactile_image"]),
            "labels": {},
        })
        p["labels"][r["option_id"]] = r["label"]
    return pairs


def score_probe_family(pairs, results):
    from evaluators.clip_probe_eval import CLIPProbeEvaluator
    from evaluators.dino_probe_eval import DINOProbeEvaluator
    from evaluators.backbone_eval import ResNetEvaluator, ViTEvaluator

    configs = [
        ("CLIP",   "old", lambda: CLIPProbeEvaluator(models_dir=OLD_MODELS / "clip_probe")),
        ("CLIP",   "new", lambda: CLIPProbeEvaluator(models_dir=NEW_MODELS / "clip_probe_v2")),
        ("DINOv2", "old", lambda: DINOProbeEvaluator(models_dir=OLD_MODELS / "dino_probe")),
        ("DINOv2", "new", lambda: DINOProbeEvaluator(models_dir=NEW_MODELS / "dino_probe_v2")),
        ("ResNet", "old", lambda: ResNetEvaluator(checkpoint=OLD_MODELS / "resnet50/resnet50_evaluator.pt")),
        ("ResNet", "new", lambda: ResNetEvaluator(checkpoint=NEW_MODELS / "resnet50_v2/resnet50_evaluator.pt")),
        ("ViT",    "old", lambda: ViTEvaluator(checkpoint=OLD_MODELS / "vit_b16/vit_b_16_evaluator.pt")),
        ("ViT",    "new", lambda: ViTEvaluator(checkpoint=NEW_MODELS / "vitb16_v2/vit_b_16_evaluator.pt")),
    ]
    for model, ver, make in configs:
        ev = make()
        for ts, p in pairs.items():
            results[f"{model}/{ver}"][ts] = ev.score_pair(p["nat"], p["tac"])
        del ev
        gc.collect(); torch.cuda.empty_cache()
        print(f"scored {model}/{ver}", flush=True)


def score_vlm(pairs, results):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel
    from evaluators.constants import QUESTIONS, SYSTEM_PROMPT

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", max_pixels=200704)
    tok = processor.tokenizer
    yes_ids = [tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("Yes")]
    no_ids = [tok.convert_tokens_to_ids("no"), tok.convert_tokens_to_ids("No")]

    @torch.no_grad()
    def score(model, nat, tac, question):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{nat}"},
                {"type": "image", "image": f"file://{tac}"},
                {"type": "text", "text": question},
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

    for ver, lora, questions in [
        ("old", OLD_MODELS / "vlm_checkpoint", dict(QUESTIONS)),
        ("new", DATA_ROOT / "output_v2/best_checkpoint",
         {**QUESTIONS, "broken_lines": BROKEN_LINES_Q_V3}),
    ]:
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map={"": "cuda"})
        model = PeftModel.from_pretrained(base, str(lora)).merge_and_unload().eval()
        for ts, p in pairs.items():
            results[f"VLM/{ver}"][ts] = {
                o: round(score(model, p["nat"], p["tac"], questions[o]), 4) for o in OPTS}
        del model, base
        gc.collect(); torch.cuda.empty_cache()
        print(f"scored VLM/{ver}", flush=True)


def main():
    pairs = load_pairs()
    print(f"gpt_test pairs: {len(pairs)}", flush=True)
    order = sorted(pairs)

    models = ["CLIP", "VLM", "ResNet", "ViT", "DINOv2"]
    results = {f"{m}/{v}": {} for m in models for v in ("old", "new")}
    score_probe_family(pairs, results)
    score_vlm(pairs, results)

    print("\n===== FLAG ERRORS at t=0.50 (9 pairs x 5 options = 45 decisions) =====")
    summary = {}
    for m in models:
        line = {}
        for ver in ("old", "new"):
            fp = fn = correct = 0
            for ts in order:
                for o in OPTS:
                    pred = results[f"{m}/{ver}"][ts][o] > 0.5
                    gt = bool(pairs[ts]["labels"][o])
                    if pred == gt: correct += 1
                    elif pred: fp += 1
                    else: fn += 1
            line[ver] = {"correct": correct, "fp": fp, "fn": fn}
        summary[m] = line
        print(f"  {m:8s} old: {line['old']['correct']}/45 correct, FP={line['old']['fp']}, FN={line['old']['fn']}"
              f"   →  new: {line['new']['correct']}/45 correct, FP={line['new']['fp']}, FN={line['new']['fn']}")

    print("\n===== REGRESSION ROWS (prob_yes, old→new) =====")
    focus = [(ts, p) for ts, p in pairs.items() if p["object"] in ("camel", "duck", "horse")]
    for ts, p in focus:
        gt = {o: p["labels"][o] for o in OPTS if p["labels"][o]}
        print(f"\n  {p['object']}  (true defects: {gt or 'none'})")
        for o in OPTS:
            row = "  ".join(f"{m}:{results[f'{m}/old'][ts][o]:.2f}→{results[f'{m}/new'][ts][o]:.2f}"
                            for m in models)
            print(f"    {o:16s} {row}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "pairs": {ts: {"object": p["object"], "labels": p["labels"]} for ts, p in pairs.items()},
        "results": results, "summary": summary}, indent=2))
    print(f"\nsaved → {OUT}")


if __name__ == "__main__":
    main()
