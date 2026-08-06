#!/usr/bin/env python3
"""Derive per-option decision thresholds for the whole fleet on the VAL split.

WHY THIS EXISTS
---------------
The UI offers two threshold modes. Neither is trustworthy without this step:

  * "Balanced" uses a flat t=0.50 for every judge. That is wrong for the VLM,
    whose scores are a softmax over yes/no logits and sit near zero for rare
    options — it flags 3 of 183 test pairs on too_thick while its AUC there is
    0.685. The signal is present; 0.50 is simply the wrong operating point.

  * "Calibrated" reads per-option thresholds, but only the CLIP and DINOv2
    probes store them in their checkpoints. ResNet-50, ViT-B/16 and the VLM
    fall back to hardcoded dicts in evaluators/constants.py that were fitted
    against an earlier fleet. Running a new fleet on them is meaningless.

This scores VAL with the deployed fleet and picks, per judge per option, the
threshold maximising F1 — then prints paste-ready dicts for constants.py.

VAL, NOT TEST. A threshold is a fitted parameter (one per option per judge).
Choosing it to maximise F1 on the split you then report is selection bias, and
with 5 options x 5 judges that is 25 parameters tuned against the headline
number. Val exists for exactly this.

Usage (GPU — the VLM is 7B):
  sbatch research/experiments/run_derive_val_thresholds.sh
  python  research/experiments/derive_val_thresholds.py --split val
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

import argparse
import gc
import json

import torch

SPLITS = DATA_ROOT / "data" / "splits_v3_unified"
IMG_ROOT = DATA_ROOT / "images"
OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
GRID = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 .. 0.95

CONST_NAME = {
    "CLIP Probe":     "CLIP_THRESHOLDS_CALIBRATED",
    "DINOv2 Probe":   "DINO_THRESHOLDS_CALIBRATED",
    "ResNet-50":      "RESNET_THRESHOLDS_CALIBRATED",
    "ViT-B/16":       "VIT_THRESHOLDS_CALIBRATED",
    "VLM Fine-tuned": "VLM_THRESHOLDS_CALIBRATED",
}


def load_split(name):
    pairs = {}
    for line in (SPLITS / f"{name}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["option_id"] not in OPTS:
            continue
        p = pairs.setdefault(r["pair_id"], {
            "nat": str(IMG_ROOT / r["natural_image"]),
            "tac": str(IMG_ROOT / r["tactile_image"]), "labels": {}})
        p["labels"][r["option_id"]] = r["label"]
    return pairs


def f1_at(scores, labels, t):
    tp = sum(1 for s, y in zip(scores, labels) if s > t and y)
    fp = sum(1 for s, y in zip(scores, labels) if s > t and not y)
    fn = sum(1 for s, y in zip(scores, labels) if s <= t and y)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val",
                    help="split to fit thresholds on (val — never test)")
    ap.add_argument("--base-model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.split == "test":
        raise SystemExit("refusing to fit thresholds on test — that is the "
                         "reported split. Use --split val.")

    pairs = load_split(args.split)
    order = sorted(pairs)
    print(f"{args.split}: {len(pairs)} pairs, {len(pairs)*len(OPTS)} decisions\n")

    from evaluators.clip_probe_eval import CLIPProbeEvaluator
    from evaluators.dino_probe_eval import DINOProbeEvaluator
    from evaluators.backbone_eval import ResNetEvaluator, ViTEvaluator

    scores = {}
    makers = {
        "CLIP Probe":   lambda: CLIPProbeEvaluator(models_dir=REPO_ROOT / "models/clip_probe"),
        "DINOv2 Probe": lambda: DINOProbeEvaluator(models_dir=REPO_ROOT / "models/dino_probe"),
        "ResNet-50":    lambda: ResNetEvaluator(checkpoint=REPO_ROOT / "models/resnet50/resnet50_evaluator.pt"),
        "ViT-B/16":     lambda: ViTEvaluator(checkpoint=REPO_ROOT / "models/vit_b16/vit_b_16_evaluator.pt"),
    }
    for name, make in makers.items():
        ev = make()
        scores[name] = {pid: ev.score_pair(pairs[pid]["nat"], pairs[pid]["tac"]) for pid in order}
        del ev; gc.collect(); torch.cuda.empty_cache()
        print(f"scored {name}", flush=True)

    # VLM through the shipping question text
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel
    from evaluators.constants import QUESTIONS, SYSTEM_PROMPT

    proc = AutoProcessor.from_pretrained(args.base_model, max_pixels=200704)
    tok = proc.tokenizer
    yes = [tok.convert_tokens_to_ids(x) for x in ("yes", "Yes")]
    no = [tok.convert_tokens_to_ids(x) for x in ("no", "No")]
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": "cuda"})
    model = PeftModel.from_pretrained(base, str(REPO_ROOT / "models/vlm_checkpoint")).merge_and_unload().eval()

    @torch.no_grad()
    def vscore(nat, tac, q):
        msg = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": [{"type": "image", "image": f"file://{nat}"},
                                            {"type": "image", "image": f"file://{tac}"},
                                            {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msg)
        inp = proc(text=[text], images=imgs, return_tensors="pt", padding=True)
        inp = {k: v.to("cuda") for k, v in inp.items() if isinstance(v, torch.Tensor)}
        lg = model(**inp).logits[0, -1, :]
        y = max(lg[i].item() for i in yes); n = max(lg[i].item() for i in no)
        return torch.softmax(torch.tensor([y, n]), dim=0)[0].item()

    scores["VLM Fine-tuned"] = {
        pid: {o: vscore(pairs[pid]["nat"], pairs[pid]["tac"], QUESTIONS[o]) for o in OPTS}
        for pid in order}
    print("scored VLM Fine-tuned\n", flush=True)

    out = {}
    for m in CONST_NAME:
        print(f"=== {m} ===")
        print(f"  {'option':<17}{'best t':>8}{'F1@t':>8}{'F1@0.50':>10}{'flags@t':>9}{'pos':>6}")
        best = {}
        for o in OPTS:
            s = [scores[m][pid][o] for pid in order]
            y = [pairs[pid]["labels"][o] for pid in order]
            cand = max(GRID, key=lambda t: (f1_at(s, y, t), -abs(t - 0.5)))
            best[o] = cand
            print(f"  {o:<17}{cand:>8.2f}{f1_at(s,y,cand):>8.3f}{f1_at(s,y,0.5):>10.3f}"
                  f"{sum(1 for x in s if x>cand):>9}{sum(y):>6}")
        out[m] = best
        print()

    print("# paste into evaluators/constants.py")
    print(f"# fitted on the {args.split} split, fleet in TactileExpert/models/")
    for m, d in out.items():
        print(f"{CONST_NAME[m]} = {{")
        for o in OPTS:
            print(f'    "{o}":{" "*(16-len(o))}{d[o]:.2f},')
        print("}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"split": args.split, "thresholds": out}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
