"""Zero-shot baselines on the GPT-domain pairs (multi-judge panel).

Judges (no task training whatsoever):
  - Qwen2-VL-2B-Instruct  (yes/no logit scoring; base of our fine-tune)
  - Qwen2-VL-7B-Instruct  (same protocol, 3.5x scale)
  - CLIP ViT-L/14 laion2b (contrastive defect-vs-clean prompt pairs; single
    image — sees only the tactile, no reference comparison)

Evaluated on all 48 trajectory-collection finals (240 decisions; zero-shot
models are untainted by training data) with the frozen 9-pair holdout also
reported separately for comparison against fine-tuned models. DINOv2 is
excluded by design: no text interface, so no zero-shot classification exists —
it only becomes a judge with a trained head (our probe).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import gc
import json
from pathlib import Path

HOME = Path.home()
TRAIN_SRC = REPO_ROOT / "generated_training_data"
OUT = DATA_ROOT / "experiments_out/zero_shot_baselines.json"

OPTS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]

SYSTEM_PROMPT = (
    "You are an expert evaluator of tactile graphics for blind and visually impaired learners. "
    "Tactile graphics are embossed line drawings that users explore through touch. "
    "You are shown two images: the first is a natural reference photograph and the second is "
    "a generated tactile graphic. "
    "Answer the quality assessment question with 'yes' or 'no' only."
)

QUESTIONS = {
    "too_thick": (
        "Do the outlines in this tactile graphic appear excessively thick, "
        "causing adjacent parts to merge together or fine structural details "
        "to collapse into solid filled areas rather than distinct lines?"
    ),
    "broken_lines": (
        "Consider only the structural outlines of the object, not any texture "
        "or fill patterns inside its regions. Are the outlines broken or "
        "discontinuous, so that a fingertip tracing an edge would encounter "
        "gaps, interruptions, or missing segments?"
    ),
    "missing_parts": (
        "Comparing this tactile graphic to the reference photograph, does the "
        "tactile drawing omit any clearly visible structural parts, anatomical "
        "features, or key functional components of the subject?"
    ),
    "missing_texture": (
        "Does this tactile graphic omit tactile texture fills that are needed "
        "to distinguish different surfaces, materials, or regions of the object?"
    ),
    "extra_parts": (
        "Comparing this tactile graphic to the reference photograph, does the "
        "tactile drawing contain invented structures, extra elements, or "
        "decorative additions that are not present in the reference?"
    ),
}

# CLIP contrastive prompt pairs (defect vs clean), single tactile image
CLIP_PROMPTS = {
    "too_thick": ("a tactile line drawing with excessively thick strokes that merge adjacent parts together",
                  "a tactile line drawing with clean, thin, well-separated strokes"),
    "broken_lines": ("a tactile line drawing with broken, discontinuous outlines full of gaps",
                     "a tactile line drawing with continuous, unbroken outlines"),
    "missing_parts": ("an incomplete tactile line drawing with missing parts of the object",
                      "a complete tactile line drawing showing every part of the object"),
    "missing_texture": ("a plain tactile line drawing with no texture fills inside its regions",
                        "a tactile line drawing with distinct texture fills differentiating its regions"),
    "extra_parts": ("a cluttered tactile line drawing with extra invented elements and stray marks",
                    "a clean tactile line drawing showing only the object itself"),
}


def load_pairs():
    trajs = [json.loads(l) for l in (TRAIN_SRC / "trajectories.jsonl").read_text().splitlines()]
    new = sorted((t for t in trajs if t["trajectory_id"].replace("traj_", "") >= "20260704"),
                 key=lambda t: t["trajectory_id"])
    holdout = set(range(4, len(new), 5))  # identical rule to build_gpt_v3_merge.py
    pairs = []
    for i, t in enumerate(new):
        ts = t["trajectory_id"].replace("traj_", "")
        pd = TRAIN_SRC / f"pair_{ts}"
        pairs.append({
            "id": ts, "object": t["object"], "holdout": i in holdout,
            "nat": str(pd / "natural.jpg"), "tac": str(pd / "tactile.jpg"),
            "labels": {o: t["human_labels"][o] for o in OPTS},
        })
    return pairs


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def f1(preds, labels):
    tp = sum(p and l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum((not p) and l for p, l in zip(preds, labels))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def score_qwen(model_id, pairs, results):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(model_id, max_pixels=200704)
    tok = processor.tokenizer
    yes_ids = [tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("Yes")]
    no_ids = [tok.convert_tokens_to_ids("no"), tok.convert_tokens_to_ids("No")]
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": "cuda"}).eval()

    @torch.no_grad()
    def score(nat, tac, question):
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

    name = model_id.split("/")[-1]
    results[name] = {}
    for p in pairs:
        results[name][p["id"]] = {o: round(score(p["nat"], p["tac"], QUESTIONS[o]), 4)
                                  for o in OPTS}
    del model
    gc.collect(); torch.cuda.empty_cache()
    print(f"scored {name}", flush=True)


def score_clip_zs(pairs, results):
    import torch
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k")
    model = model.to("cuda").eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    @torch.no_grad()
    def score(tac, opt):
        img = preprocess(Image.open(tac).convert("RGB")).unsqueeze(0).to("cuda")
        texts = tokenizer(list(CLIP_PROMPTS[opt])).to("cuda")
        img_f = model.encode_image(img)
        txt_f = model.encode_text(texts)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        sims = (100.0 * img_f @ txt_f.T).softmax(dim=-1)[0]
        return sims[0].item()  # P(defect prompt)

    results["CLIP-ZS"] = {}
    for p in pairs:
        results["CLIP-ZS"][p["id"]] = {o: round(score(p["tac"], o), 4) for o in OPTS}
    del model
    gc.collect(); torch.cuda.empty_cache()
    print("scored CLIP-ZS", flush=True)


def report(pairs, results):
    subsets = [("ALL 48 pairs", pairs), ("9-pair holdout", [p for p in pairs if p["holdout"]])]
    summary = {}
    for sub_name, sub in subsets:
        print(f"\n===== {sub_name} =====")
        for name, sc in results.items():
            per_opt = {}
            correct = fp_t = fn_t = 0
            for o in OPTS:
                scores = [sc[p["id"]][o] for p in sub]
                labels = [p["labels"][o] for p in sub]
                preds = [s > 0.5 for s in scores]
                correct += sum(pr == bool(l) for pr, l in zip(preds, labels))
                fp_t += sum(pr and not l for pr, l in zip(preds, labels))
                fn_t += sum((not pr) and l for pr, l in zip(preds, labels))
                per_opt[o] = {"f1": round(f1(preds, labels), 3),
                              "auc": round(auc(scores, labels), 3)}
            n = len(sub) * len(OPTS)
            print(f"\n  {name}: {correct}/{n} correct  FP={fp_t}  FN={fn_t}")
            for o in OPTS:
                print(f"    {o:16s} F1={per_opt[o]['f1']}  AUC={per_opt[o]['auc']}")
            summary.setdefault(name, {})[sub_name] = {
                "correct": correct, "n": n, "fp": fp_t, "fn": fn_t, "per_option": per_opt}
    return summary


def main():
    pairs = load_pairs()
    n_pos = {o: sum(p["labels"][o] for p in pairs) for o in OPTS}
    print(f"pairs: {len(pairs)} (holdout {sum(p['holdout'] for p in pairs)})  positives: {n_pos}",
          flush=True)

    results = {}
    score_clip_zs(pairs, results)
    score_qwen("Qwen/Qwen2-VL-2B-Instruct", pairs, results)
    score_qwen("Qwen/Qwen2-VL-7B-Instruct", pairs, results)

    summary = report(pairs, results)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "pairs": {p["id"]: {"object": p["object"], "holdout": p["holdout"],
                            "labels": p["labels"]} for p in pairs},
        "results": results, "summary": summary}, indent=2))
    print(f"\nsaved → {OUT}")


if __name__ == "__main__":
    main()
