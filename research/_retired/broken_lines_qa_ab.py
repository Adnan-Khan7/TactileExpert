"""A/B test: broken_lines VQA question phrasings on the 15 held-out July 4-5 pairs.

Configs: {LoRA-finetuned, zero-shot base} x {A: current question, B: outline-only
long, C: outline-only short}. Scores P(yes) from yes/no token logits, same
mechanism as evaluators/vlm_eval.py. Ground truth = human_labels from
trajectories.jsonl (pairs saved on/after 2026-07-04 — held out from VLM training).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import json
from pathlib import Path

TRAIN = REPO_ROOT / "generated_training_data"
LORA = REPO_ROOT / "models/vlm_checkpoint"
OUT = DATA_ROOT / "experiments_out/broken_lines_qa_ab.json"
MODEL = "Qwen/Qwen2-VL-2B-Instruct"
MAX_PIXELS = 200704

SYSTEM_PROMPT = (
    "You are an expert evaluator of tactile graphics for blind and visually impaired learners. "
    "Tactile graphics are embossed line drawings that users explore through touch. "
    "You are shown two images: the first is a natural reference photograph and the second is "
    "a generated tactile graphic. "
    "Answer the quality assessment question with 'yes' or 'no' only."
)

VARIANTS = {
    "A_current": (
        "Does this tactile graphic contain broken or discontinuous outlines "
        "where a fingertip tracing the edges would encounter gaps, "
        "interruptions, or missing segments?"
    ),
    "B_outline_long": (
        "Look only at the object's structural outlines in this tactile graphic and "
        "ignore all texture or fill patterns inside enclosed regions, such as dots, "
        "dashes, or hatching — those fills are intentional design and are not broken "
        "lines. Are the structural outlines themselves broken or discontinuous, so "
        "that a fingertip tracing an edge would encounter gaps, interruptions, or "
        "missing segments?"
    ),
    "C_outline_short": (
        "Ignoring any texture fill patterns (dots, dashes, hatching) inside the "
        "object, does the object's outline contain gaps, breaks, or missing segments?"
    ),
}


def load_pairs():
    pairs = []
    with open(TRAIN / "trajectories.jsonl") as f:
        for line in f:
            t = json.loads(line)
            ts = t["trajectory_id"].replace("traj_", "")
            if ts < "20260704":
                continue
            pd = TRAIN / f"pair_{ts}"
            pairs.append({
                "id": ts,
                "object": t["object"],
                "nat": str(pd / "natural.jpg"),
                "tac": str(pd / "tactile.jpg"),
                "label": t["human_labels"]["broken_lines"],
            })
    return pairs


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    pairs = load_pairs()
    print(f"pairs: {len(pairs)}  positives: {sum(p['label'] for p in pairs)}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
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

    results = {}  # config -> {pair_id: prob}

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": "cuda"},
    ).eval()

    for model_name, get_model in [("base", lambda: base), ("lora", None)]:
        if model_name == "lora":
            from peft import PeftModel
            print("merging LoRA…", flush=True)
            model = PeftModel.from_pretrained(base, str(LORA)).merge_and_unload().eval()
        else:
            model = base
        for vname, q in VARIANTS.items():
            cfg = f"{model_name}/{vname}"
            results[cfg] = {}
            for p in pairs:
                results[cfg][p["id"]] = round(score(model, p["nat"], p["tac"], q), 4)
            print(f"done {cfg}", flush=True)

    print("\n===== RESULTS =====")
    hdr = f"{'object':14s} {'gt':>2s} " + " ".join(f"{c.split('/')[0][0]}:{c.split('_')[0][-1]:>6s}"
                                                   for c in results)
    print(f"{'object':14s} {'gt':>2s} " + " ".join(f"{c:>18s}" for c in results))
    for p in pairs:
        row = " ".join(f"{results[c][p['id']]:>18.3f}" for c in results)
        print(f"{p['object']:14s} {p['label']:>2d} {row}")

    print("\nmetrics (n=15, 4 pos):")
    labels = [p["label"] for p in pairs]
    summary = {}
    for cfg, sc in results.items():
        scores = [sc[p["id"]] for p in pairs]
        a = auc(scores, labels)
        acc = sum((s > 0.5) == bool(l) for s, l in zip(scores, labels)) / len(labels)
        fp = sum(1 for s, l in zip(scores, labels) if s > 0.5 and l == 0)
        fn = sum(1 for s, l in zip(scores, labels) if s <= 0.5 and l == 1)
        summary[cfg] = {"auc": round(a, 3), "acc@0.5": round(acc, 3), "fp": fp, "fn": fn}
        print(f"  {cfg:22s} AUC={a:.3f}  acc@0.5={acc:.3f}  FP={fp}/11  FN={fn}/4")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "pairs": pairs, "results": results, "summary": summary}, indent=2))
    print(f"\nsaved → {OUT}")


if __name__ == "__main__":
    main()
