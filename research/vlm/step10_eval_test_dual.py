#!/usr/bin/env python3
"""Evaluate the best VLM checkpoint on the test split with BOTH threshold modes.

Produces the numbers missing from the June 11 report: balanced (t = 0.50)
test metrics alongside the existing val-calibrated ones. Also saves
per-record yes-probabilities so any future threshold analysis can be done
offline without re-running inference.

Outputs:
    output/test_probs.jsonl        per-record {id, option_id, label, prob_yes}
    output/test_metrics_dual.json  balanced + calibrated metrics per option

Usage (GPU node):
    python code/vlm/step10_eval_test_dual.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = DATA_ROOT
VQA_DIR      = PROJECT_ROOT / "data" / "vqa"
OUT_DIR      = PROJECT_ROOT / "output"

BALANCED_T = 0.50


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs > threshold).astype(int)
    try:
        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    except ValueError:
        auc = 0.0
    return {
        "f1":        round(f1_score(labels, preds, zero_division=0), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, zero_division=0), 4),
        "auc":       round(float(auc), 4),
        "threshold": threshold,
        "n_pos":     int(labels.sum()),
        "n_total":   int(len(labels)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--checkpoint", default=str(OUT_DIR / "best_checkpoint"))
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    device = torch.device(args.device)
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading {args.model_path}  device={device}  dtype={dtype}")
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, attn_implementation="eager",
        device_map={"": device},
    )

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}")
    from peft import PeftModel
    print(f"Merging LoRA adapter from {ckpt}")
    model = PeftModel.from_pretrained(model, str(ckpt))
    model = model.merge_and_unload()
    model.eval()

    tok = processor.tokenizer
    yes_ids = [tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("Yes")]
    no_ids  = [tok.convert_tokens_to_ids("no"),  tok.convert_tokens_to_ids("No")]

    records = [json.loads(l) for l in (VQA_DIR / "test.jsonl").open()]
    print(f"Test records: {len(records)}")

    per_opt: dict[str, dict] = defaultdict(lambda: {"probs": [], "labels": []})
    prob_rows = []

    with torch.no_grad():
        for rec in tqdm(records, desc="evaluating"):
            messages = rec["messages"]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            images, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=images,
                               return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}

            logits    = model(**inputs).logits[0, -1, :]
            yes_logit = max(logits[yes_ids[0]].item(), logits[yes_ids[1]].item())
            no_logit  = max(logits[no_ids[0]].item(),  logits[no_ids[1]].item())
            prob_yes  = torch.softmax(
                torch.tensor([yes_logit, no_logit]), dim=0)[0].item()

            opt = rec["option_id"]
            per_opt[opt]["probs"].append(prob_yes)
            per_opt[opt]["labels"].append(int(rec["label"]))
            prob_rows.append({
                "id": rec["id"], "option_id": opt,
                "label": int(rec["label"]), "prob_yes": round(prob_yes, 6),
            })

    with (OUT_DIR / "test_probs.jsonl").open("w") as f:
        for row in prob_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved per-record probs -> {OUT_DIR / 'test_probs.jsonl'}")

    # Calibrated thresholds from the June 11 test run
    calib = json.loads((OUT_DIR / "test_metrics.json").read_text())["thresholds"]

    results = {"per_option": {}, "macro_f1_balanced": 0.0, "macro_f1_calibrated": 0.0}
    f1_bal, f1_cal = [], []
    for opt, data in per_opt.items():
        probs  = np.array(data["probs"])
        labels = np.array(data["labels"])
        m_bal = compute_metrics(probs, labels, BALANCED_T)
        m_cal = compute_metrics(probs, labels, calib.get(opt, 0.5))
        results["per_option"][opt] = {"balanced": m_bal, "calibrated": m_cal}
        f1_bal.append(m_bal["f1"])
        f1_cal.append(m_cal["f1"])

    results["macro_f1_balanced"]   = round(float(np.mean(f1_bal)), 4)
    results["macro_f1_calibrated"] = round(float(np.mean(f1_cal)), 4)

    with (OUT_DIR / "test_metrics_dual.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved dual metrics -> {OUT_DIR / 'test_metrics_dual.json'}")

    print(f"\n{'Option':<20} {'F1 bal':>8} {'F1 cal':>8} {'AUC':>7} {'t_cal':>6}")
    for opt, m in results["per_option"].items():
        print(f"{opt:<20} {m['balanced']['f1']:>8.3f} {m['calibrated']['f1']:>8.3f} "
              f"{m['balanced']['auc']:>7.3f} {m['calibrated']['threshold']:>6.2f}")
    print(f"{'MACRO':<20} {results['macro_f1_balanced']:>8.4f} "
          f"{results['macro_f1_calibrated']:>8.4f}")


if __name__ == "__main__":
    main()
