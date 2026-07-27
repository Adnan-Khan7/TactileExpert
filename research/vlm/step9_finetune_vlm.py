#!/usr/bin/env python3
"""
Step 9: LoRA fine-tuning of Qwen2-VL-2B-Instruct for tactile quality evaluation.

Design:
  - Model:     Qwen2-VL-2B-Instruct (fits on 24GB with fp16, no quantisation)
  - LoRA:      r=4, alpha=8, vision encoder FROZEN — only language model layers
  - Loss:      cross-entropy on assistant tokens ("yes" / "no") only
  - Epochs:    max 8 (--epochs), early stopping on val macro-F1 (--patience)
  - Logging:   per-option F1 + AUC on val after each epoch

Run:
  python code/vlm/step9_finetune_vlm.py --vqa-dir data/vqa_v2 --out-dir output_v2

  Optional:
    --model-path  local path or HuggingFace model id (default: Qwen/Qwen2-VL-2B-Instruct)
    --epochs      max training epochs (default: 8)
    --batch-size  per-device batch size (default: 2)
    --lr          learning rate (default: 1e-4)
    --patience    early-stopping patience in epochs (default: 3)
    --seed        RNG seed — batch order, dropout, LoRA init (default: 42)
    --log         save output to log file
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = DATA_ROOT
VQA_DIR      = PROJECT_ROOT / "data" / "vqa"
OUT_DIR      = PROJECT_ROOT / "output"

try:
    from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                              get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Install: pip install transformers>=4.45 peft qwen-vl-utils accelerate")
    sys.exit(1)

try:
    from sklearn.metrics import f1_score, roc_auc_score
except ImportError:
    raise ImportError("pip install scikit-learn")


# ---------------------------------------------------------------------------
# Dataset — returns raw data only; processor runs at batch level in collate
# ---------------------------------------------------------------------------
class VQADataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.records = [json.loads(l) for l in open(jsonl_path) if l.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return {
            "messages": rec["messages"][:-1],  # system + user only
            "answer":   rec["answer"],
            "label":    rec["label"],
            "option":   rec["option_id"],
        }


def make_collate_fn(processor):
    """Returns a collate_fn that batches and pads at the batch level."""
    def collate_fn(batch):
        messages_list = [item["messages"] for item in batch]

        # Build text prompts with generation prompt
        texts = [
            processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]

        # Extract image inputs per sample
        all_images = []
        for msgs in messages_list:
            imgs, _ = process_vision_info(msgs)
            all_images.append(imgs)
        # Flatten image list for processor
        flat_images = [img for imgs in all_images for img in (imgs or [])]

        inputs = processor(
            text=texts,
            images=flat_images if flat_images else None,
            padding=True,
            return_tensors="pt",
        )

        inputs["answers"]    = [item["answer"] for item in batch]
        inputs["labels_int"] = torch.tensor([item["label"] for item in batch])
        inputs["options"]    = [item["option"] for item in batch]
        return inputs

    return collate_fn


# ---------------------------------------------------------------------------
# LoRA config — vision encoder excluded explicitly
# ---------------------------------------------------------------------------
def build_lora_model(model_path: str, device):
    print(f"Loading {model_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": device},
    )

    # Freeze vision encoder — only fine-tune language model attention layers
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad_(False)

    # Gradient checkpointing before LoRA wrap — cuts activation memory ~40%
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.3f}%)")
    return model


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def train_step(model, processor, batch, device):
    """
    Forward pass with loss computed only on the answer tokens.
    The model generates "yes" or "no" — we optimise only those tokens.
    """
    input_ids           = batch["input_ids"].to(device)
    attention_mask      = batch["attention_mask"].to(device)
    mm_token_type_ids   = batch.get("mm_token_type_ids")
    pixel_values        = batch.get("pixel_values")
    image_grid_thw      = batch.get("image_grid_thw")

    # Tokenise target answers and append to input
    answers  = batch["answers"]   # list of "yes" / "no"
    ans_ids  = processor.tokenizer(
        answers, return_tensors="pt", padding=True).input_ids.to(device)

    # Build full sequence: input + answer
    full_ids  = torch.cat([input_ids, ans_ids], dim=1)
    full_mask = torch.cat([
        attention_mask,
        torch.ones(ans_ids.shape, device=device)], dim=1)

    # Labels: -100 for input tokens, actual ids only for answer tokens
    lm_labels = torch.full(full_ids.shape, -100, device=device)
    lm_labels[:, input_ids.shape[1]:] = ans_ids

    # Extend mm_token_type_ids with zeros (text type) for the appended answer tokens
    if mm_token_type_ids is not None:
        ans_type_ids = torch.zeros(
            mm_token_type_ids.shape[0], ans_ids.shape[1],
            dtype=mm_token_type_ids.dtype)
        full_mm_token_type_ids = torch.cat(
            [mm_token_type_ids.to(device), ans_type_ids.to(device)], dim=1)
    else:
        full_mm_token_type_ids = None

    kwargs = dict(
        input_ids=full_ids,
        attention_mask=full_mask,
        labels=lm_labels,
    )
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values.to(device, dtype=torch.bfloat16)
    if image_grid_thw is not None:
        kwargs["image_grid_thw"] = image_grid_thw.to(device)
    if full_mm_token_type_ids is not None:
        kwargs["mm_token_type_ids"] = full_mm_token_type_ids

    outputs = model(**kwargs)
    return outputs.loss


# ---------------------------------------------------------------------------
# Evaluation — generate "yes"/"no" and compare to labels
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, processor, loader, device, thresholds: dict | None = None) -> dict:
    """
    Collect yes-probabilities for every sample, then score per option.

    If thresholds is None (val pass): sweep [0.1, 0.9] to find the threshold
    that maximises per-option F1 — mirrors the CLIP evaluator's calibration.
    If thresholds is provided (test pass): use those fixed values.

    Returns dict with keys: per_option, macro_f1, thresholds.
    """
    model.eval()
    per_opt: dict[str, dict] = defaultdict(lambda: {"probs": [], "labels": []})

    yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
    no_id  = processor.tokenizer.convert_tokens_to_ids("no")
    Yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
    No_id  = processor.tokenizer.convert_tokens_to_ids("No")

    for batch in tqdm(loader, desc="  evaluating", leave=False):
        input_ids         = batch["input_ids"].to(device)
        attention_mask    = batch["attention_mask"].to(device)
        mm_token_type_ids = batch.get("mm_token_type_ids")
        pixel_values      = batch.get("pixel_values")
        image_grid_thw    = batch.get("image_grid_thw")

        fwd_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if pixel_values is not None:
            fwd_kwargs["pixel_values"] = pixel_values.to(device, dtype=torch.bfloat16)
        if image_grid_thw is not None:
            fwd_kwargs["image_grid_thw"] = image_grid_thw.to(device)
        if mm_token_type_ids is not None:
            fwd_kwargs["mm_token_type_ids"] = mm_token_type_ids.to(device)

        outputs = model(**fwd_kwargs)
        logits  = outputs.logits[:, -1, :]

        yes_logit = torch.max(logits[:, yes_id], logits[:, Yes_id])
        no_logit  = torch.max(logits[:, no_id],  logits[:, No_id])
        prob_yes  = torch.softmax(
            torch.stack([yes_logit, no_logit], dim=-1), dim=-1)[:, 0]

        for i, opt in enumerate(batch["options"]):
            per_opt[opt]["probs"].append(float(prob_yes[i].cpu()))
            per_opt[opt]["labels"].append(int(batch["labels_int"][i]))

    results = {"per_option": {}, "thresholds": {}}
    f1_scores = []

    for opt, data in per_opt.items():
        probs  = np.array(data["probs"])
        labels = np.array(data["labels"])

        if thresholds is not None:
            best_thresh = thresholds.get(opt, 0.5)
        else:
            # Calibrate: sweep thresholds, pick the one that maximises F1
            best_f1_t, best_thresh = 0.0, 0.5
            for t in np.arange(0.05, 0.95, 0.05):
                preds_t = (probs > t).astype(int)
                f1_t = f1_score(labels, preds_t, zero_division=0)
                if f1_t > best_f1_t:
                    best_f1_t, best_thresh = f1_t, float(round(t, 2))

        preds = (probs > best_thresh).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        except Exception:
            auc = 0.0

        results["per_option"][opt] = {
            "f1": round(f1, 4), "auc": round(auc, 4),
            "threshold": round(best_thresh, 2),
            "n_pos": int(labels.sum()), "n_total": int(len(labels)),
        }
        results["thresholds"][opt] = round(best_thresh, 2)
        f1_scores.append(f1)

    results["macro_f1"] = round(float(np.mean(f1_scores)) if f1_scores else 0.0, 4)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--epochs",     type=int,   default=8)
    parser.add_argument("--batch-size", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=3)
    parser.add_argument("--seed",       type=int,   default=42,
                        help="RNG seed — batch order, dropout, LoRA init")
    parser.add_argument("--vqa-dir",    type=str,   default=None,
                        help="Path to VQA data directory (default: data/vqa)")
    parser.add_argument("--out-dir",    type=str,   default=None,
                        help="Output directory for checkpoints (default: output)")
    parser.add_argument("--max-pixels", type=int,   default=256*28*28,
                        help="Max image pixels fed to the vision encoder (default: 200704 ≈ 256 tokens/img)")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log",        action="store_true")
    args = parser.parse_args()

    global VQA_DIR, OUT_DIR
    if args.vqa_dir:
        VQA_DIR = Path(args.vqa_dir)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.log:
        log_file = open(OUT_DIR / "train_log.txt", "w")
        class _Tee:
            def __init__(self, *s): self.streams = s
            def write(self, d): [s.write(d) for s in self.streams]
            def flush(self):   [s.flush()   for s in self.streams]
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)

    # Seed everything BEFORE model build (LoRA init) and DataLoader creation
    # (shuffle order) so runs are reproducible and retries can vary the seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"LoRA VLM fine-tuning: {args.model_path}")
    print(f"LoRA: r=4  alpha=8  attn=eager  vision_encoder=FROZEN  dtype=bfloat16  adam_eps=1e-6")
    print(f"Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}  "
          f"max_pixels: {args.max_pixels}  seed: {args.seed}")
    print(f"{'='*60}\n")

    # Processor — cap image resolution to control visual token count
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)

    # Datasets
    train_ds = VQADataset(VQA_DIR / "train.jsonl")
    val_ds   = VQADataset(VQA_DIR / "val.jsonl")
    test_ds  = VQADataset(VQA_DIR / "test.jsonl")

    collate_fn = make_collate_fn(processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)

    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}\n")

    # Model
    model = build_lora_model(args.model_path, device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.1, eps=1e-6)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps)

    best_f1, best_epoch, patience_count = -1.0, 0, 0
    best_thresholds: dict = {}
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, skipped = 0.0, 0
        _first_nan_logged = False
        _first_weight_nan_logged = False
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            # --- DIAGNOSTIC: check inputs before forward ---
            pv = batch.get("pixel_values")
            pv_nan = pv is not None and not torch.isfinite(pv).all()
            ids_nan = not torch.isfinite(batch["input_ids"].float()).all()

            loss = train_step(model, processor, batch, device)

            if not torch.isfinite(loss):
                if not _first_nan_logged:
                    print(f"\n  [DIAG] First NaN loss at batch {batch_idx}/{len(train_loader)}"
                          f"  pv_nan={pv_nan}  ids_nan={ids_nan}"
                          f"  options={batch['options']}  answers={batch['answers']}")
                    _first_nan_logged = True
                skipped += 1
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            # --- DIAGNOSTIC: check if optimizer step produced NaN weights ---
            if not _first_weight_nan_logged:
                sample_p = next(p for p in model.parameters() if p.requires_grad)
                if not torch.isfinite(sample_p).all():
                    print(f"\n  [DIAG] Weights went NaN after batch {batch_idx}"
                          f"  (loss was {loss.item():.4f})")
                    _first_weight_nan_logged = True

        n_batches = len(train_loader) - skipped
        val_res = evaluate(model, processor, val_loader, device)
        history.append({"epoch": epoch,
                         "train_loss": round(epoch_loss / max(n_batches, 1), 5),
                         "skipped_batches": skipped,
                         "val_macro_f1": val_res["macro_f1"]})

        print(f"\nEpoch {epoch}  train_loss={epoch_loss/max(n_batches,1):.4f}  "
              f"val macro_F1={val_res['macro_f1']:.4f}"
              + (f"  [skipped {skipped} NaN batches]" if skipped else ""))
        for opt, m in val_res["per_option"].items():
            print(f"  {opt:<40} F1={m['f1']:.3f}  AUC={m['auc']:.3f}  "
                  f"thresh={m['threshold']:.2f}  (pos={m['n_pos']}/{m['n_total']})")

        if val_res["macro_f1"] > best_f1:
            best_f1        = val_res["macro_f1"]
            best_epoch     = epoch
            best_thresholds = val_res["thresholds"]   # calibrated on this val pass
            patience_count = 0
            model.save_pretrained(OUT_DIR / "best_checkpoint")
            processor.save_pretrained(OUT_DIR / "best_checkpoint")
            print(f"  [saved best checkpoint]")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Load best checkpoint and evaluate on test using val-calibrated thresholds
    from peft import PeftModel
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": device})
    model = PeftModel.from_pretrained(base_model, OUT_DIR / "best_checkpoint")

    test_res = evaluate(model, processor, test_loader, device, thresholds=best_thresholds)

    print(f"\n{'='*60}  TEST RESULTS")
    print(f"  best epoch: {best_epoch}   val macro_F1: {best_f1:.4f}")
    print(f"  val thresholds: {best_thresholds}")
    print(f"  test macro_F1: {test_res['macro_f1']:.4f}")
    for opt, m in test_res["per_option"].items():
        print(f"  {opt:<40} F1={m['f1']:.3f}  AUC={m['auc']:.3f}  thresh={m['threshold']:.2f}")

    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump(test_res, f, indent=2)
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
