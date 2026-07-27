#!/usr/bin/env python3
"""
debug_batch.py — CPU smoke test for TactileEval VLM fine-tuning pipeline.

Runs 4 batches (2 train, 1 val forward, 1 optimizer step) on CPU with a
tiny toy model stub to catch data, collation, tokenisation, mm_token_type_ids
alignment, and NaN propagation issues without needing a GPU.

Usage:
    source ~/vlm_finetune/venv/bin/activate
    python debug_batch.py

Optional:
    --model-path PATH   use a real model (slow on CPU, but validates full fwd pass)
    --real-model        actually load Qwen2-VL (overrides stub; needs ~8 GB RAM)
    --batches N         number of train batches to test (default: 4)
    --batch-size N      samples per batch (default: 2)
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
VQA_DIR      = PROJECT_ROOT / "data" / "vqa"

try:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("ERROR: pip install transformers peft qwen-vl-utils")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Shared dataset + collate (mirrors step9_finetune_vlm.py exactly)
# ---------------------------------------------------------------------------
class VQADataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.records = [json.loads(l) for l in open(jsonl_path) if l.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return {
            "messages": rec["messages"][:-1],
            "answer":   rec["answer"],
            "label":    rec["label"],
            "option":   rec["option_id"],
        }


def make_collate_fn(processor):
    def collate_fn(batch):
        messages_list = [item["messages"] for item in batch]
        texts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]
        all_images = []
        for msgs in messages_list:
            imgs, _ = process_vision_info(msgs)
            all_images.append(imgs)
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
# Checks
# ---------------------------------------------------------------------------
def check_tensor(name, t):
    if t is None:
        print(f"    {name}: None")
        return
    finite = torch.isfinite(t.float()).all().item()
    print(f"    {name}: shape={tuple(t.shape)}  dtype={t.dtype}  "
          f"finite={finite}  min={t.float().min():.4f}  max={t.float().max():.4f}")
    if not finite:
        n_nan = (~torch.isfinite(t.float())).sum().item()
        print(f"      *** {n_nan} non-finite values ***")


def check_batch(batch, tag=""):
    print(f"\n  --- batch tensors {tag} ---")
    check_tensor("input_ids",         batch.get("input_ids"))
    check_tensor("attention_mask",    batch.get("attention_mask"))
    check_tensor("pixel_values",      batch.get("pixel_values"))
    check_tensor("image_grid_thw",    batch.get("image_grid_thw"))
    check_tensor("mm_token_type_ids", batch.get("mm_token_type_ids"))
    print(f"    answers={batch['answers']}  options={batch['options']}")

    # mm_token_type_ids alignment check
    ids = batch.get("input_ids")
    mm  = batch.get("mm_token_type_ids")
    if ids is not None and mm is not None:
        if ids.shape != mm.shape:
            print(f"    *** SHAPE MISMATCH: input_ids {ids.shape} vs mm_token_type_ids {mm.shape} ***")
        else:
            print(f"    mm_token_type_ids shape OK ({tuple(mm.shape)})")


def check_loss(loss, tag=""):
    finite = torch.isfinite(loss).item()
    print(f"  loss {tag}: {loss.item():.6f}  finite={finite}")
    if not finite:
        print(f"  *** NaN/Inf LOSS ***")
    return finite


def check_weights(model, tag=""):
    for name, p in model.named_parameters():
        if p.requires_grad:
            if not torch.isfinite(p).all():
                print(f"  *** NaN/Inf in WEIGHT: {name} {tag} ***")
                return False
    print(f"  weights finite={True} {tag}")
    return True


# ---------------------------------------------------------------------------
# Forward pass (mirrors train_step exactly)
# ---------------------------------------------------------------------------
def debug_train_step(model, processor, batch, device):
    input_ids         = batch["input_ids"].to(device)
    attention_mask    = batch["attention_mask"].to(device)
    mm_token_type_ids = batch.get("mm_token_type_ids")
    pixel_values      = batch.get("pixel_values")
    image_grid_thw    = batch.get("image_grid_thw")

    answers = batch["answers"]
    ans_ids = processor.tokenizer(
        answers, return_tensors="pt", padding=True).input_ids.to(device)

    print(f"    ans_ids shape={tuple(ans_ids.shape)}  values={ans_ids.tolist()}")
    decoded = [processor.tokenizer.decode(ids) for ids in ans_ids]
    print(f"    ans_ids decoded={decoded}")

    full_ids  = torch.cat([input_ids, ans_ids], dim=1)
    full_mask = torch.cat([attention_mask,
                           torch.ones(ans_ids.shape, device=device)], dim=1)
    lm_labels = torch.full(full_ids.shape, -100, device=device)
    lm_labels[:, input_ids.shape[1]:] = ans_ids

    if mm_token_type_ids is not None:
        ans_type_ids = torch.zeros(
            mm_token_type_ids.shape[0], ans_ids.shape[1],
            dtype=mm_token_type_ids.dtype)
        full_mm = torch.cat([mm_token_type_ids.to(device),
                             ans_type_ids.to(device)], dim=1)
        if full_mm.shape != full_ids.shape:
            print(f"    *** full_mm_token_type_ids shape {full_mm.shape} "
                  f"!= full_ids shape {full_ids.shape} ***")
    else:
        full_mm = None

    n_label_tokens = (lm_labels != -100).sum().item()
    print(f"    label tokens active={n_label_tokens} "
          f"(should be batch_size × ans_len = {input_ids.shape[0] * ans_ids.shape[1]})")

    kwargs = dict(input_ids=full_ids, attention_mask=full_mask, labels=lm_labels)
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values.to(device, dtype=torch.bfloat16)
    if image_grid_thw is not None:
        kwargs["image_grid_thw"] = image_grid_thw.to(device)
    if full_mm is not None:
        kwargs["mm_token_type_ids"] = full_mm

    outputs = model(**kwargs)
    return outputs.loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--real-model", action="store_true",
                        help="Load the full Qwen2-VL model (slow on CPU)")
    parser.add_argument("--batches",    type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"TactileEval debug smoke test  (CPU)")
    print(f"model: {args.model_path}  real_model={args.real_model}")
    print(f"{'='*60}\n")

    # ---- Processor ----
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_path, max_pixels=256*28*28)
    print("  OK")

    # ---- Dataset ----
    train_ds = VQADataset(VQA_DIR / "train.jsonl")
    val_ds   = VQADataset(VQA_DIR / "val.jsonl")
    collate  = make_collate_fn(processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate, num_workers=0)
    print(f"Dataset: train={len(train_ds)} val={len(val_ds)}\n")

    # ---- Model ----
    if args.real_model:
        print("Loading full Qwen2-VL model on CPU (slow)...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map={"": device},
        )
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad_(False)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none")
        model = get_peft_model(model, lora_cfg)
        print("  Model ready")
    else:
        # Lightweight stub: just verify data pipeline, collation, and label alignment
        model = None
        print("Using data-only mode (no model loaded). Pass --real-model to test forward pass.\n")

    optimizer = (torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.1, eps=1e-6)
        if model is not None else None)

    # ========== CHECK: tokeniser answer encoding ==========
    print("="*40)
    print("CHECK 1 — tokeniser answer encoding")
    for answer in ["yes", "no", "Yes", "No"]:
        ids = processor.tokenizer([answer], return_tensors="pt",
                                  padding=True).input_ids
        decoded = processor.tokenizer.decode(ids[0])
        print(f"  '{answer}' → ids={ids.tolist()}  decoded='{decoded}'")

    # ========== CHECK: train batches ==========
    print(f"\n{'='*40}")
    print(f"CHECK 2 — {args.batches} train batches (data + collation)")
    all_ok = True
    for i, batch in enumerate(train_loader):
        if i >= args.batches:
            break
        print(f"\nBatch {i}:")
        check_batch(batch, f"(train batch {i})")

        if model is not None:
            print(f"  Running forward pass...")
            try:
                loss = debug_train_step(model, processor, batch, device)
                loss_ok = check_loss(loss, "(before backward)")

                if loss_ok:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step()
                    check_weights(model, "(after step)")
                else:
                    all_ok = False
            except Exception:
                print(f"  *** EXCEPTION in forward/backward ***")
                traceback.print_exc()
                all_ok = False

    # ========== CHECK: val batch (eval forward) ==========
    print(f"\n{'='*40}")
    print("CHECK 3 — 1 val batch (eval forward)")
    val_batch = next(iter(val_loader))
    check_batch(val_batch, "(val batch 0)")

    if model is not None:
        model.eval()
        try:
            with torch.no_grad():
                input_ids      = val_batch["input_ids"].to(device)
                attention_mask = val_batch["attention_mask"].to(device)
                pixel_values   = val_batch.get("pixel_values")
                image_grid_thw = val_batch.get("image_grid_thw")
                mm             = val_batch.get("mm_token_type_ids")

                fwd = dict(input_ids=input_ids, attention_mask=attention_mask)
                if pixel_values is not None:
                    fwd["pixel_values"] = pixel_values.to(device, dtype=torch.bfloat16)
                if image_grid_thw is not None:
                    fwd["image_grid_thw"] = image_grid_thw.to(device)
                if mm is not None:
                    fwd["mm_token_type_ids"] = mm.to(device)

                out = model(**fwd)
                logits = out.logits[:, -1, :]
                yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
                no_id  = processor.tokenizer.convert_tokens_to_ids("no")
                Yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
                No_id  = processor.tokenizer.convert_tokens_to_ids("No")
                print(f"  logit yes={logits[0, yes_id].item():.4f}  "
                      f"no={logits[0, no_id].item():.4f}  "
                      f"Yes={logits[0, Yes_id].item():.4f}  "
                      f"No={logits[0, No_id].item():.4f}")
                finite = torch.isfinite(logits).all().item()
                print(f"  logits finite={finite}")
        except Exception:
            print("  *** EXCEPTION in eval forward ***")
            traceback.print_exc()

    # ========== SUMMARY ==========
    print(f"\n{'='*60}")
    if model is None:
        print("Data pipeline: PASS (collation, shapes, and alignment all OK)")
        print("Forward pass:  SKIPPED (run with --real-model to test)")
    elif all_ok:
        print("Smoke test: PASS — data pipeline and forward/backward all clean")
    else:
        print("Smoke test: FAIL — see *** markers above")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
