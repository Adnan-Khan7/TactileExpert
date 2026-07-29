#!/usr/bin/env python3
"""Behavior cloning for the tactile edit-suggestion policy (Ch5 Loop 3).

LoRA fine-tune of Qwen2-VL so that, given (natural photo, current tactile
graphic, panel defect scores, target defect, edit history), the model writes the
edit instruction a human would have written.

PIPELINE VALIDATION, NOT AN H2 RESULT. 115 training records from 57
trajectories is far too thin to support a claim about policy quality; this
exists to prove the loop closes end to end. Report it as such.

Design — deliberately mirrors research/vlm/step9_finetune_vlm.py:
  - LoRA r=4, alpha=8 on q/k/v/o_proj, vision encoder FROZEN
  - bf16, gradient checkpointing, hand-rolled loop (the cluster venv has no trl)
  - Loss: cross-entropy on ASSISTANT tokens only
  - Early stopping on val loss (not F1 — the target is free text, not yes/no)

Differs from step9 in one way that matters: step9 supervises a single token
("yes"/"no"), so it can append answer ids to the prompt. Here the target is a
whole sentence, so the prompt/answer boundary is found by templating the
messages twice — once with the assistant turn, once without — and masking
everything up to the prompt length.

Run:
  python research/policy/train_bc_policy.py --dry-run          # CPU, no weights
  python research/policy/train_bc_policy.py --out-dir output_policy_bc
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT  # noqa: E402

import argparse
import json
import random
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
except ImportError:      # --dry-run must work on a login node with bare python
    np = torch = DataLoader = tqdm = None

SFT_DIR = DATA_ROOT / "experiments_out/policy_data/sft"
OUT_DIR = DATA_ROOT / "output_policy_bc"


def _require_deps():
    try:
        from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                                  get_cosine_schedule_with_warmup)
        from peft import LoraConfig, get_peft_model, TaskType
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        print(f"missing dependency: {e}\n"
              "Install: pip install 'transformers>=4.45' peft qwen-vl-utils accelerate")
        sys.exit(1)
    return (AutoProcessor, Qwen2VLForConditionalGeneration,
            get_cosine_schedule_with_warmup, LoraConfig, get_peft_model,
            TaskType, process_vision_info)


class PolicyDataset:
    """Plain sequence — torch's DataLoader only needs __len__ and __getitem__."""

    def __init__(self, jsonl_path: Path):
        self.records = [json.loads(l) for l in open(jsonl_path) if l.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {"messages": r["messages"], "instruction": r["instruction"]}


def make_collate_fn(processor, process_vision_info):
    """Batch and mask. Labels are -100 everywhere except the assistant turn."""
    if processor.tokenizer.padding_side != "right":
        raise ValueError(
            "padding_side must be 'right' — the answer span is located from the "
            "end of each sequence, which left padding would invalidate")

    def collate_fn(batch):
        full_texts, n_answers, all_images = [], [], []
        for item in batch:
            msgs = item["messages"]
            full = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False)
            prompt = processor.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True)
            full_texts.append(full)
            # Count the ANSWER tokens, not the prompt tokens.
            #
            # apply_chat_template emits ONE <|image_pad|> placeholder per image;
            # processor() later expands each into hundreds of tokens. So a prompt
            # length measured here does not index into the expanded sequence —
            # measuring it that way masks the wrong span entirely and silently
            # supervises the image and prompt tokens. The answer region holds no
            # image tokens, so its length is invariant under that expansion, and
            # the two unexpanded counts differ by exactly the answer.
            n_full = len(processor.tokenizer(full, add_special_tokens=False).input_ids)
            n_prompt = len(processor.tokenizer(prompt, add_special_tokens=False).input_ids)
            n_answers.append(n_full - n_prompt)
            imgs, _ = process_vision_info(msgs)
            all_images.append(imgs or [])

        flat_images = [img for imgs in all_images for img in imgs]
        inputs = processor(
            text=full_texts,
            images=flat_images if flat_images else None,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        for i, n_ans in enumerate(n_answers):
            real_len = int(inputs["attention_mask"][i].sum())
            if n_ans <= 0 or n_ans > real_len:
                raise ValueError(f"bad answer span: n_ans={n_ans} real_len={real_len}")
            labels[i, :real_len - n_ans] = -100
        inputs["labels"] = labels
        return inputs

    return collate_fn


def build_lora_model(model_path, device, Qwen2VLForConditionalGeneration,
                     LoraConfig, get_peft_model, TaskType, rank, alpha):
    print(f"Loading {model_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
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

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=None,
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    return model


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(**batch)
            n = int((batch["labels"] != -100).sum())
            total_loss += float(out.loss) * n
            total_tokens += n
    model.train()
    mean = total_loss / max(total_tokens, 1)
    return {"loss": round(mean, 4),
            "perplexity": round(float(np.exp(min(mean, 20))), 3),
            "supervised_tokens": total_tokens}


def dry_run(args):
    """Validate data, templating and masking without downloading weights."""
    for split in ("train", "val", "test"):
        p = Path(args.sft_dir or SFT_DIR) / f"{split}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing {p} — run research/policy/build_policy_dataset.py")
        ds = PolicyDataset(p)
        n_img = sum(
            1 for r in ds.records
            for c in r["messages"][1]["content"] if c["type"] == "image")
        missing = [c["image"] for r in ds.records
                   for c in r["messages"][1]["content"]
                   if c["type"] == "image" and not Path(c["image"]).exists()]
        roles = {tuple(m["role"] for m in r["messages"]) for r in ds.records}
        print(f"{split:5} n={len(ds):3}  images={n_img:3}  missing={len(missing)}  "
              f"roles={roles}")
        if missing:
            raise SystemExit(f"missing image files, e.g. {missing[:3]}")
        if roles != {("system", "user", "assistant")}:
            raise SystemExit(f"unexpected message roles in {split}: {roles}")
    print("\ndry run OK — data, roles and image paths all resolve.")
    print("Masking is validated against the real tokenizer only in a full run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct",
                    help="2B is the scaffold default — cheap on a throttled "
                         "allocation. Pass Qwen/Qwen2-VL-7B-Instruct to match "
                         "the deployed judge fleet.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8,
                    help="Gradient accumulation — effective batch = batch-size * accum")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=4)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed — batch order, dropout, LoRA init")
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28,
                    help="Must match the judge fleet's 200704 unless deliberately changed")
    ap.add_argument("--sft-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device",
                    default="cuda" if (torch and torch.cuda.is_available()) else "cpu")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate data and exit — no weights, no GPU")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    (AutoProcessor, Qwen2VLForConditionalGeneration, get_cosine_schedule_with_warmup,
     LoraConfig, get_peft_model, TaskType, process_vision_info) = _require_deps()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sft_dir = Path(args.sft_dir) if args.sft_dir else SFT_DIR
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"model: {args.model_path}  max_pixels: {args.max_pixels}  seed: {args.seed}")
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "right"

    collate = make_collate_fn(processor, process_vision_info)
    loaders = {}
    for split in ("train", "val", "test"):
        ds = PolicyDataset(sft_dir / f"{split}.jsonl")
        loaders[split] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=(split == "train"),
            collate_fn=collate)
        print(f"  {split}: {len(ds)} records")

    model = build_lora_model(args.model_path, device, Qwen2VLForConditionalGeneration,
                            LoraConfig, get_peft_model, TaskType,
                            args.lora_r, args.lora_alpha)

    steps_per_epoch = max(1, len(loaders["train"]) // args.accum)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=steps_per_epoch,
        num_training_steps=steps_per_epoch * args.epochs)

    best_loss, best_epoch, patience_count = float("inf"), 0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        optim.zero_grad()
        for i, batch in enumerate(tqdm(loaders["train"], desc=f"epoch {epoch}")):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            loss = model(**batch).loss / args.accum
            loss.backward()
            running += float(loss) * args.accum
            seen += 1
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()

        val = evaluate(model, loaders["val"], device)
        train_loss = round(running / max(seen, 1), 4)
        print(f"epoch {epoch}: train_loss={train_loss}  "
              f"val_loss={val['loss']}  val_ppl={val['perplexity']}")
        history.append({"epoch": epoch, "train_loss": train_loss, **val})

        if val["loss"] < best_loss:
            best_loss, best_epoch, patience_count = val["loss"], epoch, 0
            model.save_pretrained(out_dir / "best_checkpoint")
            processor.save_pretrained(out_dir / "best_checkpoint")
            print(f"  ↳ new best (val_loss {best_loss}) — checkpoint saved")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    test = evaluate(model, loaders["test"], device)
    summary = {
        "task": "behavior_cloning",
        "note": "PIPELINE VALIDATION — 115 train records, not an H2 result",
        "model_path": args.model_path,
        "seed": args.seed,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha},
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "test": test,
        "history": history,
    }
    (out_dir / "bc_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nbest epoch {best_epoch}  val_loss {best_loss}  test {test}")
    print(f"saved -> {out_dir}/")


if __name__ == "__main__":
    main()
