#!/usr/bin/env python3
"""DPO on logged edit-instruction preferences (Ch5 Loop 3).

Every time the annotator rewrote the instruction the app had pre-filled, that
was a preference: chosen = what the human typed, rejected = what the ensemble
suggested. 57 such pairs were logged (44 train / 8 val / 5 test after the
trajectory-grouped split).

PIPELINE VALIDATION, NOT AN H2 RESULT. 44 training pairs cannot support a claim
about preference learning; this proves the loop closes. Report it as such.

REWARD FILTER — decided 2026-07-29: all 57 pairs are kept.
19 of 57 have negative calibrated reward on the CHOSEN instruction (the notes
previously said 26; that was wrong). Exactly one pair flips sign between raw and
calibrated reward, so the choice of reward is immaterial here. DPO learns the
PREFERENCE between chosen and rejected, which does not require the panel's
defect mass to have fallen — and filtering would cut 44 training pairs to ~29.
`--min-reward` exists so the alternative can be run, but it is off by default.

No trl on the cluster venv, so the DPO objective is written out directly:

    loss = -log σ( β · [ (logπ(y_w|x) − logπ_ref(y_w|x))
                        − (logπ(y_l|x) − logπ_ref(y_l|x)) ] )

The reference policy is the same weights with the LoRA adapter disabled, so
only one model is ever resident.

Run:
  python research/policy/train_dpo_policy.py --dry-run
  python research/policy/train_dpo_policy.py --init-from output_policy_bc/best_checkpoint
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
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm
except ImportError:      # --dry-run must work on a login node with bare python
    np = torch = F = DataLoader = tqdm = None

PAIR_DIR = DATA_ROOT / "experiments_out/policy_data/dpo_pairs"
OUT_DIR = DATA_ROOT / "output_policy_dpo"


def _require_deps():
    try:
        from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                                  get_cosine_schedule_with_warmup)
        from peft import LoraConfig, PeftModel, get_peft_model, TaskType
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        print(f"missing dependency: {e}\n"
              "Install: pip install 'transformers>=4.45' peft qwen-vl-utils accelerate")
        sys.exit(1)
    return (AutoProcessor, Qwen2VLForConditionalGeneration,
            get_cosine_schedule_with_warmup, LoraConfig, PeftModel,
            get_peft_model, TaskType, process_vision_info)


class PreferenceDataset:
    """Plain sequence — torch's DataLoader only needs __len__ and __getitem__."""

    def __init__(self, jsonl_path: Path, min_reward=None):
        rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
        if min_reward is not None:
            rows = [r for r in rows
                    if r.get("reward_calibrated") is not None
                    and r["reward_calibrated"] > min_reward]
        self.records = rows

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {"messages": r["messages"], "chosen": r["chosen"],
                "rejected": r["rejected"]}


def make_collate_fn(processor, process_vision_info):
    """Build chosen and rejected sequences from one shared prompt.

    Both are emitted in a single batch of size 2N (chosen first, then rejected)
    so they pass through the vision tower under identical padding.
    """
    if processor.tokenizer.padding_side != "right":
        raise ValueError(
            "padding_side must be 'right' — the answer span is located from the "
            "end of each sequence, which left padding would invalidate")

    def encode(msgs, answer):
        """Return the full templated text and its ANSWER token count.

        Counting answer tokens rather than prompt tokens is load-bearing: the
        processor expands each <|image_pad|> placeholder into hundreds of tokens,
        so a prompt length measured on the templated text does not index into the
        expanded sequence. The answer carries no image tokens, so the difference
        of the two unexpanded counts is exactly the answer length.
        """
        full = processor.apply_chat_template(
            msgs + [{"role": "assistant", "content": answer}],
            tokenize=False, add_generation_prompt=False)
        prompt = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        n_full = len(processor.tokenizer(full, add_special_tokens=False).input_ids)
        n_prompt = len(processor.tokenizer(prompt, add_special_tokens=False).input_ids)
        return full, n_full - n_prompt

    def collate_fn(batch):
        texts, n_answers, all_images = [], [], []
        for key in ("chosen", "rejected"):
            for item in batch:
                full, n_ans = encode(item["messages"], item[key])
                texts.append(full)
                n_answers.append(n_ans)
                imgs, _ = process_vision_info(item["messages"])
                all_images.append(imgs or [])

        flat_images = [img for imgs in all_images for img in imgs]
        inputs = processor(
            text=texts,
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
        inputs["n_pairs"] = len(batch)
        return inputs

    return collate_fn


def sequence_logprobs(model, batch):
    """Sum of token log-probabilities over the supervised (answer) tokens."""
    model_inputs = {k: v for k, v in batch.items()
                    if k not in ("labels", "n_pairs")}
    logits = model(**model_inputs).logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    safe = labels.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(logits.float(), dim=-1).gather(
        2, safe.unsqueeze(2)).squeeze(2)
    return (token_logps * mask).sum(dim=1)


def dpo_loss(policy_logps, ref_logps, n_pairs, beta):
    pol_chosen, pol_rejected = policy_logps[:n_pairs], policy_logps[n_pairs:]
    ref_chosen, ref_rejected = ref_logps[:n_pairs], ref_logps[n_pairs:]
    logits = (pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)
    loss = -F.logsigmoid(beta * logits).mean()
    # Share of pairs the policy ranks correctly. NOTE the degenerate init: before
    # any update the adapter is a no-op, so policy == reference, every margin is
    # exactly 0.0 and this reads 0.00 — NOT 0.5, and not a bug. The baseline line
    # printed before epoch 1 should show loss=0.6931 (= -log 0.5), margin=0.0,
    # accuracy=0.0; anything else there means the reference pass is wrong.
    acc = (logits > 0).float().mean()
    return loss, acc, logits


def evaluate(model, loader, device, beta):
    model.eval()
    losses, accs, margins = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            n = batch["n_pairs"]
            policy = sequence_logprobs(model, batch)
            with model.disable_adapter():
                ref = sequence_logprobs(model, batch)
            loss, acc, logits = dpo_loss(policy, ref, n, beta)
            losses.append(float(loss))
            accs.append(float(acc))
            margins.append(float(logits.mean()))
    model.train()
    return {"loss": round(float(np.mean(losses)), 4),
            "pref_accuracy": round(float(np.mean(accs)), 4),
            "mean_margin": round(float(np.mean(margins)), 4)}


def dry_run(args):
    pair_dir = Path(args.pair_dir) if args.pair_dir else PAIR_DIR
    for split in ("train", "val", "test"):
        p = pair_dir / f"{split}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing {p} — run research/policy/build_policy_dataset.py")
        ds = PreferenceDataset(p, args.min_reward)
        identical = sum(1 for r in ds.records
                        if r["chosen"].strip() == r["rejected"].strip())
        missing = [c["image"] for r in ds.records
                   for c in r["messages"][1]["content"]
                   if c["type"] == "image" and not Path(c["image"]).exists()]
        roles = {tuple(m["role"] for m in r["messages"]) for r in ds.records}
        print(f"{split:5} pairs={len(ds):3}  identical_chosen_rejected={identical}  "
              f"missing_images={len(missing)}  roles={roles}")
        if missing:
            raise SystemExit(f"missing image files, e.g. {missing[:3]}")
        # The prompt must NOT carry an assistant turn — the collate appends it.
        if roles != {("system", "user")}:
            raise SystemExit(f"expected prompt-only messages in {split}, got {roles}")
        if identical:
            raise SystemExit(
                f"{identical} pair(s) have chosen == rejected — these carry no "
                "preference signal and would contribute exactly zero gradient")
    print("\ndry run OK — prompts are answer-free, pairs differ, images resolve.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--init-from", default=None,
                    help="LoRA checkpoint to start from — normally the BC "
                         "checkpoint. DPO from a cold adapter is not the "
                         "standard recipe; SFT first.")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Pairs per step; each becomes 2 sequences")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="DPO wants a far lower LR than SFT")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=4)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--min-reward", type=float, default=None,
                    help="Off by default — see the module docstring for why")
    ap.add_argument("--pair-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device",
                    default="cuda" if (torch and torch.cuda.is_available()) else "cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    (AutoProcessor, Qwen2VLForConditionalGeneration, get_cosine_schedule_with_warmup,
     LoraConfig, PeftModel, get_peft_model, TaskType, process_vision_info) = _require_deps()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pair_dir = Path(args.pair_dir) if args.pair_dir else PAIR_DIR
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "right"
    collate = make_collate_fn(processor, process_vision_info)

    loaders = {}
    for split in ("train", "val", "test"):
        ds = PreferenceDataset(pair_dir / f"{split}.jsonl", args.min_reward)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size,
                                    shuffle=(split == "train"), collate_fn=collate)
        print(f"  {split}: {len(ds)} pairs")

    print(f"Loading {args.model_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": device})
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad_(False)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    if args.init_from:
        print(f"init LoRA from {args.init_from}")
        model = PeftModel.from_pretrained(model, args.init_from, is_trainable=True)
    else:
        print("WARNING: no --init-from; running DPO on a cold adapter. "
              "The standard recipe is SFT (train_bc_policy.py) first.")
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", modules_to_save=None))

    steps_per_epoch = max(1, len(loaders["train"]) // args.accum)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=max(1, steps_per_epoch // 2),
        num_training_steps=steps_per_epoch * args.epochs)

    best_loss, best_epoch, patience_count = float("inf"), 0, 0
    history = []
    print(f"baseline (policy == reference): {evaluate(model, loaders['val'], device, args.beta)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, accs, seen = 0.0, [], 0
        optim.zero_grad()
        for i, batch in enumerate(tqdm(loaders["train"], desc=f"epoch {epoch}")):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            n = batch["n_pairs"]
            policy = sequence_logprobs(model, batch)
            with torch.no_grad(), model.disable_adapter():
                ref = sequence_logprobs(model, batch)
            loss, acc, _ = dpo_loss(policy, ref, n, args.beta)
            (loss / args.accum).backward()
            running += float(loss)
            accs.append(float(acc))
            seen += 1
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()

        val = evaluate(model, loaders["val"], device, args.beta)
        train_loss = round(running / max(seen, 1), 4)
        print(f"epoch {epoch}: train_loss={train_loss} "
              f"train_acc={round(float(np.mean(accs)), 4)}  val={val}")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_pref_accuracy": round(float(np.mean(accs)), 4), **val})

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

    test = evaluate(model, loaders["test"], device, args.beta)
    summary = {
        "task": "dpo",
        "note": "PIPELINE VALIDATION — 44 train pairs, not an H2 result",
        "model_path": args.model_path,
        "init_from": args.init_from,
        "beta": args.beta,
        "seed": args.seed,
        "min_reward": args.min_reward,
        "reward_filter": "none (all pairs kept — see module docstring)"
                         if args.min_reward is None else f"> {args.min_reward}",
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "test": test,
        "history": history,
    }
    (out_dir / "dpo_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nbest epoch {best_epoch}  val_loss {best_loss}  test {test}")
    print(f"saved -> {out_dir}/")


if __name__ == "__main__":
    main()
