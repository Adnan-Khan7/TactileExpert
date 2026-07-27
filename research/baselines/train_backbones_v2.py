#!/usr/bin/env python3
"""
Train ResNet-50 / ViT-B/16 two-stream evaluator on data/splits_v2 (5 options).

Four task heads (QN/non_conformant dropped from the taxonomy — legacy
checkpoints containing a QN head are still loadable by the evaluator):
  QL — too_thick, broken_lines
  QP — missing_parts
  QT — missing_texture
  QE — extra_parts

Backbone is fine-tuned end-to-end (low lr) while heads use a higher lr.
Loss: masked BCE with pos_weight per option for class imbalance.

Saves to:
  TactileEval_Context_And_Data/models/resnet50_v2/resnet50_evaluator.pt
  TactileEval_Context_And_Data/models/vit_b16_v2/vit_b_16_evaluator.pt

Usage:
  python code/baselines/train_backbones_v2.py --backbone resnet50
  python code/baselines/train_backbones_v2.py --backbone vit_b_16
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
except ImportError:
    raise ImportError("pip install scikit-learn")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = DATA_ROOT
DATA_DIR   = ROOT / "data" / "splits_v2"
IMAGE_ROOT = ROOT / "images"
MODELS_DIR = ROOT / "TactileEval_Context_And_Data" / "models"

# ── Option / head layout ──────────────────────────────────────────────────────
TASK_HEADS = {
    "QL": ["too_thick",   "broken_lines"],
    "QP": ["missing_parts"],
    "QT": ["missing_texture"],
    "QE": ["extra_parts"],
}
ALL_OPTIONS = [o for opts in TASK_HEADS.values() for o in opts]
HEAD_ORDER  = ["QL", "QP", "QT", "QE"]


# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms(train: bool) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────
class TactilePairDataset(Dataset):
    def __init__(self, jsonl_path: Path, transform: T.Compose):
        self.transform = transform
        pair_data: dict = defaultdict(lambda: {
            "natural_image": None, "tactile_image": None,
            "labels": {opt: None for opt in ALL_OPTIONS},
        })
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if r["option_id"] not in ALL_OPTIONS:
                    continue
                pid = r["pair_id"]
                pair_data[pid]["natural_image"] = r["natural_image"]
                pair_data[pid]["tactile_image"]  = r["tactile_image"]
                pair_data[pid]["labels"][r["option_id"]] = r["label"]
        self.pairs = list(pair_data.values())

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p   = self.pairs[idx]
        nat = self.transform(Image.open(IMAGE_ROOT / p["natural_image"]).convert("RGB"))
        tac = self.transform(Image.open(IMAGE_ROOT / p["tactile_image"]).convert("RGB"))
        labels = torch.zeros(len(ALL_OPTIONS))
        masks  = torch.zeros(len(ALL_OPTIONS))
        for i, opt in enumerate(ALL_OPTIONS):
            v = p["labels"][opt]
            if v is not None:
                labels[i] = float(v)
                masks[i]  = 1.0
        return nat, tac, labels, masks


# ── Model ─────────────────────────────────────────────────────────────────────
class TactileEvaluator(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        if backbone_name == "resnet50":
            base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
        elif backbone_name == "vit_b_16":
            base = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
            feat_dim = base.heads.head.in_features
            base.heads = nn.Identity()
        else:
            raise ValueError(backbone_name)

        self.backbone = base
        combined = feat_dim * 3

        def head(n_out):
            return nn.Sequential(
                nn.Linear(combined, 256), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(256, n_out))

        self.heads = nn.ModuleDict({dim: head(len(opts))
                                    for dim, opts in TASK_HEADS.items()})

    def forward(self, nat, tac):
        fn = self.backbone(nat)
        ft = self.backbone(tac)
        feat = torch.cat([fn, ft, fn - ft], dim=-1)
        return torch.cat([self.heads[d](feat) for d in HEAD_ORDER], dim=-1)


# ── Loss ──────────────────────────────────────────────────────────────────────
def masked_bce(logits, labels, masks, pos_weights, device):
    loss = torch.tensor(0.0, device=device)
    offset = 0
    for dim in HEAD_ORDER:
        opts = TASK_HEADS[dim]
        n  = len(opts)
        lg = logits[:, offset:offset+n]
        lb = labels[:, offset:offset+n]
        mk = masks[:,  offset:offset+n]
        pw = torch.tensor([pos_weights[o] for o in opts], dtype=torch.float32,
                           device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        loss += (bce(lg, lb) * mk).sum() / mk.sum().clamp(min=1.0)
        offset += n
    return loss


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, pos_weights):
    model.eval()
    all_probs, all_labels, all_masks, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for nat, tac, lbl, msk in loader:
            nat, tac = nat.to(device), tac.to(device)
            lbl, msk = lbl.to(device), msk.to(device)
            logits = model(nat, tac)
            total_loss += masked_bce(logits, lbl, msk, pos_weights, device).item()
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(lbl.cpu().numpy())
            all_masks.append(msk.cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    masks  = np.concatenate(all_masks)
    preds  = (probs > 0.5).astype(np.float32)

    per_opt, f1s = {}, []
    for i, opt in enumerate(ALL_OPTIONS):
        obs = masks[:, i] == 1.0
        if not obs.any(): continue
        yt, yp, yb = labels[obs, i], preds[obs, i], probs[obs, i]
        f1  = f1_score(yt, yp, zero_division=0)
        auc = roc_auc_score(yt, yb) if len(np.unique(yt)) > 1 else 0.0
        per_opt[opt] = dict(f1=round(f1,4), auc=round(auc,4),
                            precision=round(precision_score(yt,yp,zero_division=0),4),
                            recall=round(recall_score(yt,yp,zero_division=0),4),
                            n_pos=int(yt.sum()), n_total=int(obs.sum()))
        f1s.append(f1)
    return dict(loss=total_loss/max(len(loader),1),
                macro_f1=round(float(np.mean(f1s)) if f1s else 0.0, 4),
                per_option=per_opt)


def print_metrics(res, label=""):
    tag = f"[{label}] " if label else ""
    print(f"  {tag}loss={res['loss']:.4f}  macro_F1={res['macro_f1']:.4f}")
    for opt, m in res["per_option"].items():
        print(f"    {opt:<25} F1={m['f1']:.3f}  AUC={m['auc']:.3f}  "
              f"pos={m['n_pos']}/{m['n_total']}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",    required=True, choices=["resnet50", "vit_b_16"])
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-heads",   type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    bb_slug = args.backbone.replace("_", "")
    out_dir = MODELS_DIR / f"{bb_slug}_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  backbone={args.backbone}  options={ALL_OPTIONS}")
    print(f"  data: {DATA_DIR}  save: {out_dir}")
    print(f"{'='*60}\n")

    train_ds = TactilePairDataset(DATA_DIR/"train.jsonl", get_transforms(True))
    val_ds   = TactilePairDataset(DATA_DIR/"val.jsonl",   get_transforms(False))
    test_ds  = TactilePairDataset(DATA_DIR/"test.jsonl",  get_transforms(False))

    print(f"Pairs: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, num_workers=4)

    # Pos weights from training data
    pos_weights = {}
    pos_counts  = defaultdict(int)
    neg_counts  = defaultdict(int)
    for p in train_ds.pairs:
        for opt in ALL_OPTIONS:
            v = p["labels"][opt]
            if v is None: continue
            (pos_counts if v == 1 else neg_counts)[opt] += 1
    print("\nPos weights (neg/pos):")
    for opt in ALL_OPTIONS:
        w = round(neg_counts[opt] / pos_counts[opt], 2) if pos_counts[opt] else 1.0
        pos_weights[opt] = w
        print(f"  {opt:<25} pos={pos_counts[opt]}  neg={neg_counts[opt]}  w={w:.2f}")

    model = TactileEvaluator(args.backbone).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.heads.parameters(),    "lr": args.lr_heads},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_backbone * 0.01)

    best_f1, best_state, best_epoch, patience_count = -1., None, 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for nat, tac, lbl, msk in tqdm(train_loader, desc=f"Epoch {epoch:3d}",
                                        leave=False):
            nat, tac = nat.to(device), tac.to(device)
            lbl, msk = lbl.to(device), msk.to(device)
            loss = masked_bce(model(nat, tac), lbl, msk, pos_weights, device)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        val_res = evaluate(model, val_loader, device, pos_weights)
        if epoch % 5 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}  train_loss={epoch_loss/len(train_loader):.4f}")
            print_metrics(val_res, "val")

        if val_res["macro_f1"] > best_f1:
            best_f1 = val_res["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest: epoch={best_epoch}  val_macro_F1={best_f1:.4f}")
    model.load_state_dict(best_state)

    test_res = evaluate(model, test_loader, device, pos_weights)
    print("\nTest results:")
    print_metrics(test_res, "test")

    ckpt_path = out_dir / f"{args.backbone}_evaluator.pt"
    torch.save({
        "backbone":    args.backbone,
        "state_dict":  best_state,
        "best_epoch":  best_epoch,
        "best_val_f1": best_f1,
        "all_options": ALL_OPTIONS,
        "task_heads":  {k: v for k, v in TASK_HEADS.items()},
        "test_results": test_res,
    }, ckpt_path)
    print(f"\nSaved → {ckpt_path}")


if __name__ == "__main__":
    main()
