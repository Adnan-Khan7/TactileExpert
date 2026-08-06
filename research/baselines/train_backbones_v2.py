#!/usr/bin/env python3
"""
Train ResNet-50 / ViT-B/16 two-stream evaluator on data/splits_v3_unified (5 options).

Four task heads (QN/non_conformant dropped from the taxonomy — legacy
checkpoints containing a QN head are still loadable by the evaluator):
  QL — too_thick, broken_lines
  QP — missing_parts
  QT — missing_texture
  QE — extra_parts

Backbone is fine-tuned end-to-end (low lr) while heads use a higher lr.
Loss: masked BCE with pos_weight per option for class imbalance.

Saves to <--model-dir>/{resnet50,vit_b16}/ — directory names match what
evaluators/backbone_eval.py resolves, so installing a fleet is a copy.

Usage:
  python research/baselines/train_backbones_v2.py --backbone resnet50 --seed 42
  python research/baselines/train_backbones_v2.py --backbone vit_b_16 --seed 42
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
import random

import torchvision.models as tvm
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
except ImportError:
    raise ImportError("pip install scikit-learn")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = DATA_ROOT
DATA_DIR   = ROOT / "data" / "splits_v3_unified"
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


def set_seed(seed: int):
    """Seed every RNG the run touches.

    These trainers previously seeded nothing, so two runs on identical data
    differed by whatever the shuffle order and weight init happened to be. That
    made small deltas between checkpoints unreadable: a macro-F1 move of a few
    points could be a real effect or could be the seed. Fixing the seed makes a
    single run reproducible; VARYING it deliberately across runs is what gives
    the noise floor a comparison has to clear.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms(train: bool) -> T.Compose:
    """Deterministic tail, identical for train and eval.

    NOTE: this is applied to the natural and the tactile image SEPARATELY, so it
    must contain no random step. Random augmentation is sampled once per PAIR in
    paired_augment() below.
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def paired_augment(nat: Image.Image, tac: Image.Image, train: bool,
                   independent: bool = False):
    """Geometry for one pair: resize, crop, flip — sampled ONCE, applied to both.

    Why this is not a plain torchvision Compose: the label is a statement about
    the RELATIONSHIP between the two images ("does the tactile omit a part the
    photo shows"). Previously the train transform carried RandomCrop and
    RandomHorizontalFlip and was called once per image, so each image drew its
    own crop and its own mirror. That let a crop delete a structure from the
    tactile while leaving it in the photo — manufacturing a `missing_parts`
    positive that the label does not agree with — and mirrored roughly half of
    all pairs against each other. Val/test used the deterministic path, so the
    model trained under one regime and was scored under another.

    The pair is NOT pixel-registered: 278 of 300 sampled pairs have different
    aspect ratios (the tactile is a re-rendering, not an overlay). So a shared
    crop BOX is meaningless. What is shared instead is the crop POSITION as a
    fraction of each image's own frame, plus the flip decision — which keeps the
    two views framed consistently without pretending they are aligned.
    """
    nat = TF.resize(nat, 256)
    tac = TF.resize(tac, 256)
    if not train:
        return TF.center_crop(nat, 224), TF.center_crop(tac, 224)

    def place(img, u, v):
        w, h = img.size
        i = int(round(v * max(h - 224, 0)))
        j = int(round(u * max(w - 224, 0)))
        return TF.crop(img, i, j, 224, 224)

    if independent:
        # Reproduces the ORIGINAL behaviour for the ablation: geometry drawn
        # separately per image, so ~half of all pairs end up mirrored against
        # each other. Kept behind a flag rather than deleted, so the arm-A
        # results stay reproducible without git archaeology.
        out = []
        for img in (nat, tac):
            img = place(img, random.random(), random.random())
            if random.random() < 0.5:
                img = TF.hflip(img)
            out.append(img)
        return out[0], out[1]

    u, v = random.random(), random.random()          # one position per pair
    do_flip = random.random() < 0.5                  # one flip per pair
    nat, tac = place(nat, u, v), place(tac, u, v)
    if do_flip:
        nat, tac = TF.hflip(nat), TF.hflip(tac)
    return nat, tac


# ── Dataset ───────────────────────────────────────────────────────────────────
class TactilePairDataset(Dataset):
    def __init__(self, jsonl_path: Path, transform: T.Compose, train: bool = False,
                 independent_aug: bool = False):
        self.transform = transform
        self.train = train
        self.independent_aug = independent_aug
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
        nat_img = Image.open(IMAGE_ROOT / p["natural_image"]).convert("RGB")
        tac_img = Image.open(IMAGE_ROOT / p["tactile_image"]).convert("RGB")
        nat_img, tac_img = paired_augment(nat_img, tac_img, self.train,
                                          independent=self.independent_aug)
        nat = self.transform(nat_img)
        tac = self.transform(tac_img)
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
    parser.add_argument("--splits-dir",  default=None,
                        help=f"split directory (default: {DATA_DIR})")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed; vary it to measure run-to-run variance")
    parser.add_argument("--independent-aug", action="store_true",
                        help="draw crop/flip separately per image (the original, "
                             "buggy behaviour) — for reproducing ablation arm A")
    parser.add_argument("--model-dir",   default=None,
                        help=f"checkpoint output root (default: {MODELS_DIR}). "
                             "The default is the fleet the app loads; pass this to train "
                             "an ablation arm without replacing it.")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-heads",   type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_seed(args.seed)

    # Directory names match what evaluators/backbone_eval.py resolves, so
    # installing a trained fleet is a copy rather than a rename.
    DEPLOY_DIR = {"resnet50": "resnet50", "vit_b_16": "vit_b16"}
    data_dir = Path(args.splits_dir) if args.splits_dir else DATA_DIR
    models_root = Path(args.model_dir) if args.model_dir else MODELS_DIR
    out_dir  = models_root / DEPLOY_DIR[args.backbone]
    out_dir.mkdir(parents=True, exist_ok=True)
    device   = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  backbone={args.backbone}  options={ALL_OPTIONS}")
    print(f"  data: {data_dir}  save: {out_dir}")
    print(f"{'='*60}\n")

    train_ds = TactilePairDataset(data_dir/"train.jsonl", get_transforms(True),  train=True,
                                  independent_aug=args.independent_aug)
    val_ds   = TactilePairDataset(data_dir/"val.jsonl",   get_transforms(False), train=False)
    test_ds  = TactilePairDataset(data_dir/"test.jsonl",  get_transforms(False), train=False)

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

    # Per-option F1-max thresholds on VAL, stored in the checkpoint so the
    # deployed evaluator does not fall back to a hardcoded dict fitted against
    # some earlier fleet — which is exactly what ResNet/ViT did until now.
    cal_thresholds = {}
    vp, vy = [], []
    model.eval()
    with torch.no_grad():
        for nat, tac, lbl, msk in val_loader:
            pr = torch.sigmoid(model(nat.to(device), tac.to(device))).cpu().numpy()
            vp.append(pr); vy.append(lbl.numpy())
    vp = np.concatenate(vp); vy = np.concatenate(vy)
    for i, opt in enumerate(ALL_OPTIONS):
        s, y = vp[:, i], vy[:, i]
        best, bf = 0.5, -1.0
        for t in [round(0.05 * k, 2) for k in range(1, 20)]:
            tp = int(((s > t) & (y > 0.5)).sum()); fp = int(((s > t) & (y <= 0.5)).sum())
            fn = int(((s <= t) & (y > 0.5)).sum())
            f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f > bf: bf, best = f, t
        cal_thresholds[opt] = best
    print(f"  val F1-max thresholds: {cal_thresholds}")

    ckpt_path = out_dir / f"{args.backbone}_evaluator.pt"
    torch.save({
        "backbone":    args.backbone,
        "splits_dir":  str(data_dir),
        "seed":        args.seed,
        "independent_aug": bool(args.independent_aug),
        "state_dict":  best_state,
        "best_epoch":  best_epoch,
        "best_val_f1": best_f1,
        "all_options": ALL_OPTIONS,
        "task_heads":  {k: v for k, v in TASK_HEADS.items()},
        "cal_thresholds": cal_thresholds,
        "test_results": test_res,
    }, ckpt_path)
    print(f"\nSaved → {ckpt_path}")


if __name__ == "__main__":
    main()
