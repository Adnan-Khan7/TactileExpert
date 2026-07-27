#!/usr/bin/env python3
"""
DINOv2 ViT-L/14 probe — same MLP head structure as the CLIP probe but
using DINOv2 CLS-token features instead of CLIP image embeddings.

Hypothesis: DINOv2's self-supervised structural features (no language
alignment) capture line geometry (too_thick, broken_lines) better than
CLIP's semantically-aligned features.

Architecture:
  frozen DINOv2 ViT-L/14 (1024-d CLS token)
  → [nat | tac | nat-tac]  (3072-d)
  → 4 independent MLP heads (QL / QP / QT / QE)
  → Asymmetric Loss (same as CLIP probe)

Saves to: TactileEval_Context_And_Data/models/dino_probe_v2/

Usage:
  python code/baselines/train_dino_probe_v2.py --dim QL
  python code/baselines/train_dino_probe_v2.py --dim QP
  ...all five run sequentially via train_dino_probe_v2.sh
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
except ImportError:
    raise ImportError("pip install scikit-learn")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = DATA_ROOT
DATA_DIR   = ROOT / "data" / "splits_v2"
MODEL_DIR  = ROOT / "TactileEval_Context_And_Data" / "models" / "dino_probe_v2"
IMAGE_ROOT = ROOT / "images"

# ── Option / head layout (identical to CLIP probe) ─────────────────────────────
DIMENSION_OPTIONS = {
    "QL": ["too_thick",   "broken_lines"],
    "QP": ["missing_parts"],
    "QT": ["missing_texture"],
    "QE": ["extra_parts"],
}

# ── DINOv2 config ──────────────────────────────────────────────────────────────
DINO_REPO   = "facebookresearch/dinov2"
DINO_MODEL  = "dinov2_vitl14"
FEAT_DIM    = 1024          # DINOv2 ViT-L/14 CLS-token dimension
INPUT_DIM   = FEAT_DIM * 3  # [nat | tac | nat-tac]  →  3072  (no projection)
PROJ_DIM    = 512           # probe-ready projection dim (with contrastive head)

import torchvision.transforms as T

DINO_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]),
])


# ── Asymmetric Loss (same as CLIP probe) ───────────────────────────────────────
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos = gamma_neg, gamma_pos
        self.clip, self.eps = clip, eps

    def forward(self, logits, targets, mask):
        probs     = torch.sigmoid(logits)
        probs_neg = (probs + self.clip).clamp(max=1.0)
        log_pos   = torch.log(probs.clamp(min=self.eps))
        log_neg   = torch.log((1.0 - probs_neg).clamp(min=self.eps))
        loss      = targets * log_pos + (1.0 - targets) * log_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt    = targets * probs + (1.0 - targets) * (1.0 - probs)
            gamma = targets * self.gamma_pos + (1.0 - targets) * self.gamma_neg
            loss  = loss * (1.0 - pt) ** gamma
        return -(loss * mask).sum() / mask.sum().clamp(min=1.0)


# ── Feature extraction + caching ───────────────────────────────────────────────

def _extract_all_features(dino_model, jsonl_path: Path,
                           device: torch.device, cache_path: Path) -> dict:
    """
    Extract DINOv2 CLS-token features for every unique image in jsonl_path.
    Results are cached as a .npz file keyed by relative image path.
    Unlike the CLIP probe, features are shared across all dims so we cache
    once per split rather than once per split×dim.
    """
    if cache_path.exists():
        print(f"  Loading cached features from {cache_path.name}")
        d = np.load(cache_path, allow_pickle=True)
        return dict(d["feat_map"].item())

    with open(jsonl_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    unique_paths = sorted({r["natural_image"] for r in records} |
                          {r["tactile_image"]  for r in records})

    print(f"  Extracting DINOv2 features for {len(unique_paths)} images…")
    feat_map: dict[str, np.ndarray] = {}
    dino_model.eval()
    with torch.no_grad():
        for rel in tqdm(unique_paths, leave=False):
            img = Image.open(IMAGE_ROOT / rel).convert("RGB")
            t   = DINO_TRANSFORM(img).unsqueeze(0).to(device)
            f   = dino_model(t).squeeze(0).cpu().numpy()  # (1024,)
            feat_map[rel] = f

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, feat_map=feat_map)
    print(f"  Cached → {cache_path}")
    return feat_map


# ── Dataset ────────────────────────────────────────────────────────────────────

class DinoDataset(Dataset):
    def __init__(self, jsonl_path: Path, options: list[str],
                 feat_map: dict, cache_path: Path):
        with open(jsonl_path) as f:
            all_recs = [json.loads(l) for l in f if l.strip()]
        self.records = [r for r in all_recs if r["option_id"] in options]
        self.options  = options
        self.feat_map = feat_map

        # Pre-build tensors (feats, labels, masks)
        if cache_path.exists():
            d = np.load(cache_path)
            self.feats  = torch.from_numpy(d["feats"]).float()
            self.labels = torch.from_numpy(d["labels"]).float()
            self.masks  = torch.from_numpy(d["masks"]).float()
        else:
            self._build(cache_path)

    def _build(self, cache_path: Path):
        feats, labels, masks = [], [], []
        for r in self.records:
            fn = self.feat_map[r["natural_image"]]
            ft = self.feat_map[r["tactile_image"]]
            feat = np.concatenate([fn, ft, fn - ft])  # (3072,)
            lbl  = [float(r["label"]) if r["option_id"] == o else 0.0
                    for o in self.options]
            msk  = [1.0 if r["option_id"] == o else 0.0
                    for o in self.options]
            feats.append(feat); labels.append(lbl); masks.append(msk)

        F = np.array(feats,  dtype=np.float32)
        L = np.array(labels, dtype=np.float32)
        M = np.array(masks,  dtype=np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, feats=F, labels=L, masks=M)
        self.feats  = torch.from_numpy(F)
        self.labels = torch.from_numpy(L)
        self.masks  = torch.from_numpy(M)

    def __len__(self): return len(self.feats)
    def __getitem__(self, i): return self.feats[i], self.labels[i], self.masks[i]


# ── MLP head (same as CLIP probe) ──────────────────────────────────────────────

def build_head(n_out: int, hidden: int = 512,
               input_dim: int = INPUT_DIM) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, n_out),
    )


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(head, loader, options, device, loss_fn):
    head.eval()
    all_p, all_l, all_m, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for feat, lbl, msk in loader:
            feat, lbl, msk = feat.to(device), lbl.to(device), msk.to(device)
            total_loss += loss_fn(head(feat), lbl, msk).item()
            all_p.append(torch.sigmoid(head(feat)).cpu().numpy())
            all_l.append(lbl.cpu().numpy())
            all_m.append(msk.cpu().numpy())

    probs  = np.concatenate(all_p)
    labels = np.concatenate(all_l)
    masks  = np.concatenate(all_m)
    preds  = (probs > 0.5).astype(np.float32)

    per_opt, f1s = {}, []
    for i, opt in enumerate(options):
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",        required=True, choices=list(DIMENSION_OPTIONS))
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--hidden-dim", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=35)
    parser.add_argument("--gamma-neg",  type=float, default=4.0,
                        help="ASL gamma_neg — increase to suppress false positives. "
                             "Default 4.0; use 8.0 for QT (missing_texture over-flagging).")
    parser.add_argument("--projection-ckpt", type=str, default=None,
                        help="Path to contrastive projection head checkpoint. "
                             "If provided, uses projected 512-d features instead "
                             "of raw 3072-d DINOv2 features.")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Override output directory for checkpoints.")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    options    = DIMENSION_OPTIONS[args.dim]
    out_dir    = Path(args.model_dir) if args.model_dir else MODEL_DIR
    cache_dir  = out_dir / args.dim / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device(args.device)

    # Backbone feature cache is always in MODEL_DIR (shared regardless of variant)
    shared_backbone_cache = MODEL_DIR / "cache"
    shared_backbone_cache.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DINOv2 probe  dim={args.dim}  options={options}")
    print(f"  data: {DATA_DIR}")
    print(f"  save: {MODEL_DIR}/{args.dim.lower()}_evaluator.pt")
    print(f"  device={device}")
    print(f"{'='*60}")

    # Resolve effective feature dimension based on whether we use a projection head
    using_projection = args.projection_ckpt is not None
    eff_input_dim    = PROJ_DIM * 3 if using_projection else INPUT_DIM

    # Load DINOv2 backbone (frozen)
    print("Loading DINOv2 ViT-L/14…")
    dino = torch.hub.load(DINO_REPO, DINO_MODEL, verbose=False)
    dino.eval().to(device)
    for p in dino.parameters():
        p.requires_grad_(False)
    print(f"  DINOv2 ready  feat_dim={FEAT_DIM}  effective_input_dim={eff_input_dim}")

    # Optionally load pre-trained contrastive projection head
    proj_head = None
    if using_projection:
        from pretrain_tactile_contrastive import ProjectionHead
        ckpt = torch.load(args.projection_ckpt, map_location=device, weights_only=False)
        proj_head = ProjectionHead(
            in_dim=ckpt["in_dim"],
            probe_dim=ckpt["probe_dim"],
            contrastive_dim=ckpt["contrastive_dim"],
        )
        proj_head.load_state_dict(ckpt["state_dict"])
        proj_head.eval().to(device)
        for p in proj_head.parameters():
            p.requires_grad_(False)
        print(f"  Projection head loaded from {Path(args.projection_ckpt).name}"
              f"  (1024→{ckpt['probe_dim']}-d probe features)")

    # Extract / load backbone features (shared across plain and adapted variants)
    train_feats = _extract_all_features(dino, DATA_DIR/"train.jsonl", device,
                                         shared_backbone_cache/"train_feats.npz")
    val_feats   = _extract_all_features(dino, DATA_DIR/"val.jsonl",   device,
                                         shared_backbone_cache/"val_feats.npz")
    test_feats  = _extract_all_features(dino, DATA_DIR/"test.jsonl",  device,
                                         shared_backbone_cache/"test_feats.npz")

    # If using projection head, transform all cached features now
    if proj_head is not None:
        def _project(feat_map: dict) -> dict:
            projected = {}
            keys   = list(feat_map.keys())
            feats  = torch.tensor(np.stack([feat_map[k] for k in keys]),
                                  dtype=torch.float32, device=device)
            with torch.no_grad():
                out = proj_head.forward_probe(feats).cpu().numpy()
            for i, k in enumerate(keys):
                projected[k] = out[i]
            return projected
        print("  Applying projection head to all features…")
        train_feats = _project(train_feats)
        val_feats   = _project(val_feats)
        test_feats  = _project(test_feats)
        print(f"  Done — projected feature dim: {list(train_feats.values())[0].shape}")

    # Build datasets (pair feats + labels + masks, cached per dim)
    train_ds = DinoDataset(DATA_DIR/"train.jsonl", options, train_feats,
                           cache_dir/"train.npz")
    val_ds   = DinoDataset(DATA_DIR/"val.jsonl",   options, val_feats,
                           cache_dir/"val.npz")
    test_ds  = DinoDataset(DATA_DIR/"test.jsonl",  options, test_feats,
                           cache_dir/"test.npz")

    print(f"  records: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    head      = build_head(len(options), args.hidden_dim,
                           input_dim=eff_input_dim).to(device)
    loss_fn   = AsymmetricLoss(gamma_neg=args.gamma_neg)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_f1, best_state, best_epoch, patience_count = -1., None, 0, 0

    for epoch in range(1, args.epochs + 1):
        head.train()
        epoch_loss = 0.0
        for feat, lbl, msk in tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False):
            feat, lbl, msk = feat.to(device), lbl.to(device), msk.to(device)
            loss = loss_fn(head(feat), lbl, msk)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        val_res = evaluate(head, val_loader, options, device, loss_fn)
        if epoch % 10 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}  train_loss={epoch_loss/len(train_loader):.4f}")
            print_metrics(val_res, "val")

        if val_res["macro_f1"] > best_f1:
            best_f1    = val_res["macro_f1"]
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest: epoch={best_epoch}  val_macro_F1={best_f1:.4f}")
    head.load_state_dict(best_state)

    # Threshold calibration on val
    head.eval()
    vp, vl, vm = [], [], []
    with torch.no_grad():
        for feat, lbl, msk in val_loader:
            vp.append(torch.sigmoid(head(feat.to(device))).cpu().numpy())
            vl.append(lbl.numpy()); vm.append(msk.numpy())
    vp = np.concatenate(vp); vl = np.concatenate(vl); vm = np.concatenate(vm)

    cal_thresholds = {}
    print(f"\nThreshold calibration (val):")
    for i, opt in enumerate(options):
        obs = vm[:, i] == 1.0
        if not obs.any(): cal_thresholds[opt] = 0.5; continue
        best_t, best_f = 0.5, 0.0
        for t in np.arange(0.05, 0.96, 0.05):
            f = f1_score(vl[obs, i], (vp[obs, i] >= t).astype(float), zero_division=0)
            if f > best_f: best_f, best_t = f, float(t)
        cal_thresholds[opt] = round(best_t, 2)
        print(f"  {opt:<25} best_t={best_t:.2f}  val_F1={best_f:.3f}")

    test_res = evaluate(head, test_loader, options, device, loss_fn)
    print(f"\nTest results (balanced t=0.50):")
    print_metrics(test_res, "test")

    ckpt_path = out_dir / f"{args.dim.lower()}_evaluator.pt"
    torch.save({
        "dim":            args.dim,
        "options":        options,
        "state_dict":     best_state,
        "hidden_dim":     args.hidden_dim,
        "input_dim":      INPUT_DIM,
        "feat_dim":       FEAT_DIM,
        "backbone":       "dinov2_vitl14",
        "best_epoch":     best_epoch,
        "best_val_f1":    best_f1,
        "cal_thresholds": cal_thresholds,
        "test_results":   test_res,
    }, ckpt_path)
    print(f"\nSaved → {ckpt_path}")


if __name__ == "__main__":
    main()
