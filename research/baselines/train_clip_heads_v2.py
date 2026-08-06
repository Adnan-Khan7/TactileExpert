#!/usr/bin/env python3
"""
Train CLIP Probe heads on the data/splits_v3_unified dataset (5 options).

Four heads (QN/non_conformant dropped from the taxonomy):
  QL  — too_thick, broken_lines
  QP  — missing_parts
  QT  — missing_texture
  QE  — extra_parts

Saves to <--model-dir>/, one checkpoint per dim. Features are cached under
--cache-dir (default: alongside --model-dir); several seeds can share one cache
because features depend on the images, not the seed.

Usage — one head at a time (run via train_clip_heads_v2.sh for all four):
  python research/baselines/train_clip_heads_v2.py --dim QL --seed 42
  python research/baselines/train_clip_heads_v2.py --dim QE --seed 42
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import json
import sys
from pathlib import Path

import random

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

import open_clip

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = DATA_ROOT
DATA_DIR   = ROOT / "data" / "splits_v3_unified"
MODEL_DIR  = ROOT / "TactileEval_Context_And_Data" / "models" / "clip_probe_v2"
IMAGE_ROOT = ROOT / "images"

# ── Architecture ──────────────────────────────────────────────────────────────
CLIP_MODEL      = "ViT-L-14"
CLIP_PRETRAINED = "laion2b_s32b_b82k"
FEAT_DIM        = 768
INPUT_DIM       = FEAT_DIM * 3

DIMENSION_OPTIONS = {
    "QL": ["too_thick",   "broken_lines"],
    "QP": ["missing_parts"],
    "QT": ["missing_texture"],
    "QE": ["extra_parts"],
}


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


# ── Asymmetric Loss ───────────────────────────────────────────────────────────
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


# ── Dataset ───────────────────────────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, jsonl_path, options, clip_model, preprocess,
                 device, cache_path=None):
        self.options = options
        with open(jsonl_path) as f:
            all_recs = [json.loads(l) for l in f if l.strip()]
        self.records = [r for r in all_recs if r["option_id"] in options]

        # Cache FEATURES only, never labels. Features depend on the images;
        # labels change every time a verification pass lands. Caching them
        # together means a relabelled split silently trains on stale labels,
        # because the cache key (the split directory) has not changed.
        # The fingerprint pins the record IDENTITY and order, so the cached
        # feature rows still line up; labels and masks are rebuilt every run.
        cache_path = Path(cache_path) if cache_path else None
        fp = self._fingerprint()
        self.feats = None
        if cache_path and cache_path.exists():
            d = np.load(cache_path, allow_pickle=True)
            if str(d.get("fingerprint", "")) == fp:
                self.feats = torch.from_numpy(d["feats"]).float()
            else:
                print("  cache fingerprint mismatch (record set changed) — re-extracting")
        if self.feats is None:
            self.feats = self._extract(clip_model, preprocess, cache_path, device, fp)
        self.labels, self.masks = self._labels()

    def _fingerprint(self):
        import hashlib
        h = hashlib.sha1()
        for r in self.records:
            h.update(f'{r["pair_id"]}|{r["option_id"]}\n'.encode())
        return h.hexdigest()

    def _labels(self):
        L = np.array([[float(r["label"]) if r["option_id"] == o else 0.0
                       for o in self.options] for r in self.records], dtype=np.float32)
        M = np.array([[1.0 if r["option_id"] == o else 0.0
                       for o in self.options] for r in self.records], dtype=np.float32)
        return torch.from_numpy(L), torch.from_numpy(M)

    def _extract(self, model, preprocess, cache_path, device, fingerprint):
        print(f"  Extracting CLIP features for {len(self.records)} records...")
        model.eval()
        feats = []
        with torch.no_grad():
            for rec in tqdm(self.records, leave=False):
                fn = preprocess(Image.open(IMAGE_ROOT / rec["natural_image"]).convert("RGB")
                                ).unsqueeze(0).to(device)
                ft = preprocess(Image.open(IMAGE_ROOT / rec["tactile_image"]).convert("RGB")
                                ).unsqueeze(0).to(device)
                en = model.encode_image(fn); en = en / en.norm(dim=-1, keepdim=True)
                et = model.encode_image(ft); et = et / et.norm(dim=-1, keepdim=True)
                feats.append(torch.cat([en, et, en - et], dim=-1).squeeze(0).cpu().numpy())

        F = np.stack(feats).astype(np.float32)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, feats=F, fingerprint=fingerprint)
        return torch.from_numpy(F)

    def __len__(self): return len(self.feats)
    def __getitem__(self, i): return self.feats[i], self.labels[i], self.masks[i]


# ── Head architecture ─────────────────────────────────────────────────────────
def build_head(n_out, hidden=512):
    return nn.Sequential(
        nn.Linear(INPUT_DIM, hidden), nn.BatchNorm1d(hidden),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, n_out))


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, options, device, loss_fn):
    model.eval()
    all_probs, all_labels, all_masks = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for feat, lbl, msk in loader:
            feat, lbl, msk = feat.to(device), lbl.to(device), msk.to(device)
            total_loss += loss_fn(model(feat), lbl, msk).item()
            all_probs.append(torch.sigmoid(model(feat)).cpu().numpy())
            all_labels.append(lbl.cpu().numpy())
            all_masks.append(msk.cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    masks  = np.concatenate(all_masks)
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
                macro_f1=round(float(np.mean(f1s)) if f1s else 0.0,4),
                per_option=per_opt)


def print_metrics(res, label=""):
    tag = f"[{label}] " if label else ""
    print(f"  {tag}loss={res['loss']:.4f}  macro_F1={res['macro_f1']:.4f}")
    for opt, m in res["per_option"].items():
        print(f"    {opt:<25} F1={m['f1']:.3f}  AUC={m['auc']:.3f}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"pos={m['n_pos']}/{m['n_total']}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",        required=True, choices=list(DIMENSION_OPTIONS))
    parser.add_argument("--cache-dir", default=None,
                        help="feature-cache root (default: alongside --model-dir). "
                             "Point several seeds at one cache to extract features once.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed; vary it to measure run-to-run variance")
    parser.add_argument("--splits-dir", default=None,
                        help=f"split directory (default: {DATA_DIR})")
    parser.add_argument("--model-dir",  default=None,
                        help=f"checkpoint output directory (default: {MODEL_DIR}). "
                             "The default is the fleet the app loads; pass this to train "
                             "an ablation arm without replacing it.")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--hidden-dim", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=35)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_seed(args.seed)

    options   = DIMENSION_OPTIONS[args.dim]
    data_dir  = Path(args.splits_dir) if args.splits_dir else DATA_DIR
    # Cache is namespaced by split directory. The .npz holds feats AND labels AND
    # masks, so a cache built from a different split silently trains on the wrong
    # labels — the run looks completely normal. Namespacing makes that impossible.
    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    _cache_root = Path(args.cache_dir) if args.cache_dir else model_dir
    cache_dir = _cache_root / args.dim / "cache" / f"{data_dir.name}_evaltf"
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  dim={args.dim}  options={options}")
    print(f"  data:  {data_dir}")
    print(f"  cache: {cache_dir}")
    print(f"  save:  {model_dir}/{args.dim.lower()}_evaluator.pt")
    print(f"  device={device}")
    print(f"{'='*60}")

    # open_clip returns (model, preprocess_TRAIN, preprocess_VAL). Take the VAL
    # transform. preprocess_train is RandomResizedCrop(scale=(0.9,1.0)) — random
    # — and this code path extracts train, val AND test features with whatever it
    # is handed, then CACHES them. Using the train transform therefore (a) drew a
    # different random crop for the natural and the tactile image of a pair,
    # (b) made val/test features nondeterministic, and (c) froze one arbitrary
    # crop into the cache. It also disagreed with evaluators/clip_probe_eval.py,
    # which scores with preprocess_val — so the heads were fit on one transform
    # and deployed on another.
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device)
    clip_model.eval()
    for p in clip_model.parameters(): p.requires_grad_(False)

    train_ds = PairDataset(data_dir/"train.jsonl", options, clip_model,
                           preprocess, device, cache_dir/"train.npz")
    val_ds   = PairDataset(data_dir/"val.jsonl",   options, clip_model,
                           preprocess, device, cache_dir/"val.npz")
    test_ds  = PairDataset(data_dir/"test.jsonl",  options, clip_model,
                           preprocess, device, cache_dir/"test.npz")

    print(f"  records: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    head      = build_head(len(options), args.hidden_dim).to(device)
    loss_fn   = AsymmetricLoss()
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
            best_f1   = val_res["macro_f1"]
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

    # ── Threshold calibration on val ─────────────────────────────────────────
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

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_res = evaluate(head, test_loader, options, device, loss_fn)
    print(f"\nTest results (balanced t=0.50):")
    print_metrics(test_res, "test")

    # Calibrated test metrics
    tp_arr, tl_arr, tm_arr = [], [], []
    head.eval()
    with torch.no_grad():
        for feat, lbl, msk in test_loader:
            tp_arr.append(torch.sigmoid(head(feat.to(device))).cpu().numpy())
            tl_arr.append(lbl.numpy()); tm_arr.append(msk.numpy())
    tp_arr = np.concatenate(tp_arr); tl_arr = np.concatenate(tl_arr); tm_arr = np.concatenate(tm_arr)

    print(f"\nTest results (calibrated thresholds):")
    cal_test = {}
    for i, opt in enumerate(options):
        obs = tm_arr[:, i] == 1.0
        if not obs.any(): continue
        t  = cal_thresholds[opt]
        yt = tl_arr[obs, i]
        yp = (tp_arr[obs, i] >= t).astype(float)
        f1  = f1_score(yt, yp, zero_division=0)
        auc = roc_auc_score(yt, tp_arr[obs, i]) if len(np.unique(yt)) > 1 else 0.0
        cal_test[opt] = dict(f1=round(f1,4), auc=round(auc,4), threshold=t,
                             n_pos=int(yt.sum()), n_total=int(obs.sum()))
        print(f"  {opt:<25} F1={f1:.3f}  AUC={auc:.3f}  t={t:.2f}  "
              f"pos={int(yt.sum())}/{int(obs.sum())}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = model_dir / f"{args.dim.lower()}_evaluator.pt"
    torch.save({
        "dim":            args.dim,
        "options":        options,
        "splits_dir":     str(data_dir),
        "seed":        args.seed,
        "state_dict":     best_state,
        "hidden_dim":     args.hidden_dim,
        "best_epoch":     best_epoch,
        "best_val_f1":    best_f1,
        "cal_thresholds": cal_thresholds,
        "test_results":   {
            "balanced":   test_res,
            "calibrated": cal_test,
            "per_option": {
                opt: {**test_res["per_option"].get(opt, {}),
                      "threshold_calibrated": cal_thresholds.get(opt, 0.5)}
                for opt in options
            },
        },
    }, ckpt_path)
    print(f"\nSaved → {ckpt_path}")


if __name__ == "__main__":
    main()
