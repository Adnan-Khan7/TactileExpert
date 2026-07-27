#!/usr/bin/env python3
"""
Domain-adaptive contrastive pre-training for tactile graphic evaluation.

We sidestep the need for external sketch datasets by generating pseudo-tactile
edge maps directly from our own natural images using differentiable Sobel
edge detection (PyTorch-native, no OpenCV required).

Positive pair : (DINOv2(natural_image),  DINOv2(edge_map_of_same_image))
Negative pairs: all other images in the batch (NT-Xent / SimCLR loss)

A lightweight projection head P (1024→512→256) is trained to align the
photo feature space with the edge-map feature space. The adapted probe
then uses [P(nat) | P(tac) | P(nat)−P(tac)] = 768-d as input to the
MLP defect-detection heads instead of raw 3072-d DINOv2 features.

After training, the projection head checkpoint is loaded by
train_dino_probe_v2.py via --projection-ckpt to train the adapted probe.

Saves to:
  TactileEval_Context_And_Data/models/dino_probe_v2/projection_head.pt

Usage:
  python code/baselines/pretrain_tactile_contrastive.py
  python code/baselines/pretrain_tactile_contrastive.py --epochs 100 --batch-size 64
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = DATA_ROOT
DATA_DIR   = ROOT / "data" / "splits_v2"
IMAGE_ROOT = ROOT / "images"
MODEL_DIR  = ROOT / "TactileEval_Context_And_Data" / "models" / "dino_probe_v2"

DINO_REPO  = "facebookresearch/dinov2"
DINO_MODEL = "dinov2_vitl14"
FEAT_DIM   = 1024
PROJ_DIM   = 256      # contrastive projection output dim
PROBE_DIM  = 512      # width of final projection used for the defect probe

IMAGENET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]),
])


# ── Edge map generation (PyTorch-native, GPU-accelerated) ─────────────────────

def _gaussian_kernel(size: int = 5, sigma: float = 1.0,
                     device="cpu") -> torch.Tensor:
    ax  = torch.arange(size, device=device) - size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k   = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (k / k.sum()).view(1, 1, size, size)


_SOBEL_X = torch.tensor(
    [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor(
    [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def generate_edge_map(img_tensor: torch.Tensor,
                      threshold: float = 0.12,
                      blur_sigma: float = 1.2,
                      dilate: bool = False) -> torch.Tensor:
    """
    Convert an ImageNet-normalised image tensor to a binary edge map.

    img_tensor : (3, H, W)  — the natural image (already normalised)
    Returns    : (3, H, W)  — edge map repeated across 3 channels,
                               normalised to ImageNet stats so DINOv2
                               can process it directly.
    """
    device = img_tensor.device
    x = img_tensor.unsqueeze(0)                          # (1, 3, H, W)

    # Convert to grayscale
    gray = (0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]).unsqueeze(1)

    # Gaussian blur to reduce noise
    gk = _gaussian_kernel(5, blur_sigma, device=device)
    blurred = F.conv2d(gray, gk, padding=2)

    # Sobel gradient magnitude
    kx = _SOBEL_X.to(device)
    ky = _SOBEL_Y.to(device)
    gx = F.conv2d(blurred, kx, padding=1)
    gy = F.conv2d(blurred, ky, padding=1)
    mag = (gx**2 + gy**2).sqrt()

    # Normalise to [0, 1] and threshold
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    edges = (mag > threshold).float()

    # Optional dilation (creates thicker lines — simulates tactile embossing)
    if dilate:
        kernel = torch.ones(1, 1, 3, 3, device=device)
        edges  = (F.conv2d(edges, kernel, padding=1) > 0).float()

    # Convert binary edge map to 3-channel and re-normalise to ImageNet stats
    # so DINOv2 receives the same input distribution it was trained on
    edges_rgb = edges.expand(1, 3, -1, -1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    edges_norm = (edges_rgb - mean) / std

    return edges_norm.squeeze(0)                         # (3, H, W)


# ── Dataset ────────────────────────────────────────────────────────────────────

class NaturalImageDataset(Dataset):
    """
    Returns unique natural images from all three splits.
    Each __getitem__ returns (img_tensor, img_tensor) — the same image twice
    so the DataLoader can generate both anchor and edge-map on the fly.
    """
    def __init__(self):
        seen, self.paths = set(), []
        for split in ["train", "val", "test"]:
            with open(DATA_DIR / f"{split}.jsonl") as f:
                for line in f:
                    r = json.loads(line)
                    nat = r["natural_image"]
                    if nat not in seen:
                        seen.add(nat)
                        self.paths.append(nat)
        print(f"  Unique natural images: {len(self.paths)}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(IMAGE_ROOT / self.paths[idx]).convert("RGB")
        t   = IMAGENET_TRANSFORM(img)
        return t   # (3, 224, 224)


# ── NT-Xent loss (SimCLR) ──────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    z1, z2 : (N, D)  — already L2-normalised
    Positive pair: (z1[i], z2[i])
    Negatives:     all other N−1 samples in the batch (both views)
    """
    N  = z1.shape[0]
    z  = torch.cat([z1, z2], dim=0)           # (2N, D)
    sim = torch.mm(z, z.T) / temperature       # (2N, 2N)

    # Mask self-similarities
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Labels: for each z1[i] the positive is z2[i] = index N+i, and vice versa
    labels = torch.cat([
        torch.arange(N, 2*N, device=z.device),
        torch.arange(N, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ── Projection head ────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Maps DINOv2 1024-d CLS tokens to a shared contrastive embedding space,
    then to a wider probe-ready embedding used for defect classification.

    contrastive_out (256-d): used during pre-training with NT-Xent loss
    probe_out       (512-d): used as input features for the defect probe
    """
    def __init__(self, in_dim=FEAT_DIM, hidden_dim=512,
                 contrastive_dim=PROJ_DIM, probe_dim=PROBE_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.contrastive_head = nn.Linear(hidden_dim, contrastive_dim)
        self.probe_head       = nn.Sequential(
            nn.Linear(hidden_dim, probe_dim),
            nn.BatchNorm1d(probe_dim),
        )

    def forward_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.contrastive_head(self.shared(x)), dim=1)

    def forward_probe(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe_head(self.shared(x))


# ── Augmentation helpers ───────────────────────────────────────────────────────

def random_edge_params() -> tuple[float, float, bool]:
    """Sample random edge-map parameters for augmentation diversity."""
    threshold  = random.uniform(0.08, 0.22)
    blur_sigma = random.uniform(0.8, 2.0)
    dilate     = random.random() < 0.3   # 30% chance of thick-line effect
    return threshold, blur_sigma, dilate


# ── Training ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--batch-size",  type=int,   default=48)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TactileAug contrastive pre-training")
    print(f"  positive pair: (natural image, its Sobel edge map)")
    print(f"  backbone: frozen DINOv2 ViT-L/14")
    print(f"  projection: {FEAT_DIM}→512→{PROJ_DIM} (contrastive)")
    print(f"              {FEAT_DIM}→512→{PROBE_DIM} (probe features)")
    print(f"  device={device}  epochs={args.epochs}  τ={args.temperature}")
    print(f"{'='*60}\n")

    # Load frozen DINOv2
    print("Loading DINOv2 ViT-L/14 (frozen)…")
    dino = torch.hub.load(DINO_REPO, DINO_MODEL, verbose=False)
    dino.eval().to(device)
    for p in dino.parameters():
        p.requires_grad_(False)
    print("  DINOv2 ready.\n")

    # Dataset + loader
    dataset = NaturalImageDataset()
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=4,
                         pin_memory=(device.type == "cuda"),
                         drop_last=True)

    # Projection head (only trained component)
    proj      = ProjectionHead().to(device)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_loss, best_state = float("inf"), None
    history = []

    print("Pre-training…")
    for epoch in range(1, args.epochs + 1):
        proj.train()
        epoch_loss = 0.0
        n_batches  = 0

        for img_batch in tqdm(loader, desc=f"Epoch {epoch:3d}", leave=False):
            img_batch = img_batch.to(device)               # (B, 3, H, W)
            B = img_batch.shape[0]

            # Generate edge maps with random parameters per sample
            edge_batch = torch.stack([
                generate_edge_map(img_batch[i],
                                  *random_edge_params())
                for i in range(B)
            ])                                             # (B, 3, H, W)

            # Extract DINOv2 features (frozen)
            with torch.no_grad():
                feat_nat  = dino(img_batch)               # (B, 1024)
                feat_edge = dino(edge_batch)               # (B, 1024)

            # Project to contrastive space
            z_nat  = proj.forward_contrastive(feat_nat)   # (B, 256)
            z_edge = proj.forward_contrastive(feat_edge)  # (B, 256)

            # NT-Xent loss
            loss = nt_xent_loss(z_nat, z_edge, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history.append({"epoch": epoch, "loss": round(avg_loss, 5)})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in proj.state_dict().items()}

    print(f"\nBest loss: {best_loss:.4f}")

    # Save checkpoint
    ckpt_path = MODEL_DIR / "projection_head.pt"
    torch.save({
        "state_dict":      best_state,
        "in_dim":          FEAT_DIM,
        "probe_dim":       PROBE_DIM,
        "contrastive_dim": PROJ_DIM,
        "backbone":        DINO_MODEL,
        "best_loss":       best_loss,
        "history":         history,
    }, ckpt_path)
    print(f"Saved → {ckpt_path}")
    print(f"\nNext: run train_dino_probe_v2.py --projection-ckpt {ckpt_path}")


if __name__ == "__main__":
    main()
