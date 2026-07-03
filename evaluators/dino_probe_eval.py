"""DINOv2 ViT-L/14 probe evaluator for TactileExpert.

Uses frozen DINOv2 self-supervised features (1024-d CLS token) instead of
CLIP's language-aligned features.  DINOv2 captures structural / geometric
properties better than CLIP, improving too_thick and missing_parts detection.

Architecture: frozen DINOv2 ViT-L/14 → [nat|tac|nat-tac] (3072-d) → 4 MLP heads
Heads: QL (too_thick, broken_lines) | QP (missing_parts) | QT (missing_texture) | QE (extra_parts)
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from .constants import DINO_THRESHOLDS_CALIBRATED, DEFAULT_THRESHOLD_MODE, build_result

HERE = Path(__file__).resolve().parent

DINO_REPO  = "facebookresearch/dinov2"
DINO_MODEL = "dinov2_vitl14"
FEAT_DIM   = 1024
INPUT_DIM  = FEAT_DIM * 3   # 3072

DINO_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]),
])


class DINOProbeEvaluator:
    """DINOv2 ViT-L/14 (frozen) + four 2-layer MLP task heads."""

    _HEAD_CONFIGS = [
        ("QL", "ql_evaluator.pt", ["too_thick", "broken_lines"]),
        ("QP", "qp_evaluator.pt", ["missing_parts"]),
        ("QT", "qt_evaluator.pt", ["missing_texture"]),
        ("QE", "qe_evaluator.pt", ["extra_parts"]),
    ]

    def __init__(self, models_dir=None, device=None):
        import torch.hub
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        models_dir  = Path(models_dir) if models_dir else HERE.parent / "models" / "dino_probe"

        print(f"[DINOv2 Probe] Loading {DINO_MODEL}…")
        self._dino = torch.hub.load(DINO_REPO, DINO_MODEL, verbose=False)
        self._dino.eval().to(self.device)
        for p in self._dino.parameters():
            p.requires_grad_(False)

        self._heads: dict[str, tuple[nn.Module, list[str]]] = {}
        self.calibrated_thresholds = dict(DINO_THRESHOLDS_CALIBRATED)

        for _dim, pt_name, options in self._HEAD_CONFIGS:
            ckpt_path = models_dir / pt_name
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            head = nn.Sequential(
                nn.Linear(INPUT_DIM, ckpt["hidden_dim"]),
                nn.BatchNorm1d(ckpt["hidden_dim"]),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(ckpt["hidden_dim"], len(options)),
            )
            head.load_state_dict(ckpt["state_dict"])
            head.eval().to(self.device)
            self._heads[_dim] = (head, options)
            for opt, t in ckpt.get("cal_thresholds", {}).items():
                self.calibrated_thresholds[opt] = t

        print("[DINOv2 Probe] Ready.")

    @torch.no_grad()
    def _encode(self, path: str) -> torch.Tensor:
        img = DINO_TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
        return self._dino(img)   # (1, 1024) — CLS token, no normalisation needed

    @torch.no_grad()
    def _feat(self, nat_path: str, tac_path: str) -> torch.Tensor:
        n = self._encode(nat_path)
        t = self._encode(tac_path)
        return torch.cat([n, t, n - t], dim=-1)   # (1, 3072)

    def score_pair(self, nat_path: str, tac_path: str) -> dict[str, float]:
        feat = self._feat(nat_path, tac_path)
        probs: dict[str, float] = {}
        for _dim, (head, options) in self._heads.items():
            raw = torch.sigmoid(head(feat))[0]
            for j, opt in enumerate(options):
                probs[opt] = round(raw[j].item(), 4)
        return probs

    def evaluate(self, nat_path: str, tac_path: str,
                 mode: str = DEFAULT_THRESHOLD_MODE) -> dict:
        probs = self.score_pair(nat_path, tac_path)
        return build_result(probs, self.calibrated_thresholds, mode)
