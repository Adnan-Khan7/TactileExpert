"""ResNet-50 and ViT-B/16 two-stream evaluators for TactileExpert.

Both share the same architecture (backbone + 4 task heads covering the
5 options) and the same wrapper class — only the checkpoint path differs.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

from .constants import (
    ALL_OPTIONS, RESNET_THRESHOLDS_CALIBRATED, VIT_THRESHOLDS_CALIBRATED,
    DEFAULT_THRESHOLD_MODE, build_result,
)

HERE = Path(__file__).resolve().parent

TASK_HEADS = {
    "QL": ["too_thick",   "broken_lines"],
    "QP": ["missing_parts"],
    "QT": ["missing_texture"],
    "QE": ["extra_parts"],
}
# Concatenation order of head outputs — must expand to exactly ALL_OPTIONS.
HEAD_ORDER = ["QL", "QP", "QT", "QE"]

_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]),
])


def _build_model(backbone_name: str) -> nn.Module:
    if backbone_name == "resnet50":
        base = tvm.resnet50(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
    elif backbone_name == "vit_b_16":
        base = tvm.vit_b_16(weights=None)
        feat_dim = base.heads.head.in_features
        base.heads = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    combined = feat_dim * 3

    def head(n):
        return nn.Sequential(
            nn.Linear(combined, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, n))

    class TactileEvaluator(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = base
            self.heads = nn.ModuleDict(
                {d: head(len(opts)) for d, opts in TASK_HEADS.items()})

        def forward(self, nat, tac):
            fn = self.backbone(nat)
            ft = self.backbone(tac)
            feat = torch.cat([fn, ft, fn - ft], dim=-1)
            return torch.cat([self.heads[d](feat) for d in HEAD_ORDER], dim=-1)

    return TactileEvaluator()


class BackboneEvaluator:
    """Shared wrapper for ResNet-50 and ViT-B/16 two-stream evaluators."""

    def __init__(self, backbone_name: str,
                 checkpoint: str | Path | None = None,
                 device: str | None = None):
        self.backbone_name = backbone_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        default_dirs = {
            "resnet50":  HERE.parent / "models" / "resnet50",
            "vit_b_16":  HERE.parent / "models" / "vit_b16",
        }
        ckpt_path = Path(checkpoint) if checkpoint else \
                    default_dirs[backbone_name] / f"{backbone_name}_evaluator.pt"

        cal_defaults = {
            "resnet50":  RESNET_THRESHOLDS_CALIBRATED,
            "vit_b_16":  VIT_THRESHOLDS_CALIBRATED,
        }
        self.calibrated_thresholds = dict(cal_defaults[backbone_name])

        print(f"[{backbone_name}] Loading from {ckpt_path.name}…")
        ckpt  = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model = _build_model(backbone_name)
        # v2/v3 checkpoints carry a QN (non_conformant) head — option removed
        # from the taxonomy, so its weights are dropped here.
        state = {k: v for k, v in ckpt["state_dict"].items()
                 if not k.startswith("heads.QN")}
        model.load_state_dict(state)
        model.eval().to(self.device)
        self._model = model
        print(f"[{backbone_name}] Ready.")

    @torch.no_grad()
    def score_pair(self, nat_path: str, tac_path: str) -> dict[str, float]:
        nat = _TRANSFORM(Image.open(nat_path).convert("RGB")).unsqueeze(0).to(self.device)
        tac = _TRANSFORM(Image.open(tac_path).convert("RGB")).unsqueeze(0).to(self.device)
        raw = torch.sigmoid(self._model(nat, tac))[0].cpu().tolist()
        return {opt: round(raw[i], 4) for i, opt in enumerate(ALL_OPTIONS)}

    def evaluate(self, nat_path: str, tac_path: str,
                 mode: str = DEFAULT_THRESHOLD_MODE) -> dict:
        probs = self.score_pair(nat_path, tac_path)
        return build_result(probs, self.calibrated_thresholds, mode)


class ResNetEvaluator(BackboneEvaluator):
    def __init__(self, **kwargs):
        super().__init__("resnet50", **kwargs)


class ViTEvaluator(BackboneEvaluator):
    def __init__(self, **kwargs):
        super().__init__("vit_b_16", **kwargs)
