"""CLIP ViT-L/14 + 3 MLP task heads evaluator for TactileExpert."""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from .constants import (
    CLIP_THRESHOLDS_CALIBRATED, DEFAULT_THRESHOLD_MODE, build_result,
)

HERE = Path(__file__).resolve().parent


class CLIPProbeEvaluator:
    """CLIP ViT-L/14 (laion2b_s32b_b82k) backbone + three 2-layer MLP task heads."""

    _HEAD_CONFIGS = [
        ("QL", "ql_evaluator.pt", ["too_thick", "broken_lines"]),
        ("QP", "qp_evaluator.pt", ["missing_parts"]),
        ("QT", "qt_evaluator.pt", ["missing_texture"]),
    ]

    def __init__(
        self,
        models_dir: str | Path | None = None,
        device: str | None = None,
    ):
        import open_clip

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        models_dir = Path(models_dir) if models_dir else HERE.parent / "models" / "clip_probe"

        print("[CLIP Probe] Loading ViT-L/14 (laion2b_s32b_b82k)…")
        # Use the val transform (Resize + CenterCrop) — the train transform
        # contains RandomResizedCrop, which makes inference nondeterministic.
        self._clip, _, self._preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k", device=self.device)
        self._clip.eval()

        self._heads: dict[str, tuple[nn.Module, list[str]]] = {}
        self.calibrated_thresholds: dict[str, float] = dict(CLIP_THRESHOLDS_CALIBRATED)

        for _dim, pt_name, options in self._HEAD_CONFIGS:
            ckpt = torch.load(models_dir / pt_name, map_location=self.device, weights_only=False)
            head = nn.Sequential(
                nn.Linear(ckpt["input_dim"], ckpt["hidden_dim"]),
                nn.BatchNorm1d(ckpt["hidden_dim"]),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(ckpt["hidden_dim"], len(options)),
            )
            head.load_state_dict(ckpt["state_dict"])
            head.eval().to(self.device)
            self._heads[_dim] = (head, options)
            # Prefer thresholds stored in the checkpoint over defaults
            for opt, t in ckpt.get("calibrated_thresholds", {}).items():
                self.calibrated_thresholds[opt] = t

        print("[CLIP Probe] Ready.")

    @torch.no_grad()
    def _encode(self, path: str) -> torch.Tensor:
        img = self._preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
        f = self._clip.encode_image(img)
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _feat(self, nat_path: str, tac_path: str) -> torch.Tensor:
        n = self._encode(nat_path)
        t = self._encode(tac_path)
        return torch.cat([n, t, n - t], dim=-1)

    def score_pair(self, nat_path: str, tac_path: str) -> dict[str, float]:
        """Raw prob_yes per option — no threshold applied."""
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
