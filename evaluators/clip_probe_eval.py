"""CLIP ViT-L/14 + 3 MLP task heads evaluator for TactileExpert."""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from .constants import (
    CLIP_THRESHOLDS_CALIBRATED, DEFAULT_THRESHOLD_MODE, build_result,
)

HERE = Path(__file__).resolve().parent

# Curated category list for zero-shot object classification.
# Covers the six TactileNet object families.  Editable by the user in the UI.
OBJECT_CATEGORIES = [
    # Animals & Creatures
    "bat", "bee", "beluga whale", "bird", "butterfly", "camel", "cat", "crab",
    "dinosaur", "dog", "dolphin", "elephant", "fish", "flamingo", "frog",
    "giraffe", "gorilla", "horse", "jellyfish", "kangaroo", "lion", "lizard",
    "monkey", "octopus", "owl", "parrot", "peacock", "penguin", "rabbit",
    "shark", "snake", "spider", "tiger", "turtle", "whale", "wolf", "zebra",
    # Food, Nature & Simple Objects
    "apple", "banana", "cactus", "egg", "flower", "leaf", "mushroom",
    "pineapple", "rose", "strawberry", "sunflower", "tree", "watermelon",
    # Furniture & Structures
    "bed", "bridge", "chair", "door", "hut", "lamp", "sofa", "table",
    # Tools, Instruments & Appliances
    "camera", "guitar", "hammer", "headphones", "kettle", "microscope",
    "piano", "scissors", "telephone", "violin", "watch",
    # Vehicles & Flight Systems
    "airplane", "bicycle", "boat", "bus", "car", "helicopter", "motorcycle",
    "rocket", "ship", "submarine", "train",
    # Wearables & Accessories
    "backpack", "glasses", "hat", "purse", "shoe", "umbrella",
]


class CLIPProbeEvaluator:
    """CLIP ViT-L/14 (laion2b_s32b_b82k) backbone + three 2-layer MLP task heads."""

    _HEAD_CONFIGS = [
        ("QL", "ql_evaluator.pt", ["too_thick",   "broken_lines"]),
        ("QP", "qp_evaluator.pt", ["missing_parts"]),
        ("QT", "qt_evaluator.pt", ["missing_texture"]),
        ("QE", "qe_evaluator.pt", ["extra_parts"]),
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
                nn.Linear(ckpt.get("input_dim", 768 * 3), ckpt["hidden_dim"]),
                nn.BatchNorm1d(ckpt["hidden_dim"]),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(ckpt["hidden_dim"], len(options)),
            )
            head.load_state_dict(ckpt["state_dict"])
            head.eval().to(self.device)
            self._heads[_dim] = (head, options)
            # Prefer thresholds stored in the checkpoint over defaults.
            # Key name differs by training-script vintage: v5 checkpoints write
            # "cal_thresholds" (matching the DINOv2 probe), older ones wrote
            # "calibrated_thresholds". Reading only the latter silently fell
            # back to the stale module defaults.
            stored = ckpt.get("cal_thresholds") or ckpt.get("calibrated_thresholds") or {}
            for opt, t in stored.items():
                self.calibrated_thresholds[opt] = t

        # Precompute text embeddings for zero-shot object classification.
        # Uses the same CLIP backbone — no extra model or download.
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        prompts   = [f"a photo of a {c}" for c in OBJECT_CATEGORIES]
        with torch.no_grad():
            tokens = tokenizer(prompts).to(self.device)
            self._text_features = self._clip.encode_text(tokens)
            self._text_features /= self._text_features.norm(dim=-1, keepdim=True)

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

    @torch.no_grad()
    def classify_object(self, pil_image: Image.Image,
                        top_k: int = 1) -> list[tuple[str, float]]:
        """Zero-shot object classification using the already-loaded CLIP backbone.

        Returns a list of (category, confidence) tuples, highest confidence first.
        Runs in ~10 ms — text embeddings are precomputed at init time.
        """
        img   = self._preprocess(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        feat  = self._clip.encode_image(img)
        feat  = feat / feat.norm(dim=-1, keepdim=True)
        sims  = (feat @ self._text_features.T).squeeze(0)
        probs = sims.softmax(dim=-1)
        top   = probs.topk(min(top_k, len(OBJECT_CATEGORIES)))
        return [(OBJECT_CATEGORIES[i], round(v.item(), 3))
                for v, i in zip(top.values, top.indices)]
