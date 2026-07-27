"""
TactileEval model evaluators — single source of truth for all inference wrappers.

Classes
-------
CLIPProbeEvaluator   CLIP ViT-L/14 backbone + 3 task-specific MLP heads
TwoStreamEvaluator   ResNet-50 or ViT-B/16 fine-tuned two-stream
VLMEvaluator         Qwen2-VL-2B with optional LoRA adapter

All classes implement:
    .evaluate(nat_path, tac_path, ...) -> dict
    .score_pair(nat_path, tac_path)    -> dict[option, float]  (raw probs only)

The score_pair interface is used by the scoring pipeline for batch processing.
The evaluate interface (with flagged/top_issue) is used by the Gradio demo.
"""

from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image

from .constants import (
    ALL_OPTIONS, QUESTIONS, DIMENSION_LABELS, EDIT_INSTRUCTIONS,
    SYSTEM_PROMPT, VLM_THRESHOLDS, CLIP_THRESHOLDS,
)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_result(per_option: dict) -> dict:
    flagged = sorted(
        [o for o, d in per_option.items() if d["flagged"]],
        key=lambda o: -per_option[o]["prob_yes"],
    )
    top = None
    if flagged:
        t = flagged[0]
        top = {"option": t, **per_option[t]}
    return {"options": per_option, "flagged": flagged, "top_issue": top, "all_clear": not flagged}


# ---------------------------------------------------------------------------
# CLIP Probe
# ---------------------------------------------------------------------------

class CLIPProbeEvaluator:
    """CLIP ViT-L/14 (laion2b_s32b_b82k) backbone + three 2-layer MLP task heads."""

    _HEAD_CONFIGS = [
        ("QL", "ql_evaluator.pt", ["too_thick", "broken_lines"]),
        ("QP", "qp_evaluator.pt", ["missing_parts"]),
        ("QT", "qt_evaluator.pt", ["missing_texture"]),
    ]

    def __init__(self, models_dir: str | Path, device: str | None = None):
        import open_clip

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        models_dir = Path(models_dir)

        print("[CLIPProbeEvaluator] Loading CLIP ViT-L/14 (laion2b_s32b_b82k)…")
        self._clip, self._preprocess, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k", device=self.device)
        self._clip.eval()

        self._heads: dict[str, tuple[nn.Module, list[str]]] = {}
        self._thresholds: dict[str, float] = {}

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
            cal = ckpt.get("calibrated_thresholds", {})
            for opt in options:
                self._thresholds[opt] = cal.get(opt, CLIP_THRESHOLDS.get(opt, 0.5))

        print("[CLIPProbeEvaluator] Ready.")

    @torch.no_grad()
    def _encode(self, path: str) -> torch.Tensor:
        img = self._preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
        f = self._clip.encode_image(img)
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _combined_feat(self, nat_path: str, tac_path: str) -> torch.Tensor:
        nat_f = self._encode(nat_path)
        tac_f = self._encode(tac_path)
        return torch.cat([nat_f, tac_f, nat_f - tac_f], dim=-1)

    def score_pair(self, nat_path: str, tac_path: str) -> dict[str, float]:
        """Returns raw prob_yes per option (no threshold applied)."""
        combined = self._combined_feat(nat_path, tac_path)
        probs: dict[str, float] = {}
        for _dim, (head, options) in self._heads.items():
            raw = torch.sigmoid(head(combined))[0]
            for j, opt in enumerate(options):
                probs[opt] = round(raw[j].item(), 4)
        return probs

    def evaluate(self, nat_path: str, tac_path: str, progress_fn=None) -> dict:
        combined = self._combined_feat(nat_path, tac_path)
        per_option: dict = {}
        step = 0
        for _dim, (head, options) in self._heads.items():
            probs = torch.sigmoid(head(combined))[0]
            for j, opt in enumerate(options):
                if progress_fn:
                    progress_fn(opt, step, len(ALL_OPTIONS))
                prob   = probs[j].item()
                thresh = self._thresholds.get(opt, 0.5)
                per_option[opt] = {
                    "prob_yes":         round(prob, 4),
                    "threshold":        thresh,
                    "flagged":          prob > thresh,
                    "dimension":        DIMENSION_LABELS.get(opt, ""),
                    "edit_instruction": EDIT_INSTRUCTIONS[opt] if prob > thresh else None,
                }
                step += 1
        return _build_result(per_option)


# ---------------------------------------------------------------------------
# ResNet-50 / ViT-B/16 two-stream
# ---------------------------------------------------------------------------

class TwoStreamEvaluator:
    """ResNet-50 or ViT-B/16 fine-tuned two-stream (shared backbone + task-head MLP)."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        import torchvision.models as tvm
        import torchvision.transforms as T

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._transform = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        backbone_name = ckpt.get("backbone", "resnet50")
        print(f"[TwoStreamEvaluator] Loading {backbone_name} from {Path(checkpoint_path).name}…")

        if backbone_name == "resnet50":
            base = tvm.resnet50()
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
        else:  # vit_b_16
            base = tvm.vit_b_16()
            feat_dim = base.heads.head.in_features
            base.heads = nn.Identity()

        cdim = feat_dim * 3

        def _head(n_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(cdim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_out))

        class _Net(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.backbone = base
                self_.heads = nn.ModuleDict({"QL": _head(2), "QP": _head(1), "QT": _head(1)})

            def forward(self_, nat, tac):
                fn = self_.backbone(nat)
                ft = self_.backbone(tac)
                x  = torch.cat([fn, ft, fn - ft], dim=-1)
                return torch.cat(
                    [self_.heads["QL"](x), self_.heads["QP"](x), self_.heads["QT"](x)], dim=-1)

        self._model = _Net()
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval().to(self.device)
        self._thresholds: dict[str, float] = ckpt.get(
            "calibrated_thresholds", {opt: 0.5 for opt in ALL_OPTIONS})
        print("[TwoStreamEvaluator] Ready.")

    @torch.no_grad()
    def _forward(self, nat_path: str, tac_path: str) -> torch.Tensor:
        nat = self._transform(Image.open(nat_path).convert("RGB")).unsqueeze(0).to(self.device)
        tac = self._transform(Image.open(tac_path).convert("RGB")).unsqueeze(0).to(self.device)
        return torch.sigmoid(self._model(nat, tac))[0]  # (4,)

    def score_pair(self, nat_path: str, tac_path: str) -> dict[str, float]:
        """Returns raw prob_yes per option (no threshold applied)."""
        probs = self._forward(nat_path, tac_path)
        return {opt: round(probs[i].item(), 4) for i, opt in enumerate(ALL_OPTIONS)}

    def evaluate(self, nat_path: str, tac_path: str, progress_fn=None) -> dict:
        probs = self._forward(nat_path, tac_path)
        per_option: dict = {}
        for i, opt in enumerate(ALL_OPTIONS):
            if progress_fn:
                progress_fn(opt, i, len(ALL_OPTIONS))
            prob   = probs[i].item()
            thresh = self._thresholds.get(opt, 0.5)
            per_option[opt] = {
                "prob_yes":         round(prob, 4),
                "threshold":        thresh,
                "flagged":          prob > thresh,
                "dimension":        DIMENSION_LABELS.get(opt, ""),
                "edit_instruction": EDIT_INSTRUCTIONS[opt] if prob > thresh else None,
            }
        return _build_result(per_option)


# ---------------------------------------------------------------------------
# VLM (Qwen2-VL-2B + optional LoRA)
# ---------------------------------------------------------------------------

class VLMEvaluator:
    """
    Qwen2-VL-2B with optional LoRA adapter.
    lora_checkpoint=None → zero-shot (base model only).
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        lora_checkpoint: str | None = None,
        device: str | None = None,
        max_pixels: int = 200704,
        thresholds: dict | None = None,
    ):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info

        self._process_vision_info = process_vision_info
        self.thresholds = thresholds or VLM_THRESHOLDS
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"[VLMEvaluator] device={self.device}  lora={'yes' if lora_checkpoint else 'no (zero-shot)'}")

        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map={"": self.device},
        )

        if lora_checkpoint is not None:
            from peft import PeftModel
            lora_path = Path(lora_checkpoint)
            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
            print(f"[VLMEvaluator] Loading LoRA adapter from {lora_path}...")
            model = PeftModel.from_pretrained(model, str(lora_path))
            model = model.merge_and_unload()

        model.eval()
        self.model = model

        tok = self.processor.tokenizer
        self._yes_ids = [tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("Yes")]
        self._no_ids  = [tok.convert_tokens_to_ids("no"),  tok.convert_tokens_to_ids("No")]
        print("[VLMEvaluator] Ready.")

    @torch.no_grad()
    def _score_option(self, natural_path: str, tactile_path: str, option: str) -> float:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{Path(natural_path).resolve()}"},
                {"type": "image", "image": f"file://{Path(tactile_path).resolve()}"},
                {"type": "text",  "text": QUESTIONS[option]},
            ]},
        ]
        text   = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, _ = self._process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        outputs  = self.model(**inputs)
        logits   = outputs.logits[0, -1, :]
        yes_logit = max(logits[self._yes_ids[0]].item(), logits[self._yes_ids[1]].item())
        no_logit  = max(logits[self._no_ids[0]].item(),  logits[self._no_ids[1]].item())
        return torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)[0].item()

    def score_pair(self, nat_path: str, tac_path: str) -> dict[str, float]:
        """Returns raw prob_yes per option (no threshold applied). Runs 4 forward passes."""
        return {opt: round(self._score_option(nat_path, tac_path, opt), 4) for opt in ALL_OPTIONS}

    def evaluate(
        self,
        natural_path: str,
        tactile_path: str,
        options: list[str] | None = None,
        progress_fn=None,
    ) -> dict:
        if options is None:
            options = list(QUESTIONS.keys())
        per_option = {}
        for idx, opt in enumerate(options):
            if progress_fn:
                progress_fn(opt, idx, len(options))
            prob   = self._score_option(natural_path, tactile_path, opt)
            thresh = self.thresholds.get(opt, 0.5)
            per_option[opt] = {
                "prob_yes":         round(prob, 4),
                "threshold":        thresh,
                "flagged":          prob > thresh,
                "dimension":        DIMENSION_LABELS.get(opt, ""),
                "edit_instruction": EDIT_INSTRUCTIONS[opt] if prob > thresh else None,
            }
        flagged_opts = sorted(
            [o for o, d in per_option.items() if d["flagged"]],
            key=lambda o: -per_option[o]["prob_yes"],
        )
        top_issue = None
        if flagged_opts:
            top = flagged_opts[0]
            top_issue = {"option": top, **per_option[top]}
        return {
            "options":   per_option,
            "flagged":   flagged_opts,
            "top_issue": top_issue,
            "all_clear": len(flagged_opts) == 0,
        }
