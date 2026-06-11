"""Qwen2-VL-2B + LoRA adapter evaluator for TactileExpert."""

from pathlib import Path

import torch

from .constants import (
    ALL_OPTIONS, VLM_THRESHOLDS, QUESTIONS,
    DIMENSION_LABELS, EDIT_INSTRUCTIONS, SYSTEM_PROMPT,
)

HERE = Path(__file__).resolve().parent


class VLMEvaluator:
    """Qwen2-VL-2B-Instruct with merged LoRA adapter (vision encoder frozen)."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        lora_checkpoint: str | Path | None = None,
        device: str | None = None,
        max_pixels: int = 200704,
    ):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info

        self._process_vision_info = process_vision_info
        self.thresholds = dict(VLM_THRESHOLDS)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        lora_checkpoint = lora_checkpoint or (HERE.parent / "models" / "vlm_checkpoint")

        print(f"[VLM Evaluator] Loading {model_path}  device={self.device}")
        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map={"": self.device},
        )

        lora_path = Path(lora_checkpoint)
        if lora_path.exists():
            from peft import PeftModel
            print(f"[VLM Evaluator] Merging LoRA adapter from {lora_path}…")
            model = PeftModel.from_pretrained(model, str(lora_path))
            model = model.merge_and_unload()
        else:
            print("[VLM Evaluator] No LoRA checkpoint found — running zero-shot.")

        model.eval()
        self.model = model

        tok = self.processor.tokenizer
        self._yes_ids = [tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("Yes")]
        self._no_ids  = [tok.convert_tokens_to_ids("no"),  tok.convert_tokens_to_ids("No")]
        print("[VLM Evaluator] Ready.")

    @torch.no_grad()
    def _score_option(self, nat_path: str, tac_path: str, option: str) -> float:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{Path(nat_path).resolve()}"},
                {"type": "image", "image": f"file://{Path(tac_path).resolve()}"},
                {"type": "text",  "text": QUESTIONS[option]},
            ]},
        ]
        text   = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, _ = self._process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        logits    = self.model(**inputs).logits[0, -1, :]
        yes_logit = max(logits[self._yes_ids[0]].item(), logits[self._yes_ids[1]].item())
        no_logit  = max(logits[self._no_ids[0]].item(),  logits[self._no_ids[1]].item())
        return torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)[0].item()

    def score_pair(self, nat_path: str, tac_path: str) -> dict[str, float]:
        """Raw prob_yes per option — no threshold applied."""
        return {opt: round(self._score_option(nat_path, tac_path, opt), 4) for opt in ALL_OPTIONS}

    def evaluate(self, nat_path: str, tac_path: str) -> dict:
        per_option: dict = {}
        for opt in ALL_OPTIONS:
            prob   = self._score_option(nat_path, tac_path, opt)
            thresh = self.thresholds.get(opt, 0.5)
            per_option[opt] = {
                "prob_yes":         round(prob, 4),
                "threshold":        thresh,
                "flagged":          prob > thresh,
                "dimension":        DIMENSION_LABELS.get(opt, ""),
                "edit_instruction": EDIT_INSTRUCTIONS[opt] if prob > thresh else None,
            }
        flagged = sorted(
            [o for o, d in per_option.items() if d["flagged"]],
            key=lambda o: -per_option[o]["prob_yes"],
        )
        top = None
        if flagged:
            top = {"option": flagged[0], **per_option[flagged[0]]}
        return {"options": per_option, "flagged": flagged, "top_issue": top, "all_clear": not flagged}
