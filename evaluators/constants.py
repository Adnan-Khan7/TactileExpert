"""Shared vocabulary and configuration for TactileExpert evaluators.

v2 — June 2026: expanded from 4 to 6 options after full manual annotation
pass (987 pairs).  All four model families retrained on new splits_v2.
"""

ALL_OPTIONS = [
    "too_thick",
    "broken_lines",
    "missing_parts",
    "missing_texture",
    "extra_parts",
]

OPTION_DISPLAY = {
    "too_thick":       "Too Thick",
    "broken_lines":    "Broken Lines",
    "missing_parts":   "Missing Parts",
    "missing_texture": "Missing Texture",
    "extra_parts":     "Extra Parts",
}

DIMENSION_LABELS = {
    "too_thick":       "QL — Line Quality",
    "broken_lines":    "QL — Line Quality",
    "missing_texture": "QT — Texture",
    "missing_parts":   "QP — Part Completeness",
    "extra_parts":     "QE — Extra/Hallucinated Content",
}

# non_conformant removed per advisor feedback — too subjective, confuses models.

QUESTIONS = {
    # v3: outline-only definition — MUST match the question the deployed VLM
    # checkpoint was fine-tuned with (code/vlm/build_vqa_v2.py).
    "broken_lines": (
        "Consider only the structural outlines of the object, not any texture "
        "or fill patterns inside its regions. Are the outlines broken or "
        "discontinuous, so that a fingertip tracing an edge would encounter "
        "gaps, interruptions, or missing segments?"
    ),
    "too_thick": (
        "Does this tactile graphic contain strokes that are excessively thick, "
        "causing adjacent parts to merge or fine details to collapse into "
        "solid areas?"
    ),
    "missing_texture": (
        "Does this tactile graphic omit tactile texture fills that are present "
        "in the reference image to distinguish different surfaces, materials, "
        "or regions?"
    ),
    "missing_parts": (
        "Does this tactile graphic omit any anatomical or structural parts "
        "that are clearly present in the reference photograph?"
    ),
    "extra_parts": (
        "Comparing this tactile graphic to the reference photograph, does the "
        "tactile drawing contain invented structures, hallucinated appendages, "
        "or decorative additions that are not present in the reference?"
    ),
}

# Templates say "the object"; the app substitutes the known object name at
# prefill time ("the object" → "the giraffe"), so keep that exact wording.
EDIT_INSTRUCTIONS = {
    "broken_lines": (
        "Restore continuous outlines: close the broken segments so every line "
        "of the object is continuous, with no embossing gaps."
    ),
    "too_thick": (
        "Thin the strokes: reduce line width so adjacent parts of the object "
        "remain distinct and fine details are preserved at embossed resolution."
    ),
    "missing_texture": (
        "Add texture fills: apply distinct tactile patterns to differentiate "
        "the object's surface regions visible in the reference."
    ),
    "missing_parts": (
        "Restore missing parts: include all anatomical and structural elements "
        "of the object visible in the reference photograph."
    ),
    "extra_parts": (
        "Remove hallucinated content: the tactile contains elements not present "
        "in the reference image. Keep only the object itself."
    ),
}

SYSTEM_PROMPT = (
    "You are an expert evaluator of tactile graphics for blind and visually impaired learners. "
    "Tactile graphics are embossed line drawings that users explore through touch. "
    "You are shown two images: the first is a natural reference photograph and the second is "
    "a generated tactile graphic. "
    "Answer the quality assessment question with 'yes' or 'no' only."
)

# ── Threshold configuration ────────────────────────────────────────────────────
# Two modes available in the UI:
#   balanced   — fixed t = 0.50 for every option (all-clear state reachable)
#   calibrated — per-option val-F1-maximising thresholds (June 2026 retraining)

THRESHOLD_MODES = ["balanced", "calibrated"]
DEFAULT_THRESHOLD_MODE = "balanced"

BALANCED_THRESHOLDS = {opt: 0.50 for opt in ALL_OPTIONS}

# Val-calibrated thresholds — CLIP Probe v3 (trajectory-collection GPT data)
CLIP_THRESHOLDS_CALIBRATED = {
    "too_thick":       0.45,
    "broken_lines":    0.45,
    "missing_parts":   0.50,
    "missing_texture": 0.50,
    "extra_parts":     0.50,
}

# Val-calibrated thresholds — VLM Fine-tuned v3 (Qwen2-VL-2B, reworded
# broken_lines question, epoch 4)
VLM_THRESHOLDS_CALIBRATED = {
    "too_thick":       0.05,
    "broken_lines":    0.25,
    "missing_parts":   0.15,
    "missing_texture": 0.05,
    "extra_parts":     0.15,
}

# Val-calibrated thresholds — ResNet-50 v2 (retrained June 2026)
RESNET_THRESHOLDS_CALIBRATED = {
    "too_thick":       0.20,
    "broken_lines":    0.15,
    "missing_parts":   0.35,
    "missing_texture": 0.30,
    "extra_parts":     0.45,
}

# Val-calibrated thresholds — ViT-B/16 v2 (retrained June 2026)
VIT_THRESHOLDS_CALIBRATED = {
    "too_thick":       0.05,
    "broken_lines":    0.10,
    "missing_parts":   0.40,
    "missing_texture": 0.10,
    "extra_parts":     0.85,
}

# Val-calibrated thresholds — DINOv2 ViT-L/14 probe v3 (trajectory-collection GPT data)
DINO_THRESHOLDS_CALIBRATED = {
    "too_thick":       0.75,
    "broken_lines":    0.50,
    "missing_parts":   0.50,
    "missing_texture": 0.40,
    "extra_parts":     0.50,
}

# Backward-compatible aliases
CLIP_THRESHOLDS = CLIP_THRESHOLDS_CALIBRATED
VLM_THRESHOLDS  = VLM_THRESHOLDS_CALIBRATED


# ── Weighted per-option ensemble ──────────────────────────────────────────────
# Weights = per-option F1 on the original 153-pair test set, v3 checkpoints
# (balanced t=0.50). Regenerate at every retrain — do not hand-edit. Once the
# GPT-domain holdout (gpt_test.jsonl) is large enough for stable per-option
# F1, switch to a balanced average of the two domains.

ENSEMBLE_NAME = "Ensemble (weighted)"

ENSEMBLE_WEIGHTS = {
    #                  CLIP   VLM    ResNet ViT    DINOv2
    "too_thick":       {"CLIP Probe": 0.588, "VLM Fine-tuned": 0.577, "ResNet-50": 0.521, "ViT-B/16": 0.614, "DINOv2 Probe": 0.625},
    "broken_lines":    {"CLIP Probe": 0.581, "VLM Fine-tuned": 0.651, "ResNet-50": 0.571, "ViT-B/16": 0.520, "DINOv2 Probe": 0.569},
    "missing_parts":   {"CLIP Probe": 0.642, "VLM Fine-tuned": 0.632, "ResNet-50": 0.690, "ViT-B/16": 0.636, "DINOv2 Probe": 0.722},
    "missing_texture": {"CLIP Probe": 0.884, "VLM Fine-tuned": 0.908, "ResNet-50": 0.884, "ViT-B/16": 0.814, "DINOv2 Probe": 0.898},
    "extra_parts":     {"CLIP Probe": 0.395, "VLM Fine-tuned": 0.486, "ResNet-50": 0.406, "ViT-B/16": 0.242, "DINOv2 Probe": 0.242},
}


def build_ensemble_result(model_results: dict, mode: str = DEFAULT_THRESHOLD_MODE) -> dict:
    """Combine per-model build_result outputs into an ensemble result.

    Flag  = weighted vote: each model votes its own flag (at its threshold for
            the current mode), votes weighted by per-option F1, flag if the
            weighted vote share exceeds 0.5.
    Prob  = weighted mean of per-model prob_yes (drives the UI bar).
    Weights are renormalised over the models actually present, so the ensemble
    degrades gracefully when a subset of evaluators is selected.
    """
    per_option: dict = {}
    for opt in ALL_OPTIONS:
        entries = [(m, r["options"][opt]) for m, r in model_results.items()
                   if m in ENSEMBLE_WEIGHTS[opt] and opt in r["options"]]
        if not entries:
            continue
        total_w = sum(ENSEMBLE_WEIGHTS[opt][m] for m, _ in entries)
        prob = sum(ENSEMBLE_WEIGHTS[opt][m] * d["prob_yes"] for m, d in entries) / total_w
        vote = sum(ENSEMBLE_WEIGHTS[opt][m] for m, d in entries if d["flagged"]) / total_w
        flag = vote > 0.5
        n_agree = sum(1 for _, d in entries if d["flagged"])
        per_option[opt] = {
            "prob_yes":         round(prob, 4),
            "threshold":        0.5,   # majority of weighted vote
            "flagged":          flag,
            "vote_share":       round(vote, 3),
            "n_agree":          n_agree,
            "n_models":         len(entries),
            "dimension":        DIMENSION_LABELS.get(opt, ""),
            "edit_instruction": EDIT_INSTRUCTIONS[opt] if flag else None,
        }
    flagged = sorted(
        [o for o, d in per_option.items() if d["flagged"]],
        key=lambda o: -per_option[o]["prob_yes"],
    )
    top = {"option": flagged[0], **per_option[flagged[0]]} if flagged else None
    return {
        "options":   per_option,
        "flagged":   flagged,
        "top_issue": top,
        "all_clear": not flagged,
        "mode":      mode,
    }


def build_result(probs: dict[str, float], calibrated: dict[str, float],
                 mode: str = DEFAULT_THRESHOLD_MODE) -> dict:
    """Apply a threshold mode to raw per-option probabilities."""
    thresholds = BALANCED_THRESHOLDS if mode == "balanced" else calibrated
    per_option: dict = {}
    for opt in ALL_OPTIONS:
        prob   = probs.get(opt, 0.0)
        thresh = thresholds.get(opt, 0.5)
        flag   = prob > thresh
        per_option[opt] = {
            "prob_yes":         round(prob, 4),
            "threshold":        thresh,
            "flagged":          flag,
            "dimension":        DIMENSION_LABELS.get(opt, ""),
            "edit_instruction": EDIT_INSTRUCTIONS[opt] if flag else None,
        }
    flagged = sorted(
        [o for o, d in per_option.items() if d["flagged"]],
        key=lambda o: -per_option[o]["prob_yes"],
    )
    top = {"option": flagged[0], **per_option[flagged[0]]} if flagged else None
    return {
        "options":   per_option,
        "flagged":   flagged,
        "top_issue": top,
        "all_clear": not flagged,
        "mode":      mode,
    }
