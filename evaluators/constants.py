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
    "broken_lines": (
        "Does this tactile graphic contain broken or discontinuous outlines "
        "where a fingertip tracing the edges would encounter gaps, "
        "interruptions, or missing segments?"
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

EDIT_INSTRUCTIONS = {
    "broken_lines": (
        "Restore continuous outlines: regenerate with stroke settings that avoid "
        "embossing gaps, or post-process to close broken segments."
    ),
    "too_thick": (
        "Thin the strokes: reduce line width so adjacent parts remain distinct "
        "and fine details are preserved at embossed resolution."
    ),
    "missing_texture": (
        "Add texture fills: apply distinct tactile patterns to differentiate "
        "surface types and regions visible in the reference."
    ),
    "missing_parts": (
        "Restore missing parts: regenerate or edit to include all anatomical "
        "and structural elements visible in the reference photograph."
    ),
    "extra_parts": (
        "Remove hallucinated content: the tactile contains elements not present "
        "in the reference image. Regenerate keeping only the actual subject."
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

# Val-calibrated thresholds — CLIP Probe v2 (5-head, retrained June 2026)
CLIP_THRESHOLDS_CALIBRATED = {
    "too_thick":       0.60,
    "broken_lines":    0.20,
    "missing_parts":   0.50,
    "missing_texture": 0.50,
    "extra_parts":     0.50,
}

# Val-calibrated thresholds — VLM Fine-tuned v2 (Qwen2-VL-2B, epoch 4)
VLM_THRESHOLDS_CALIBRATED = {
    "too_thick":       0.15,
    "broken_lines":    0.10,
    "missing_parts":   0.15,
    "missing_texture": 0.10,
    "extra_parts":     0.20,
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

# Val-calibrated thresholds — DINOv2 ViT-L/14 probe (retrained July 2026, gamma_neg=8 for QT)
DINO_THRESHOLDS_CALIBRATED = {
    "too_thick":       0.55,
    "broken_lines":    0.60,
    "missing_parts":   0.50,
    "missing_texture": 0.50,
    "extra_parts":     0.55,
}

# Backward-compatible aliases
CLIP_THRESHOLDS = CLIP_THRESHOLDS_CALIBRATED
VLM_THRESHOLDS  = VLM_THRESHOLDS_CALIBRATED


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
