"""Shared vocabulary and configuration for TactileExpert evaluators."""

ALL_OPTIONS = ["too_thick", "broken_lines", "missing_parts", "missing_texture"]

OPTION_DISPLAY = {
    "too_thick":       "Too Thick",
    "broken_lines":    "Broken Lines",
    "missing_parts":   "Missing Parts",
    "missing_texture": "Missing Texture",
}

DIMENSION_LABELS = {
    "broken_lines":    "QL — Line Quality",
    "too_thick":       "QL — Line Quality",
    "missing_texture": "QT — Texture",
    "missing_parts":   "QP — Part Completeness",
}

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
}

SYSTEM_PROMPT = (
    "You are an expert evaluator of tactile graphics for blind and visually impaired learners. "
    "Tactile graphics are embossed line drawings that users explore through touch. "
    "You are shown two images: the first is a natural reference photograph and the second is "
    "a generated tactile graphic. "
    "Answer the quality assessment question with 'yes' or 'no' only."
)

# ---------------------------------------------------------------------------
# Decision thresholds — two modes, both reported in the June 12 status report
#
#   balanced   : fixed t = 0.50 for every option (same convention as
#                eval_baselines.py "balanced" metrics)
#   calibrated : val-calibrated F1-max thresholds (June 11 retraining)
#
# Note: the VLM calibrated thresholds for missing_parts / missing_texture are
# 0.05, which flags almost any candidate. Balanced mode is the default in the
# pipeline UI so the all-clear state is reachable; the user can switch modes.
# ---------------------------------------------------------------------------

THRESHOLD_MODES = ["balanced", "calibrated"]
DEFAULT_THRESHOLD_MODE = "balanced"

BALANCED_THRESHOLDS = {opt: 0.50 for opt in ALL_OPTIONS}

# Val-calibrated thresholds — CLIP Probe (retrained June 11, 2026)
CLIP_THRESHOLDS_CALIBRATED = {
    "too_thick":       0.85,
    "broken_lines":    0.35,
    "missing_parts":   0.70,
    "missing_texture": 0.55,
}

# Val-calibrated thresholds — VLM Fine-tuned epoch 7 (retrained June 11, 2026)
VLM_THRESHOLDS_CALIBRATED = {
    "too_thick":       0.80,
    "broken_lines":    0.30,
    "missing_parts":   0.05,
    "missing_texture": 0.05,
}

# Backwards-compatible aliases (calibrated was the only mode before June 11)
CLIP_THRESHOLDS = CLIP_THRESHOLDS_CALIBRATED
VLM_THRESHOLDS = VLM_THRESHOLDS_CALIBRATED


def build_result(probs: dict[str, float], calibrated: dict[str, float],
                 mode: str = DEFAULT_THRESHOLD_MODE) -> dict:
    """Apply a threshold mode to raw per-option probabilities.

    Works on stored probabilities, so the UI can switch modes without
    re-running inference.
    """
    thresholds = BALANCED_THRESHOLDS if mode == "balanced" else calibrated
    per_option: dict = {}
    for opt in ALL_OPTIONS:
        prob   = probs[opt]
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
