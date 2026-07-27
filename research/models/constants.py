"""Shared vocabulary and configuration for all TactileEval evaluators."""

ALL_OPTIONS = ["too_thick", "broken_lines", "missing_parts", "missing_texture"]

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

DIMENSION_LABELS = {
    "broken_lines":    "QL — Line Quality",
    "too_thick":       "QL — Line Quality",
    "missing_texture": "QT — Texture",
    "missing_parts":   "QP — Part Completeness",
}

SYSTEM_PROMPT = (
    "You are an expert evaluator of tactile graphics for blind and visually impaired learners. "
    "Tactile graphics are embossed line drawings that users explore through touch. "
    "You are shown two images: the first is a natural reference photograph and the second is "
    "a generated tactile graphic. "
    "Answer the quality assessment question with 'yes' or 'no' only."
)

# Val-calibrated thresholds from VLM Fine-tuned epoch 3
VLM_THRESHOLDS = {
    "broken_lines":    0.60,
    "missing_parts":   0.25,
    "missing_texture": 0.05,
    "too_thick":       0.05,
}

# Calibrated thresholds from CLIP Probe validation
CLIP_THRESHOLDS = {
    "too_thick":       0.75,
    "broken_lines":    0.65,
    "missing_parts":   0.50,
    "missing_texture": 0.50,
}
