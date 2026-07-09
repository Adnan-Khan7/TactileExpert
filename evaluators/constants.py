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
# Phrasing mirrors the collected human edit instructions (targeted, imperative,
# names the failure and the fix) so the pre-fill needs minimal editing.
EDIT_INSTRUCTIONS = {
    "broken_lines": (
        "Close the broken lines so every outline of the object is continuous — "
        "each element should read as one solid, unbroken line under a fingertip."
    ),
    "too_thick": (
        "Reduce the line thickness of the object so adjacent elements stay "
        "separate and do not merge or collide."
    ),
    "missing_texture": (
        "Texture each element of the object with its own distinct pattern — "
        "adjacent parts and parts at different depths must not share the same texture."
    ),
    "missing_parts": (
        "Add the missing parts of the object so every structural element "
        "visible in the reference photograph is present."
    ),
    "extra_parts": (
        "Remove the extra elements that are not present in the reference "
        "photograph, leaving only the object itself with clean lines."
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
# Weights = per-option F1 on the original 153-pair test set at t=0.50,
# measured on the DEPLOYED fleet (CLIP/DINOv2/ViT v4; ResNet/VLM v3 — see
# models/*/TRAINING_RECIPE.txt). Regenerated by
# vlm_finetune/code/experiments/eval_full_v4.py at every retrain — do not
# hand-edit. Once the GPT-domain holdout is large enough for stable
# per-option F1, switch to a balanced average of the two domains.

ENSEMBLE_NAME = "Ensemble (weighted)"

ENSEMBLE_WEIGHTS = {
    "too_thick":       {"CLIP Probe": 0.591, "VLM Fine-tuned": 0.393, "ResNet-50": 0.515, "ViT-B/16": 0.549, "DINOv2 Probe": 0.622},
    "broken_lines":    {"CLIP Probe": 0.620, "VLM Fine-tuned": 0.634, "ResNet-50": 0.571, "ViT-B/16": 0.607, "DINOv2 Probe": 0.556},
    "missing_parts":   {"CLIP Probe": 0.626, "VLM Fine-tuned": 0.614, "ResNet-50": 0.690, "ViT-B/16": 0.686, "DINOv2 Probe": 0.722},
    "missing_texture": {"CLIP Probe": 0.887, "VLM Fine-tuned": 0.861, "ResNet-50": 0.884, "ViT-B/16": 0.838, "DINOv2 Probe": 0.884},
    "extra_parts":     {"CLIP Probe": 0.432, "VLM Fine-tuned": 0.270, "ResNet-50": 0.406, "ViT-B/16": 0.290, "DINOv2 Probe": 0.405},
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
