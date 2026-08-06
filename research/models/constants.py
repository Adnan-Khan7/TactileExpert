"""Compatibility shim — the real vocabulary lives in evaluators/constants.py.

C4, 2026-07-30. This file used to be an independent 4-option copy:

    ALL_OPTIONS = ["too_thick", "broken_lines", "missing_parts", "missing_texture"]

`extra_parts` was absent entirely, and the `broken_lines` question was the
pre-v3 wording rather than the outline-only text the deployed VLM was fine-tuned
on. Everything importing this — research/review/{score_pipeline,apply_labels,
prepare_inspection}.py and the research/ui/ shims — therefore worked over four
options with a stale prompt, while the deployed fleet used five. Numbers produced
through this path are not comparable to experiments_out/eval_current_fleet.json.

It now re-exports the deployed constants, so there is exactly ONE definition of
the option vocabulary in the repo and the drift cannot recur. Loaded by file path
rather than `from evaluators.constants import ...` because that executes
evaluators/__init__.py, which imports torch — unwanted for CPU-only label tooling
such as apply_labels.py.

NOTE for the label-verification pass: research/annotator/annotator.py does NOT
import from here. It carries its own list (all five options plus the deprecated
noisy_background and non_conformant), so it was never affected by this bug.
"""

import importlib.util as _ilu
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
_CONSTANTS = _REPO_ROOT / "evaluators" / "constants.py"

_spec = _ilu.spec_from_file_location("_tactile_deployed_constants", _CONSTANTS)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ALL_OPTIONS = _mod.ALL_OPTIONS
OPTION_DISPLAY = _mod.OPTION_DISPLAY
QUESTIONS = _mod.QUESTIONS
EDIT_INSTRUCTIONS = _mod.EDIT_INSTRUCTIONS
DIMENSION_LABELS = _mod.DIMENSION_LABELS
SYSTEM_PROMPT = _mod.SYSTEM_PROMPT

# Threshold names kept under their historical spellings so existing imports in
# research/models/evaluators.py and research/ui/vlm_evaluator.py keep working.
VLM_THRESHOLDS = _mod.VLM_THRESHOLDS_CALIBRATED
CLIP_THRESHOLDS = _mod.CLIP_THRESHOLDS_CALIBRATED

__all__ = [
    "ALL_OPTIONS", "OPTION_DISPLAY", "QUESTIONS", "EDIT_INSTRUCTIONS",
    "DIMENSION_LABELS", "SYSTEM_PROMPT", "VLM_THRESHOLDS", "CLIP_THRESHOLDS",
]
