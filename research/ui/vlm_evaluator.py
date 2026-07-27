"""
Backward-compatibility shim — real implementations live in code/models/.
Importing from here continues to work for app.py and precompute.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from models.constants import (                                   # noqa: F401
    QUESTIONS, DIMENSION_LABELS, EDIT_INSTRUCTIONS,
    SYSTEM_PROMPT, VLM_THRESHOLDS as DEFAULT_THRESHOLDS,
)
from models.evaluators import VLMEvaluator                       # noqa: F401

__all__ = [
    "QUESTIONS", "DIMENSION_LABELS", "EDIT_INSTRUCTIONS",
    "SYSTEM_PROMPT", "DEFAULT_THRESHOLDS", "VLMEvaluator",
]
