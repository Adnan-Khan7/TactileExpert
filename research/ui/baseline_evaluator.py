"""
Backward-compatibility shim — real implementations live in code/models/.
Importing from here continues to work for app.py and precompute.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from models.evaluators import CLIPProbeEvaluator, TwoStreamEvaluator  # noqa: F401
from models.constants import ALL_OPTIONS as _ALL_OPTIONS               # noqa: F401

__all__ = ["CLIPProbeEvaluator", "TwoStreamEvaluator"]
