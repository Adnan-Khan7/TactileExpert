"""Path anchors for the research scripts.

These scripts live in the git repo but read and write the research workspace
(image corpus, splits, checkpoints, experiment outputs), which stays outside
the repo — it is large and partly consent-restricted (AMT/CUREB), so it must
never be committed or baked into a container image.

    REPO_ROOT  the TactileExpert checkout (this file's grandparent).
               Deployed code: app.py, evaluators/, generation/, models/.

    DATA_ROOT  the research workspace. Defaults to ~/vlm_finetune; override
               with the TACTILE_DATA_ROOT environment variable. Holds
               data/, images/ (symlink to $SCRATCH), experiments_out/,
               output_*/, TactileEval_Context_And_Data/.

Usage from a script at research/<subdir>/<name>.py:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import DATA_ROOT, REPO_ROOT

Nothing here hardcodes a home directory, so the same scripts run unchanged
in a container with TACTILE_DATA_ROOT pointed at a mounted volume.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(
    os.environ.get("TACTILE_DATA_ROOT", Path.home() / "vlm_finetune")
).expanduser().resolve()


def require_data_root() -> Path:
    """Fail loudly, not silently, when the research workspace is absent.

    Without this a missing DATA_ROOT surfaces as an empty split or a
    FileNotFoundError three call frames deep.
    """
    if not DATA_ROOT.is_dir():
        raise SystemExit(
            f"research workspace not found: {DATA_ROOT}\n"
            f"Set TACTILE_DATA_ROOT to the directory holding data/, images/ "
            f"and experiments_out/ (default: ~/vlm_finetune)."
        )
    return DATA_ROOT
