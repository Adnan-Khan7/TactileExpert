# research/

Training, evaluation and experiment scripts. Everything the deployed app in
`../evaluators/` was produced by.

These scripts are versioned here; the data they operate on is not.

## The two roots

| anchor | default | holds |
|---|---|---|
| `REPO_ROOT` | this checkout | deployed code, `models/` checkpoints, `generated_training_data/` |
| `DATA_ROOT` | `~/vlm_finetune` | `data/` splits, `images/` corpus, `experiments_out/`, training run outputs |

`DATA_ROOT` is overridable:

```bash
export TACTILE_DATA_ROOT=/path/to/workspace
python research/experiments/rederive_fleet.py
```

Both are resolved in [`_paths.py`](_paths.py). No script under `research/`
hardcodes a home directory, which is what lets the same code run inside a
container with the workspace mounted as a volume.

The workspace is kept out of the repo on purpose: the image corpus is large,
and the human annotations are consent-restricted (AMT / CUREB) — code and
model cards are releasable, the labelled pairs are not.

## Layout

```
baselines/     CLIP · DINOv2 · ResNet-50 · ViT-B/16 probe training + eval
vlm/           VQA dataset build (build_vqa_v2.py) + Qwen2-VL LoRA fine-tuning
experiments/   deploy gates, ensemble-weight derivation, calibration, zero-shot panel
policy/        BC / DPO edit-policy data extraction from trajectories
annotator/     Gradio labelling tool, split builder, CSV import/export
review/  ui/   earlier review + demo stack (imports the local models/ package)
models/        constants + evaluator classes used by review/ and ui/ ONLY
```

### A note on `models/`

`research/models/` is a **fork** of `../evaluators/`, carried by the older
`review/` and `ui/` stacks. It has drifted from the deployed code and is not
what the app runs. Treat `../evaluators/` as authoritative; do not fix a bug
here and assume it reached production.

(`.gitignore` root-anchors `/models/` so the repo's weights directory stays
untracked without also dropping this package from a clone.)

## Running on the cluster

Shell wrappers set both anchors and `cd` into the workspace before invoking
the Python entry point, so SLURM `--output=logs/...` paths still resolve
relative to the workspace:

```bash
cd "$TACTILE_DATA_ROOT"          # logs/ lives here
sbatch ~/TactileExpert/research/experiments/rederive_fleet.sh
```

Override either anchor per-job with `TACTILE_REPO` / `TACTILE_DATA_ROOT`.

## Known issues

Verified defects in this tree are tracked in the audit punch-list in
`project_notes/SESSION_NOTES_JULY24.md` (not published). The load-bearing
ones: `extract_policy_data.py` applies current-fleet calibrators to raw
scores logged by earlier fleets, and `rederive_fleet.py` derives ensemble
weights as raw accuracy on a ~5%-positive set.
