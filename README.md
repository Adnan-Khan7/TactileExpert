# TactileExpert

Human-in-the-loop pipeline for generating and evaluating tactile graphics
using GPT-image-1 and BANA-grounded quality evaluators.

![Pipeline](pipeline_diagram.png)

## Overview

TactileExpert takes a natural reference photograph and helps produce a
high-quality tactile graphic through iterative generation and evaluation.
Two AI evaluators score each candidate against seven BANA quality dimensions,
giving the user actionable feedback on what to fix before the next iteration.

## Pipeline

1. **Upload** a natural reference photograph
2. **Generate** — GPT-image-1 produces a tactile graphic candidate from a
   BANA-grounded prompt (editable per object)
3. **Evaluate** — both evaluators score the candidate on seven defect dimensions
4. **Edit** — select which evaluator verdict to trust, refine the edit
   instruction if needed, and send to GPT-image-1 edit mode
5. **Iterate** — repeat until the evaluators show no defects

API key is entered directly in the UI for each call and is never stored.

### UI Modes

| Mode | Description |
|---|---|
| **Full pipeline** | Generate + evaluate + edit via GPT-image-1 (requires API key) |
| **Eval only** | Upload any tactile graphic directly and run evaluators — no API calls |

### Threshold Modes

| Mode | Description |
|---|---|
| **Balanced** (default) | Fixed t = 0.50 for all options — all-clear state is reachable |
| **Calibrated** | Per-option val-F1-maximising thresholds |

Switch between modes with the radio toggle; raw probabilities are cached so
switching re-renders all panels without re-running the models.

## Defect Dimensions

| # | Option | Description |
|---|---|---|
| 1 | Too Thick | Lines/outlines rendered too thick or bold |
| 2 | Broken Lines | Continuous lines appear broken or interrupted |
| 3 | Missing Parts | A structural part visible in the natural image is absent |
| 4 | Missing Texture | Surface texture of the object is missing or insufficient |
| 5 | Extra Parts | Tactile contains elements not present in the natural image |
| 6 | Non-Conformant | Overall tactile does not match the reference image concept |

All options: **1 = defect present**, 0 = not present.

## Evaluators

| Model | Architecture | Macro F1 (test, 4 options) |
|---|---|---|
| CLIP Probe | CLIP ViT-L/14 + MLP heads (QL / QP / QT) | 0.676 |
| VLM Fine-tuned | Qwen2-VL-2B-Instruct + LoRA r=4 (epoch 7) | 0.541 |

Both evaluators were trained on a manually cleaned and annotated dataset of
987 natural–tactile image pairs across six object families.
Checkpoints are stored in `models/` (not tracked by git — see setup below).

## Repository Structure

```
TactileExpert/
├── evaluators/
│   ├── constants.py          options, prompts, threshold sets, result builder
│   ├── clip_probe_eval.py    CLIP ViT-L/14 + MLP heads (frozen)
│   └── vlm_eval.py           Qwen2-VL-2B + LoRA adapter (frozen)
├── generation/
│   └── gpt_generator.py      GPT-image-1 generate + edit wrappers
├── models/                   checkpoints — download separately (see below)
│   ├── clip_probe/           ql/qp/qt_evaluator.pt
│   └── vlm_checkpoint/       LoRA adapter + tokenizer files
├── app.py                    Gradio pipeline interface
├── run_local.sh              CPU-only launch on login node
├── slurm/run_pipeline.sh     SLURM job script (GPU, 4 h)
├── pipeline_diagram.png
└── requirements.txt
```

## Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Model checkpoints** are not tracked in git. Copy from the research directory:
```bash
# CLIP Probe heads (~14 MB)
cp ~/vlm_finetune/TactileEval_Context_And_Data/models/clip_probe/*.pt models/clip_probe/

# VLM LoRA adapter (~15 MB + tokenizer)
cp ~/vlm_finetune/output/best_checkpoint/* models/vlm_checkpoint/
```

The VLM evaluator also requires the Qwen2-VL-2B-Instruct base model from
HuggingFace, which is loaded at runtime from `$SCRATCH/hf_cache`.

## Running

### On Compute Canada (recommended — GPU node via SLURM)

```bash
cd ~/TactileExpert
mkdir -p logs
sbatch slurm/run_pipeline.sh
```

The job prints a public Gradio share link to `logs/pipeline_<jobid>.out`:
```bash
tail -f logs/pipeline_<jobid>.out   # look for "Running on public URL: https://..."
```

Open the link in any browser — no SSH tunnel needed.

### CPU-only (login node, for testing / Eval-only mode)

```bash
bash run_local.sh
```

VLM inference is ~50 s/pair in fp32 on CPU. Use Eval-only mode to skip
generation and test the evaluator loop quickly.

## Project

Part of the **TactileEval** research project on automated quality evaluation
of AI-generated tactile graphics for blind and visually impaired users.
