# TactileExpert

Human-in-the-loop pipeline for generating and evaluating tactile graphics
using GPT-image-1 and five BANA-grounded quality evaluators.

![Pipeline](pipeline_diagram.png)

## Overview

TactileExpert takes a natural reference photograph and produces a high-quality
tactile graphic through iterative AI generation and multi-model evaluation.
Five evaluators score each candidate against five BANA quality dimensions,
giving the user actionable feedback on what to fix before the next iteration.

## Pipeline

1. **Upload** a natural reference photograph — the object is auto-detected
   and the object name pre-filled (CLIP zero-shot, ~10 ms)
2. **Generate** — GPT-image-1 produces a tactile graphic candidate from a
   BANA-grounded prompt (fully editable)
3. **Evaluate** — all four evaluators score the candidate on six defect
   dimensions; results shown as a per-option table with a consensus column
4. **Edit** — select which evaluator verdict to trust, refine the edit
   instruction if needed; previous corrections are automatically summarised
   and injected as context (GPT-image-1 has no text memory across calls)
5. **Iterate** — repeat until all selected models show no defects
6. **Save** — approve and save the final pair directly to the training dataset

API key is entered in the UI for each call and is never stored.

### UI Modes

| Mode | Description |
|---|---|
| **Full pipeline** | Generate + evaluate + edit via GPT-image-1 (requires API key) |
| **Eval only** | Upload any tactile graphic and run evaluators — no API calls |

### Threshold Modes

| Mode | Description |
|---|---|
| **Balanced** (default) | Fixed t = 0.50 for all options — all-clear state is reachable |
| **Calibrated** | Per-option val-F1-maximising thresholds (see table below) |

## Defect Dimensions (5 options)

| # | Option | Supervisor label | Description |
|---|---|---|---|
| 1 | Too Thick | QL — Line Quality | Lines/outlines rendered too thick or bold |
| 2 | Broken Lines | QL — Line Quality | Continuous lines appear broken or interrupted |
| 3 | Missing Parts | QP — Part Completeness | A structural part visible in the natural image is absent |
| 4 | Missing Texture | QT — Texture | Surface texture of the object is missing or insufficient |
| 5 | Extra Parts | QE — Extra/Hallucinated Content | Tactile contains elements not in the natural image |

All options: **1 = defect present**, 0 = not present.

## Evaluation Results

All models trained on a manually-annotated dataset of **987 natural–tactile
image pairs** (human-verified, 5 options) + **30 GPT-generated pairs**.
Evaluated on a held-out test set of **153 pairs** with thresholds calibrated on val.

### F1 Score (balanced threshold t = 0.50)

| Option | CLIP Probe | VLM (Qwen2-VL) | ResNet-50 | ViT-B/16 | **DINOv2 Probe** |
|---|:---:|:---:|:---:|:---:|:---:|
| Too Thick | 0.581 | 0.553 | 0.500 | 0.613 | **0.667** |
| Broken Lines | 0.605 | **0.647** | 0.457 | 0.552 | 0.585 |
| Missing Parts | 0.659 | 0.662 | **0.700** | 0.595 | 0.714 |
| Missing Texture | 0.888 | **0.904** | 0.839 | 0.849 | 0.881 |
| Extra Parts | 0.405 | **0.458** | 0.368 | 0.286 | 0.323 |
| **Macro F1** | 0.628 | **0.645** | 0.573 | 0.579 | 0.634 |

### AUC (threshold-independent)

| Option | CLIP Probe | VLM (Qwen2-VL) | ResNet-50 | ViT-B/16 | **DINOv2 Probe** |
|---|:---:|:---:|:---:|:---:|:---:|
| Too Thick | 0.800 | 0.833 | 0.788 | 0.852 | **0.838** |
| Broken Lines | **0.761** | 0.782 | 0.664 | 0.714 | 0.723 |
| Missing Parts | 0.736 | 0.735 | 0.761 | 0.727 | **0.755** |
| Missing Texture | 0.817 | **0.844** | 0.722 | 0.766 | 0.736 |
| Extra Parts | 0.725 | 0.683 | **0.755** | 0.622 | 0.682 |
| **Macro AUC** | **0.768** | 0.775 | 0.738 | 0.736 | 0.747 |

**Key findings:**
- VLM leads on macro F1 (0.645) and macro AUC (0.775)
- **DINOv2 Probe leads on Too Thick (F1 0.667, AUC 0.838)** — self-supervised structural features outperform language-aligned CLIP on line geometry
- DINOv2 matches CLIP on macro F1 (0.634 vs 0.628) without any language supervision
- ViT-B/16 achieves best AUC on Too Thick (0.852) among the original four models
- Calibrated val-F1-max thresholds improve all models; see `evaluators/constants.py`

## Dataset

- **987 natural–tactile image pairs** across 6 object families
  (Animals, Food/Nature, Furniture, Tools, Vehicles, Wearables)
- **100% human-annotated** (two annotators + flight review pass, June 2026)
- **37 reference-quality pairs** — all 6 defect options absent (generator targets)
- **6 defect dimensions** annotated (noisy_background collected but excluded
  from training: only 22/987 positives)

| Split | Pairs | Records (6 options) |
|---|---|---|
| Train | 688 | 4,128 |
| Val | 146 | 876 |
| Test | 153 | 918 |

## Repository Structure

```
TactileExpert/
├── evaluators/
│   ├── constants.py          options, prompts, all threshold sets, build_result()
│   ├── clip_probe_eval.py    CLIP ViT-L/14 + 4 MLP heads (QL/QP/QT/QE)
│   ├── backbone_eval.py      ResNet-50 and ViT-B/16 two-stream evaluators
│   ├── vlm_eval.py           Qwen2-VL-2B-Instruct + LoRA adapter
│   └── dino_probe_eval.py    DINOv2 ViT-L/14 (frozen) + 4 MLP heads
├── generation/
│   └── gpt_generator.py      GPT-image-1 generate + edit wrappers
├── models/                   checkpoints — not tracked in git (see setup)
│   ├── clip_probe/           ql/qp/qt/qe_evaluator.pt
│   ├── vlm_checkpoint/       LoRA adapter + tokenizer files
│   ├── resnet50/             resnet50_evaluator.pt
│   ├── vit_b16/              vit_b_16_evaluator.pt
│   └── dino_probe/           ql/qp/qt/qe_evaluator.pt
├── app.py                    Gradio pipeline interface
├── run_local.sh              CPU-only launch (login node / testing)
├── slurm/run_pipeline.sh     SLURM job script (GPU, 8 h)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Model checkpoints are not tracked in git (binary files, ~100 MB total).
Copy from the research directory on Nibi:

```bash
# CLIP Probe (4 heads)
cp ~/vlm_finetune/TactileEval_Context_And_Data/models/clip_probe_v2/*.pt models/clip_probe/

# VLM LoRA adapter
cp ~/vlm_finetune/output_v2/best_checkpoint/* models/vlm_checkpoint/

# ResNet-50
cp ~/vlm_finetune/TactileEval_Context_And_Data/models/resnet50_v2/resnet50_evaluator.pt models/resnet50/

# ViT-B/16
cp ~/vlm_finetune/TactileEval_Context_And_Data/models/vitb16_v2/vit_b_16_evaluator.pt models/vit_b16/

# DINOv2 Probe (4 heads — DINOv2 ViT-L/14 weights downloaded automatically via torch.hub)
cp ~/vlm_finetune/TactileEval_Context_And_Data/models/dino_probe_v2/{ql,qp,qt,qe}_evaluator.pt models/dino_probe/
```

The VLM requires `Qwen/Qwen2-VL-2B-Instruct` from HuggingFace (loaded at runtime
from `$SCRATCH/hf_cache`).

## Running

### On Compute Canada — GPU node (SLURM)

```bash
cd ~/TactileExpert && mkdir -p logs
sbatch slurm/run_pipeline.sh
tail -f logs/pipeline_<jobid>.out   # look for the gradio.live URL
```

Open the public URL in any browser — no SSH tunnel needed.

### CPU-only (login node, for testing / Eval-only mode)

```bash
bash run_local.sh
```

VLM inference is ~50 s/pair in fp32 on CPU. Use Eval-only mode to test
the evaluation loop quickly without waiting for VLM.

## Project

Part of the **TactileEval** research project on automated quality evaluation
of AI-generated tactile graphics for blind and visually impaired users.
