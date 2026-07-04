# TactileExpert

**Human-in-the-loop pipeline for generating and evaluating tactile graphics for blind and visually impaired learners.**

Tactile graphics — embossed line drawings explored through touch — are essential learning materials for BVI students, yet remain costly and expert-dependent to produce. TactileExpert is a human-in-the-loop system that generates BANA-compliant tactile graphics from natural reference photographs using GPT-image-1, evaluates candidate quality across five defect dimensions with an ensemble of five fine-tuned models, and enables iterative refinement with automatic edit-history injection. Approved pairs are saved directly to a training dataset, closing the loop between generation and model improvement.

![Pipeline Diagram](pipeline_diagram.png)

---

## Research Contributions

- **Annotated dataset** — 987 natural–tactile image pairs across six object categories, 100% human-annotated for five BANA-grounded defect dimensions (+30 GPT-generated pairs for training augmentation)
- **Five evaluator models** — CLIP Probe, fine-tuned VLM (Qwen2-VL-2B + LoRA), ResNet-50 two-stream, ViT-B/16 two-stream, and DINOv2 ViT-L/14 Probe; all trained on the same annotated splits
- **End-to-end pipeline** — reference photo → GPT-image-1 generation → multi-model evaluation → iterative refinement → training-data collection, in a single Gradio interface

---

## Pipeline

| Step | Action |
|---|---|
| **1. Upload** | Natural reference photograph; CLIP zero-shot classifies the object and pre-fills the object name field (~10 ms) |
| **2. Generate** | GPT-image-1 generates a tactile graphic candidate using a BANA-grounded prompt; the reference photo is sent directly to the API |
| **3. Evaluate** | Five models independently score the candidate across five defect dimensions; results shown as per-option probability bars with consensus column |
| **4. Edit** | Select which model to trust; prior corrections are compressed and injected as context (GPT-image-1 has no text memory across calls) |
| **5. Iterate** | Repeat evaluate → edit until all selected models show no defects |
| **6. Save** | Approve the final pair; image files and defect labels saved to the training dataset in JSONL format |

The API key is entered in the UI per session and is never stored or logged.

### UI Modes

| Mode | Description |
|---|---|
| **Full pipeline** | Generate + evaluate + edit via GPT-image-1 (requires OpenAI API key) |
| **Eval only** | Upload any existing tactile graphic and run the evaluators without API calls |

### Threshold Modes

| Mode | Description |
|---|---|
| **Balanced** (default) | Fixed t = 0.50 for all options; all-clear state is reachable |
| **Calibrated** | Per-option val-F1-maximising thresholds (see `evaluators/constants.py`) |

---

## Defect Dimensions

Five dimensions defined per BANA guidelines, each treated as an independent binary classification problem:

| # | Dimension | Label | Description |
|---|---|---|---|
| 1 | **Too Thick** | QL | Lines or outlines rendered too thick; adjacent parts merge or fine detail collapses |
| 2 | **Broken Lines** | QL | Continuous outlines appear broken or interrupted; fingertip encounters gaps |
| 3 | **Missing Parts** | QP | A structural part visible in the reference photograph is absent from the tactile |
| 4 | **Missing Texture** | QT | Surface texture needed to distinguish regions or materials is missing |
| 5 | **Extra Parts** | QE | Tactile contains hallucinated elements not present in the reference photograph |

All labels: **1 = defect present**, 0 = absent.

---

## Dataset

| Property | Value |
|---|---|
| Total pairs | 987 natural–tactile image pairs |
| Annotation | 100% human-annotated (two annotators + flight review, June 2026) |
| Object categories | Animals, Food/Nature, Furniture, Tools, Vehicles, Wearables |
| Reference-quality pairs | 37 pairs with all five defect options absent |
| GPT-augmented pairs | +30 GPT-image-1 generated pairs (included in training) |

| Split | Pairs | Option-centric records |
|---|---|---|
| Train | 718 + 30 GPT | 4,278 |
| Val | 146 | 876 |
| Test | 153 | 918 |

---

## Evaluator Models

Five models, all trained on the same `splits_v2` dataset:

| Model | Architecture | Notes |
|---|---|---|
| **CLIP Probe** | Frozen CLIP ViT-L/14 + 4 two-layer MLP heads (QL/QP/QT/QE) | Language-aligned features; 768-d CLS → 1536-d [nat\|tac] |
| **VLM** | Qwen2-VL-2B-Instruct + LoRA fine-tuning | Vision-language model; answers BANA-grounded yes/no questions |
| **ResNet-50** | Two-stream ResNet-50; late fusion | Spatial features; separate encoders for natural and tactile |
| **ViT-B/16** | Two-stream ViT-B/16; late fusion | Transformer features; separate encoders for natural and tactile |
| **DINOv2 Probe** | Frozen DINOv2 ViT-L/14 + 4 two-layer MLP heads (QL/QP/QT/QE) | Self-supervised structural features; 1024-d CLS → 3072-d [nat\|tac\|nat−tac] |

All probe/two-stream models are trained with **Asymmetric Loss** (γ⁻=4.0; γ⁻=8.0 for the DINOv2 Missing Texture head to suppress false positives).

---

## Results

Test set: **153 pairs**, balanced threshold t = 0.50. Models trained on splits_v2 including 30 GPT-augmented pairs.

### F1 Score

| Dimension | CLIP Probe | VLM | ResNet-50 | ViT-B/16 | DINOv2 Probe |
|---|:---:|:---:|:---:|:---:|:---:|
| Too Thick | 0.581 | 0.553 | 0.500 | 0.613 | 0.585 |
| Broken Lines | 0.605 | **0.647** | 0.457 | 0.552 | 0.630 |
| Missing Parts | 0.659 | 0.662 | **0.700** | 0.595 | 0.683 |
| Missing Texture | 0.888 | **0.904** | 0.839 | 0.849 | 0.884 |
| Extra Parts | 0.405 | **0.458** | 0.368 | 0.286 | 0.411 |
| **Macro F1** | 0.628 | **0.645** | 0.573 | 0.579 | 0.639 |

### AUC (threshold-independent)

| Dimension | CLIP Probe | VLM | ResNet-50 | ViT-B/16 | DINOv2 Probe |
|---|:---:|:---:|:---:|:---:|:---:|
| Too Thick | 0.800 | 0.833 | 0.788 | **0.852** | 0.831 |
| Broken Lines | **0.761** | 0.782 | 0.664 | 0.714 | 0.721 |
| Missing Parts | 0.736 | 0.735 | 0.761 | 0.727 | 0.722 |
| Missing Texture | 0.817 | **0.844** | 0.722 | 0.766 | 0.789 |
| Extra Parts | 0.725 | 0.683 | **0.755** | 0.622 | 0.736 |
| **Macro AUC** | 0.768 | **0.775** | 0.738 | 0.736 | 0.760 |

**Key findings:**
- VLM leads on macro F1 (0.645) — fine-tuned vision-language reasoning benefits generalisation
- DINOv2 Probe macro F1 (0.639) matches CLIP (0.628) without any language supervision; self-supervised structural features are competitive for line-art quality assessment
- Extra Parts improved most with GPT-augmented training data (+88% relative, 0.323 → 0.411 F1) — GPT-generated graphics systematically over-generate, providing useful training signal
- Calibrated val-F1-max thresholds improve precision across all models (see `evaluators/constants.py`)

> CLIP, VLM, ResNet-50, and ViT-B/16 numbers are from the pre-augmentation run; GPT-augmented retraining for these models is in progress.

---

## Repository Structure

```
TactileExpert/
├── app.py                        Gradio pipeline UI
├── evaluators/
│   ├── constants.py              ALL_OPTIONS, prompts, threshold sets, build_result()
│   ├── clip_probe_eval.py        CLIP ViT-L/14 + 4 MLP heads
│   ├── backbone_eval.py          ResNet-50 and ViT-B/16 two-stream evaluators
│   ├── vlm_eval.py               Qwen2-VL-2B-Instruct + LoRA adapter
│   └── dino_probe_eval.py        DINOv2 ViT-L/14 (frozen) + 4 MLP heads
├── generation/
│   └── gpt_generator.py          GPT-image-1 generate and edit wrappers
├── models/                       Model checkpoints (not tracked in git)
│   ├── clip_probe/               ql / qp / qt / qe _evaluator.pt
│   ├── vlm_checkpoint/           LoRA adapter + tokenizer files
│   ├── resnet50/                 resnet50_evaluator.pt
│   ├── vit_b16/                  vit_b_16_evaluator.pt
│   └── dino_probe/               ql / qp / qt / qe _evaluator.pt
├── project_notes/                Research notes and to-do lists
├── pipeline_diagram.png          Pipeline overview figure
├── run_local.sh                  CPU-only launch script
├── slurm/run_pipeline.sh         SLURM GPU job script
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Model checkpoints are not tracked in git. Place them under `models/` following the structure above. DINOv2 ViT-L/14 backbone weights are downloaded automatically via `torch.hub` on first run.

The VLM requires `Qwen/Qwen2-VL-2B-Instruct` from HuggingFace (set `HF_HOME` to a local cache directory).

### Running

```bash
# GPU (recommended for VLM inference)
sbatch slurm/run_pipeline.sh
# → follow the gradio.live public URL printed in the job log

# CPU-only (for testing / Eval-only mode)
bash run_local.sh
```

---

## Citation

If you use TactileExpert or the annotated dataset in your work, please cite:

```bibtex
@misc{tactileexpert2026,
  author = {Khan, Adnan and Majid, Majid},
  title  = {TactileExpert: Human-in-the-Loop Generation and Quality Evaluation of Tactile Graphics},
  year   = {2026},
  url    = {https://github.com/Adnan-Khan7/TactileExpert}
}
```
