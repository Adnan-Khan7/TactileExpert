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
| Too Thick | **0.602** | 0.587 | 0.500 | 0.469 | 0.585 |
| Broken Lines | 0.612 | **0.667** | 0.571 | 0.540 | 0.630 |
| Missing Parts | 0.655 | 0.667 | **0.689** | 0.618 | 0.683 |
| Missing Texture | 0.882 | **0.916** | 0.855 | 0.848 | 0.884 |
| Extra Parts | 0.356 | 0.418 | **0.536** | 0.271 | 0.411 |
| **Macro F1** | 0.621 | **0.651** | 0.630 | 0.549 | 0.639 |

### AUC (threshold-independent)

| Dimension | CLIP Probe | VLM | ResNet-50 | ViT-B/16 | DINOv2 Probe |
|---|:---:|:---:|:---:|:---:|:---:|
| Too Thick | 0.793 | 0.824 | 0.766 | 0.756 | **0.831** |
| Broken Lines | 0.733 | **0.793** | 0.695 | 0.669 | 0.721 |
| Missing Parts | **0.745** | 0.729 | **0.745** | 0.712 | 0.722 |
| Missing Texture | **0.852** | 0.825 | 0.775 | 0.699 | 0.789 |
| Extra Parts | 0.705 | 0.629 | **0.786** | 0.620 | 0.736 |
| **Macro AUC** | **0.766** | 0.760 | 0.753 | 0.691 | 0.760 |

**Key findings:**
- VLM leads macro F1 (0.651); CLIP Probe leads macro AUC (0.766) — fine-tuned vision-language reasoning generalises well across defect types
- DINOv2 Probe (0.639 macro F1) and ResNet-50 (0.630) are competitive without language supervision
- ResNet-50 Extra Parts jumped from 0.368 → 0.536 with GPT-augmented training — the largest single gain across all models and dimensions
- GPT augmentation improved VLM F1 on four of five dimensions (macro 0.645 → 0.651), though its Extra Parts F1 regressed (0.458 → 0.418) and ResNet-50 now leads that dimension
- Missing Texture is the highest-scoring dimension across all models (F1 0.848–0.916), consistent with it being the most visually salient defect
- Calibrated val-F1-max thresholds improve precision across all models (see `evaluators/constants.py`)

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

## Using the UI

### First-time setup per session

1. Start the app (`sbatch slurm/run_pipeline.sh` on GPU, or `bash run_local.sh` for CPU testing)
2. Find the `gradio.live` public URL in the job log: `tail -f logs/pipeline_<jobid>.out`
3. Open the URL in any browser — no SSH tunnel needed

### Left panel — controls

| Control | What it does |
|---|---|
| **Mode** radio | Switch between *Full pipeline* (generate + evaluate + edit) and *Eval only* (upload any existing tactile and evaluate it) |
| **Natural reference photograph** | Upload your natural image — CLIP auto-detects the object and fills the object name field |
| **Object name** | Confirm or correct the detected name — this drives the BANA prompt |
| **BANA prompt** | Auto-generated from the object name; fully editable before generation |
| **OpenAI API key** | Entered per session, never stored |
| **Generate tactile graphic** | Calls GPT-image-1 with the reference photo + prompt, then runs all selected evaluators |
| **Run evaluators** checkboxes | Select which of the 5 models to run (all 5 recommended when saving for training) |
| **Pre-fill edit instruction from** | Which model's verdict drives the suggested edit text |
| **Edit instruction** | Auto-filled from the trusted model's top flagged issue — edit freely before applying |
| **Defect being addressed** | Dropdown: which defect this edit targets — auto-filled, change if you're overriding the model |
| **Apply edit** | Sends the instruction to GPT-image-1, re-evaluates, appends a new iteration |
| **Re-evaluate current image** *(right panel)* | Re-runs evaluators on the current image without a new GPT call — useful when switching models mid-session |

### Right panel — results

- **Evaluation table** — per-option probability bars for every selected model, with a consensus column
- **Iteration Gallery** — thumbnails of natural reference + every tactile iteration; click to expand
- **Edit Log** — every instruction you applied, plus the compressed context sent to the next API call
- **Save to Training** — approve the final image, correct the defect checkboxes, save

### Typical session flow

```
Upload natural image
  → CLIP auto-detects object → confirm name
  → Generate tactile graphic
  → Inspect image visually (ignore scores first — form your own opinion)
  → If defects visible:
      Adjust "Defect being addressed" dropdown if needed
      Edit instruction → Apply edit
      Repeat 2-4 times
  → Save to Training tab
      Review EVERY checkbox against the actual image
      Correct any wrong pre-fills
      Save
  → Reset session → next image
```

---

## Ongoing Experiments

### Motivation

Each time a human annotator uses the pipeline, they generate more than a final approved image — they generate a *trajectory*: a sequence of generate → evaluate → edit steps, where each step records what was tried, which defects were flagged, what instruction was given, and whether the edit helped. Accumulating these trajectories enables two things beyond standard supervised retraining:

1. **Reward-guided edit policy** — the evaluator score improvement at each step is a dense reward signal. A lightweight language model can be trained on these trajectories to generate edit instructions automatically, replacing the human in the loop at inference time and enabling fully automated iterative refinement.

2. **Reward model improvement** — whenever the human corrects the evaluator's pre-filled labels before saving, that correction is logged separately (`evaluator_flags` vs `human_labels`). These disagreements are training signal for the evaluators themselves, making the reward signal more reliable over time.

The goal is a self-improving flywheel: human corrections retrain the evaluators → better evaluators produce better rewards → better rewards train a stronger edit policy → stronger policy reaches all-clear in fewer iterations → less human effort per image.

### Data collection loop

Follow these steps each session. Each completed save produces one trajectory record used for both evaluator retraining and edit policy training.

**Before starting**
- Confirm the pipeline app is running (`tail -f logs/pipeline_<jobid>.out` for the gradio.live URL)
- Have a folder of natural reference images ready

**Per image — repeat for each natural image**

| Step | Action | Notes |
|---|---|---|
| 1 | Upload the natural reference photo | CLIP auto-detects the object; confirm or correct the name |
| 2 | Select **all 5 evaluators** in the sidebar | Running all models gives the best annotation coverage — defects missed by one model may be caught by another |
| 3 | Set the trusted model to whichever you trust most for this image type | DINOv2 or VLM are good defaults |
| 4 | Click **Generate tactile graphic** | Wait for generation + evaluation to complete |
| 5 | Inspect the result carefully — look at the image directly, not just the model scores | Models are imperfect; your visual judgment is ground truth |
| 6 | If defects are visible: review the pre-filled edit instruction, adjust if needed, click **Apply edit** | Aim for targeted single-issue instructions ("thin the strokes" not "fix everything") |
| 7 | Repeat steps 5–6 for **2–4 edit iterations** — don't stop at the first acceptable output | More iterations = richer trajectory = more (state, action, reward) training pairs |
| 8 | When satisfied, go to the **Save to Training** tab | |
| 9 | **Review every checkbox manually** against the actual image — do not blindly accept the pre-fill | This is the most important step: your corrections here are the ground truth labels and the signal for detecting evaluator errors |
| 10 | Click **Save to training data** | Saves the final pair to `annotations.jsonl` and the full trajectory to `trajectories.jsonl` |

**After each batch of ~10–20 images**
- Check `generated_training_data/annotations.jsonl` line count — confirm saves are landing
- Check `generated_training_data/trajectories.jsonl` — confirm trajectories have multiple steps and non-null rewards
- Note any consistent evaluator errors (same model repeatedly wrong on same defect type) — flag for threshold or retraining

### Current targets

| Milestone | Status |
|---|---|
| 30 GPT-generated pairs collected and annotated | ✅ Done |
| All 5 models retrained with GPT-augmented data | ✅ Done (Jul 2026) |
| 150 pairs collected — sufficient for edit policy training | 🔄 In progress |
| Evaluator retraining pass 2 (GPT-only baseline) | ⬜ Pending |
| Evaluator retraining pass 3 (full combined dataset) | ⬜ Pending |
| Edit policy training (supervised cloning on trajectories) | ⬜ Pending |
| Edit policy improvement via DPO (preference pairs) | ⬜ Pending |

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
