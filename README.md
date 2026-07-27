# TactileExpert

**A human-in-the-loop system that generates tactile graphics for blind and visually impaired (BVI) learners, judges its own output with a panel of trained evaluator models, and turns every human correction into training data.**

![Pipeline Diagram](pipeline_diagram.png)

---

## 1. Motivation

Tactile graphics — embossed line drawings that BVI students read with their fingertips — are essential learning materials, but making one is slow and needs an expert. Image generators like GPT-image-1 can draft them in seconds, but the drafts have defects a sighted designer would catch instantly: lines too thick to tell apart, gaps in outlines, missing parts, missing texture, invented extras.

So the question this project answers is: **can we teach models to catch those defects, so a human only has to make the final call — and can every human correction make the system smarter?**

The system has three jobs:

1. **Generate** a tactile graphic from a photo, following BANA (Braille Authority of North America) design rules.
2. **Evaluate** it with five independently trained models — each scoring the same five defect dimensions — plus an ensemble that combines their votes.
3. **Learn** from the human: every edit instruction, every corrected label, and every accepted or rejected suggestion is logged as training data.

---

## 2. The pipeline, step by step

| Step | What happens |
|---|---|
| **1. Upload** | User uploads a natural reference photo. CLIP zero-shot identifies the object and pre-fills its name (~10 ms). |
| **2. Generate** | GPT-image-1 draws a tactile graphic candidate from the photo, guided by a BANA-grounded prompt (plain white background, continuous bold outlines, distinct texture fills, no text or logos, no invented elements). |
| **3. Evaluate** | The evaluator panel scores the candidate on five defect dimensions. The UI shows the **ensemble verdict card** — issues flagged, ordered by priority, each with a suggested edit — with the full per-model decision table one click away. |
| **4. Edit** | The top-priority suggestion pre-fills the edit box (naming the actual object, e.g. *"close the broken lines so every outline of the motorcycle is continuous"*). The user adjusts or rewrites it, picks which defect the edit targets, and applies. GPT-image-1 edits the image; the panel re-scores it. |
| **5. Iterate** | Repeat inspect → edit until the image is right. Every step is logged: scores before and after, the instruction, the target defect, and whether the suggestion was accepted or rewritten. |
| **6. Save** | The user reviews five defect checkboxes against the final image (the ground truth), and saves. One save writes the image pair, the final labels, and the full step-by-step trajectory. |

The OpenAI API key is entered in the UI per session and never stored.

---

## 3. The judges: five models + one ensemble

All five models answer the same five questions about a (photo, tactile) pair: **Too Thick? Broken Lines? Missing Parts? Missing Texture? Extra Parts?** Each answer is a probability.

| Model | Backbone | Trainable part | Input → Output |
|---|---|---|---|
| **CLIP Probe** | CLIP ViT-L/14 (frozen) | Per-dimension 2-layer MLP heads | 2 images → 768-d CLS each → **[nat \| tac \| nat−tac] = 2304-d** → head → P(defect) |
| **DINOv2 Probe** | DINOv2 ViT-L/14 (frozen) | Per-dimension 2-layer MLP heads | 2 images → 1024-d CLS each → **[nat \| tac \| nat−tac] = 3072-d** → head → P(defect) |
| **ResNet-50** | Two-stream ResNet-50 | Entire network | 2 × (224×224×3) → 2048-d per stream → late fusion → sigmoid per dimension |
| **ViT-B/16** | Two-stream ViT-B/16 | Entire network | 2 × (224×224×3) → 768-d per stream → late fusion → sigmoid per dimension |
| **VLM** | Qwen2-VL-7B-Instruct | LoRA adapter (r=4, α=8) on q/k/v/o projections — **2.52 M of 8.29 B params (0.030%)**, vision encoder frozen | 2 images (≤200,704 px each) + a yes/no question per dimension → P("yes") from the yes/no token logits |
| **Ensemble** | — (derived) | — | Per dimension: each model votes at t = 0.50; votes weighted by that model's per-dimension accuracy on collected live data; **flag if weighted vote share > 0.5**. Probability bar = weighted mean. |

Training details shared by the probe / two-stream models: Asymmetric Loss (γ⁻ = 4.0; γ⁻ = 8.0 on the DINOv2 Missing-Texture head). The VLM is fine-tuned on question-answer pairs built from the same splits; its Broken-Lines question was reworded to an outline-only definition (*"consider only the structural outlines, not texture fills…"*) after we found the earlier phrasing conflated texture with breakage.

Each checkpoint folder under `models/` carries a `TRAINING_RECIPE.txt` stating exactly what data trained it and its headline results.

> **On the ensemble weights.** They are per-dimension accuracies measured on collected live data, and they currently span a narrow range (0.63–0.94). Over all 32 possible flag patterns across five models, the weighted vote produces **the same decision as an unweighted 3-of-5 majority in every case** — the weights change the displayed probability bar but never a verdict. Treat the ensemble as majority voting until the weighting is redesigned; see [§6](#6-what-the-data-has-taught-us-so-far-open-problems-honestly).

---

## 4. Training data

- **987 human-annotated pairs** across six object categories — every pair labeled by a human for all five defect dimensions. These form `splits_v2`: **966 train / 146 val / 153 test** pairs.
- **154 GPT-generated pairs** collected through the pipeline itself, each with checkbox-verified final labels, contributing 894 option-rows to the training split (5,022 rows total).
- **Inferred labels from edit actions**: when the user targets a defect with an edit, the pre-edit image is labeled positive for that defect — the action itself is the label.
- **A frozen GPT-domain holdout** — every 5th collected pair, never trained on: **30 pairs / 150 flag decisions**. This is what measures whether the models are learning the generated-image domain rather than the original corpus.

---

## 5. Results

Deployed fleet: CLIP / DINOv2 / ResNet-50 / ViT-B/16 **v5** + **Qwen2-VL-7B** LoRA, per-model selection documented in `models/*/TRAINING_RECIPE.txt`. Every number below is measured on **this exact serving fleet** at a uniform threshold t = 0.50, with the ensemble computed through the shipping code path. Reproduce with `python research/experiments/score_deployed_fleet.py` (CPU); raw per-pair scores in `experiments_out/eval_current_fleet.json`.

### Original test set — 153 human-annotated pairs, 765 decisions

| Dimension | CLIP Probe | VLM 7B | ResNet-50 | ViT-B/16 | DINOv2 Probe | Ensemble |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Too Thick | 0.549 | 0.438 | 0.532 | 0.513 | **0.648** | 0.630 |
| Broken Lines | 0.583 | 0.609 | **0.649** | 0.627 | 0.548 | 0.643 |
| Missing Parts | 0.693 | 0.689 | 0.707 | 0.593 | 0.679 | **0.720** |
| Missing Texture | 0.885 | 0.879 | 0.885 | 0.864 | **0.893** | 0.889 |
| Extra Parts | 0.359 | 0.372 | 0.337 | 0.306 | 0.389 | **0.394** |
| **Macro F1** | 0.614 | 0.597 | 0.622 | 0.581 | 0.632 | **0.655** |
| **Macro AUC** | 0.754 | 0.731 | 0.755 | 0.719 | 0.753 | **0.807** |

On the original corpus the ensemble is the strongest judge — best macro F1 and, by a clear margin, best macro AUC (0.807 vs 0.755 for the best single model). The AUC gap is the more meaningful of the two: averaging five probability estimates produces a better-*ranked* score than any member, even though, as noted in §3, the weighting never changes a flag decision.

### GPT-domain holdout — 30 generated pairs, 150 decisions

This is the domain the live app actually runs on: images produced by the pipeline itself, never trained on. **Only 8 of the 150 labels are positive**, so it is primarily a false-alarm test.

| Judge | Correct decisions | |
|---|:---:|---|
| *always answer "clean"* | *142/150 (94.7%)* | *trivial baseline* |
| **VLM 7B** | **132/150 (88.0%)** | best trained judge |
| Ensemble | 117/150 (78.0%) | |
| CLIP Probe | 114/150 (76.0%) | |
| ResNet-50 | 110/150 (73.3%) | |
| ViT-B/16 | 100/150 (66.7%) | |
| DINOv2 Probe | 92/150 (61.3%) | |

**Read these two tables together — they disagree, and that is the finding.** The ensemble wins on the original corpus and loses on generated images, where the single fine-tuned VLM is 15 decisions better. And no judge, the ensemble included, beats the trivial always-clean baseline: every model still over-flags this domain, the VLM least. Because the weighted vote is decision-identical to a 3-of-5 majority (§3), the panel inherits its members' false alarms instead of cancelling them, and four of the five members over-flag more than the VLM does. A judge is only worth adding to a vote if its errors are uncorrelated with the others'; here they are not.

This is why the ensemble remains the default suggestion source in the UI but the VLM is the honest single-model recommendation on generated images — and why redesigning the combiner is the top open item in §6.

### Zero-shot baselines — why fine-tuning is the whole story

The same questions asked of untrained general-purpose models, on 48 collected pairs (240 decisions):

| Zero-shot judge | Correct | False alarms |
|---|:---:|:---:|
| *always answer "clean"* | *226/240* | *0* |
| CLIP ViT-L/14 (prompt matching) | 149/240 | 83 |
| **Qwen2-VL-7B** (untrained) | **119/240** | 114 |
| Qwen2-VL-2B (untrained) | 51/240 | 188 |

Every zero-shot generalist scores far below the trivial baseline — they raise alarms on nearly everything. The third row is the same architecture now deployed as our best judge: **Qwen2-VL-7B goes from 119/240 zero-shot to the strongest judge in the fleet after a LoRA adapter touching 0.030% of its parameters.** Off-the-shelf models cannot do this task; domain training is what makes it work.

---

## 6. What the data has taught us so far (open problems, honestly)

- **Dashed textures get mistaken for broken lines.** A duck whose body is filled with neat rows of tactile dashes — good design — sets off the "broken lines" alarm in most models. They learned "many short line fragments ≈ broken" instead of "gaps in the outline ≈ broken". Rewording the question at inference time made it *worse* (telling a model to ignore dashes makes it stare at the dashes); retraining with the corrected definition helps but hasn't fully fixed it.
- **Real outline gaps sometimes get missed** by the same models that over-flag textures — the concept is genuinely mislearned, not just miscalibrated. Fixing it needs contrast examples: textured-but-clean images labeled clean, genuinely-broken images labeled broken. Our collection produces exactly these.
- **"Extra parts" is a meaning question, not a pixels question.** Whether a shape *belongs* depends on knowing what the object is. It remains the weakest dimension for every model in the fleet.
- **Evaluators over-flag any new drawing style until they've seen it.** Every retrain on more collected data improves the GPT-domain holdout; the gap is data, not architecture.
- **The ensemble is not yet earning its keep.** Weighting by accuracy on a set that is ~5% positive rewards a model for staying quiet, which compresses the weights until the vote degenerates into a plain majority (see §3). A majority of five judges that all over-flag inherits their false alarms rather than cancelling them, which is why the single best judge currently beats the panel on the holdout. Balanced accuracy or per-dimension F1 weighting is the obvious next thing to try.
- **The fine-tuned VLM is prompted slightly off-distribution.** Three of the five inference questions and the system prompt differ in wording from the text the LoRA was trained on. Every number in §5 is measured with the deployed inference wording, so they are honest — but aligning the two is untested upside, not a fix for a broken measurement.

---

## 7. Ongoing work — the reinforcement-learning story

Every session in the pipeline quietly produces the three ingredients of reinforcement learning:

- **State** — the current image pair and its evaluator scores.
- **Action** — the edit instruction the human actually gave, plus which defect it targeted.
- **Reward** — how much the defect scores improved after the edit.

On top of the final labels, we log two kinds of *disagreement*, and each one trains a different part of the system:

1. **Human vs. evaluators** (corrected checkboxes) → retrains the evaluators. They are our *reward model*, and the human keeps them honest.
2. **Human vs. suggestion** (the pre-filled edit vs. what the user actually typed) → preference pairs for training an *edit policy* — a model that learns to write the right edit instruction itself. Suggested-but-rejected = worse; human-written = better. This is exactly the data format DPO (Direct Preference Optimization) trains on.

Collected so far: **157 behaviour-cloning records** (one per human edit step) and **57 preference pairs**, with an effective suggestion-acceptance rate of **36.7%**.

The flywheel: human corrections → better evaluators → better suggestions → richer trajectories → an edit policy that needs fewer human corrections. The long-term goal is a system that iterates to a clean tactile graphic on its own, with the human only auditing.

**Next steps:** redesign the ensemble combiner so the weights actually discriminate · align the VLM's inference prompts with its training prompts and re-gate · re-score the trajectory corpus with the current fleet so policy rewards are single-fleet · behaviour-clone the edit policy, then DPO on the preference pairs.

---

## 8. Using the UI

### Start

```bash
sbatch slurm/run_pipeline.sh          # GPU (recommended)
tail -f logs/pipeline_<jobid>.out     # → open the gradio.live URL
bash run_local.sh                     # CPU-only fallback
```

### Controls (left panel)

| Control | What it does |
|---|---|
| **Mode** radio | *Full pipeline* (generate + evaluate + edit) or *Eval only* (upload an existing tactile) |
| **Natural reference photograph** | CLIP auto-detects the object and fills the name field |
| **Object name** | Confirm/correct — drives the BANA prompt *and* the object-grounded edit suggestions |
| **OpenAI API key** | Per session, never stored |
| **Run evaluators** | Which models to run (keep all 5 + Ensemble when saving for training) |
| **Pre-fill edit instruction from** | Which judge's verdict drives the suggestion (default: Ensemble) |
| **Combine all flagged issues in pre-fill** | Off by default — one targeted edit at a time; on = numbered list of every flagged issue |
| **Defect being addressed** | Which defect this edit targets — auto-filled, **change it whenever you override the suggestion** |
| **Re-evaluate current image** | Re-score without a new GPT call |

All decisions use a uniform threshold of t = 0.50. A per-dimension "calibrated" threshold mode exists in the code but is disabled in the UI: its thresholds predate the current fleet and would mis-flag.

### Results (right panel)

- **Ensemble verdict card** — issues by priority with suggested edits; expand *"Show full per-model decision table"* for every model's probability bars and the consensus column
- **Iteration gallery + edit log** — every version and every instruction
- **Save to Training** — review every checkbox against the image (your eyes are ground truth), save once

### Per-image flow

```
Upload photo → confirm name → Generate
  → look at the IMAGE first, scores second
  → edit 2–4 times (one defect per edit; flip the dropdown when overriding)
  → Save to Training: verify every checkbox → Save → Reset → next image
```

---

## 9. Repository structure

```
TactileExpert/
├── app.py                        Gradio pipeline UI (ensemble card, trajectory logging)
├── evaluators/                   ── the deployed fleet ──────────────────────
│   ├── constants.py              options, questions, thresholds, ensemble weights + rules
│   ├── clip_probe_eval.py        CLIP ViT-L/14 probe
│   ├── dino_probe_eval.py        DINOv2 ViT-L/14 probe
│   ├── backbone_eval.py          ResNet-50 / ViT-B/16 two-stream
│   └── vlm_eval.py               Qwen2-VL-7B + LoRA
├── generation/gpt_generator.py   GPT-image-1 generate/edit wrappers (BANA prompt)
├── research/                     ── training / eval / experiments ───────────
│   ├── _paths.py                 REPO_ROOT + DATA_ROOT resolution (see below)
│   ├── baselines/                CLIP · DINOv2 · ResNet · ViT probe training
│   ├── vlm/                      VQA dataset build + Qwen2-VL LoRA fine-tuning
│   ├── experiments/              deploy gates, fleet scoring, ensemble weights, calibration
│   ├── policy/                   BC / DPO edit-policy data extraction
│   ├── annotator/                labelling tool + split builder
│   └── review/, ui/, models/     earlier review + demo stack
├── models/                       checkpoints (untracked) + TRAINING_RECIPE.txt per model
├── generated_training_data/      annotations.jsonl · trajectories.jsonl · pair folders (untracked)
├── pipeline_diagram.png
├── slurm/run_pipeline.sh         GPU job script
└── run_local.sh                  CPU-only launch
```

**Code lives here; the corpus does not.** `research/` scripts read and write a
research workspace — image corpus, splits, checkpoints, experiment outputs —
that is deliberately outside the repo: it is large, and the human annotations
are consent-restricted. Its location comes from `TACTILE_DATA_ROOT`
(default `~/vlm_finetune`), resolved in `research/_paths.py`. Nothing under
`research/` hardcodes a home directory, so the same scripts run unchanged
against a mounted volume in a container. See [research/README.md](research/README.md).

Setup: `pip install -r requirements.txt`; checkpoints go under `models/`; the VLM base model downloads from HuggingFace (`Qwen/Qwen2-VL-7B-Instruct`, set `HF_HOME`).
