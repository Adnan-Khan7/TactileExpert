# TactileExpert / TactileEval

Automatic quality evaluation for **tactile graphics** — raised-line images for
blind and low-vision readers. A photograph is turned into a tactile rendering,
a fleet of judges scores that rendering for defects, and an expert edits it
until it is clean.

**The pipeline.** `photograph → generated tactile rendering → five-judge
evaluation → expert edit → re-score`.

**What is judged.** Every (photograph, rendering) pair is scored on five
independent binary defects, grounded in the BANA Guidelines and Standards for
Tactile Graphics: `too_thick`, `broken_lines`, `missing_parts`,
`missing_texture`, `extra_parts`. They are not mutually exclusive — a pair can
carry any number of them.

**The judges.** Five models: a CLIP probe, a DINOv2 probe, ResNet-50, ViT-B/16,
and a Qwen2-VL-7B with a LoRA adapter.

**How results are reported.**

- Every model is trained at **12 random seeds** and reported as mean ± 2σ.
  Seeds matter: ResNet macro AUC read 0.782 at 5 seeds and 0.777 at 12, so a
  low-seed number is not stable enough to compare against.
- **AUC is the primary metric.** F1 at a fixed threshold rewards a judge that
  simply flags less often, which at these base rates looks better than it is.
- Results are always **per option**. A single accuracy figure is degenerate
  here — answering "clean" to everything already scores about 65%.
- Every results table is stamped with the **dataset** it was measured on.
  Numbers from different dataset states are not comparable.

---

# Status — 2026-08-21

## Action points

| # | Action point | Status |
|---|---|---|
| 1 | Finish reviewing/relabelling the training set so the models can be re-evaluated | **Completed.** Dataset audit in **Table 1**; evaluator results in **Table 2** (AUC) and **Table 3** (F1). |
| 2 | Present results in a fixed, standardised table format with the same headers across meetings, per-model AUC and F1 on the test set | **Incorporated.** See Tables 2, 3, 5 and 6 — identical headers, one table per metric. Please advise if the format needs changing. |
| 3 | Look into literature on defect detection and histology / patch-based classification for fine-grained line continuity | **Done.** See [Approaches tried](#approaches-tried). |
| 4 | Generate synthetic positives with clear, visible line discontinuities by VLM/API deliberate corruption | **Done — 425 generated, 218 reviewed and kept.** Dataset audit in **Table 4**; results in **Table 5** and **Table 6** (pending). |

---

## Dataset 1 — full relabelled dataset

**1,350 pairs · 6,750 decisions · 100% human-verified.**

### Table 1 — Dataset audit

| Option | train + / − | val + / − | test + / − |
|---|---|---|---|
| `too_thick` | 259 / 715 | 32 / 151 | 31 / 162 |
| `broken_lines` | 183 / 791 | 36 / 147 | 39 / 154 |
| `missing_parts` | 369 / 605 | 63 / 120 | 71 / 122 |
| `missing_texture` | 617 / 357 | 131 / 52 | 136 / 57 |
| `extra_parts` | 361 / 613 | 48 / 135 | 53 / 140 |
| **Pairs** | **974** | **183** | **193** |

### Table 2 — AUC, test set

**Dataset 1.** 193 pairs, 965 decisions. 12 seeds, mean ± 2σ.

| Model | Macro AUC | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.826 ±0.018** | 0.794 ±0.057 | **0.780 ±0.057** | 0.817 ±0.038 | **0.963 ±0.016** | **0.776 ±0.036** | 12 |
| CLIP Probe | 0.790 ±0.014 | 0.704 ±0.023 | 0.737 ±0.019 | **0.832 ±0.037** | 0.914 ±0.015 | 0.762 ±0.039 | 12 |
| ViT-B/16 | 0.780 ±0.026 | **0.842 ±0.056** | 0.699 ±0.056 | 0.798 ±0.037 | 0.901 ±0.037 | 0.659 ±0.074 | 12 |
| ResNet-50 | 0.778 ±0.018 | 0.757 ±0.051 | 0.731 ±0.044 | 0.807 ±0.037 | 0.921 ±0.036 | 0.672 ±0.044 | 12 |
| DINOv2 Probe | 0.767 ±0.016 | 0.742 ±0.068 | 0.763 ±0.024 | 0.770 ±0.038 | 0.887 ±0.010 | 0.675 ±0.039 | 12 |

### Table 3 — F1, test set

**Dataset 1.** Same headers as Table 2. 12 seeds, mean ± 2σ.

| Model | Macro F1 | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.623 ±0.028** | 0.435 ±0.108 | 0.478 ±0.064 | 0.671 ±0.049 | **0.961 ±0.030** | **0.571 ±0.060** | 12 |
| CLIP Probe | 0.594 ±0.018 | 0.367 ±0.048 | **0.480 ±0.049** | **0.684 ±0.045** | 0.906 ±0.019 | 0.534 ±0.026 | 12 |
| ViT-B/16 | 0.584 ±0.025 | **0.495 ±0.088** | 0.404 ±0.091 | 0.662 ±0.045 | 0.892 ±0.032 | 0.468 ±0.095 | 12 |
| ResNet-50 | 0.584 ±0.023 | 0.419 ±0.068 | 0.444 ±0.069 | 0.667 ±0.058 | 0.908 ±0.020 | 0.481 ±0.058 | 12 |
| DINOv2 Probe | 0.569 ±0.016 | 0.403 ±0.062 | 0.467 ±0.045 | 0.628 ±0.034 | 0.878 ±0.008 | 0.471 ±0.054 | 12 |

---

## Dataset 2 — full dataset + synthetic corruptions

**1,568 pairs · 7,840 decisions · 100% human-verified.**

**425 corrupted renderings generated** by deliberate API corruption (228
targeting thickness, 197 targeting line continuity). Each was reviewed by hand;
**218 were kept** and 207 rejected. A corrupted pair inherits its source pair's
split, so no drawing appears in more than one split.

### Table 4 — Dataset audit

| Option | train + / − | val + / − | test + / − |
|---|---|---|---|
| `too_thick` | 335 / 794 | 46 / 166 | 48 / 179 |
| `broken_lines` | 293 / 836 | 57 / 155 | 64 / 163 |
| `missing_parts` | 445 / 684 | 78 / 134 | 84 / 143 |
| `missing_texture` | 755 / 374 | 157 / 55 | 168 / 59 |
| `extra_parts` | 431 / 698 | 55 / 157 | 62 / 165 |
| **Pairs** | **1,129** | **212** | **227** |

### Table 5 — AUC, test set

**Dataset 2.** 227 pairs, 1,135 decisions. 12 seeds, mean ± 2σ.
*Retraining in progress — results pending.*

| Model | Macro AUC | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | — | — | — | — | — | — | 12 |
| CLIP Probe | — | — | — | — | — | — | 12 |
| ViT-B/16 | — | — | — | — | — | — | 12 |
| ResNet-50 | — | — | — | — | — | — | 12 |
| DINOv2 Probe | — | — | — | — | — | — | 12 |

### Table 6 — F1, test set

**Dataset 2.** Same headers as Table 5.
*Retraining in progress — results pending.*

| Model | Macro F1 | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | — | — | — | — | — | — | 12 |
| CLIP Probe | — | — | — | — | — | — | 12 |
| ViT-B/16 | — | — | — | — | — | — | 12 |
| ResNet-50 | — | — | — | — | — | — | 12 |
| DINOv2 Probe | — | — | — | — | — | — | 12 |

---

## Approaches tried

Addressing action point 3 — defect detection and histology / patch-based
methods for fine-grained line continuity.

| Source | What transfers | Status |
|---|---|---|
| **PatchCore** and the industrial anomaly-detection line | Representing an image as local patch descriptors rather than one global vector. The supervision regime does **not** transfer: those methods train on defect-free examples only, while we hold supervised labels for five named defects. | Reviewed, not adopted |
| **Computational pathology / MIL** (ABMIL, CLAM) | The closest match. A slide carries one label explained by sparse local evidence — the same shape as a pair-level label explained by a three-pixel gap. Patch tokens plus an attention head; only the head trains, and the attention localises the defect. | **Next to implement** |
| **clDice**, **Skeleton Recall Loss** | Built around the topology of thin structures. They are losses requiring segmentation masks, which we do not have — so the *measurements* transfer, not the losses. | Measurements adopted |

**Implemented from the above: a morphological / topological feature set.** The
distance transform of the ink mask gives stroke width at every pixel and its
ratio to the nearest gap is the BANA separation criterion directly; skeleton
endpoints and connected components give line continuity. About 30 scale-invariant
features, a small classifier, CPU only.

On Dataset 1 this reaches **`too_thick` AUC 0.808** — second in the fleet, above
the VLM (0.794), ResNet-50 (0.757), DINOv2 (0.742) and CLIP (0.704), and below
only ViT-B/16 (0.842) — with no GPU and no pretrained backbone. It is also
measurably *not* the ink-density shortcut the neural judges rely on. The
patch-attention half of the method has not been run yet.

---

## Next

1. **Morphological features added to the existing probes** — they carry signal
   the neural judges do not, so concatenate rather than replace. Hours.
2. **Patch tokens + attention pooling** — the unrun half of action point 3.
3. **More synthetic corruptions**, targeting co-occurring defects and continuing
   `broken_lines`, which yields roughly twice as many usable positives per image
   generated as thickness does.

---

## Repository

| Path | Contents |
|---|---|
| `evaluators/` | the five judges; `constants.py` holds the options and the BANA-grounded annotation criteria |
| `app.py` | collection and editing interface |
| `research/baselines/` | the four vision judges and their seed sweeps |
| `research/vlm/` | VQA build and LoRA fine-tune |
| `research/corruption/` | deliberate-corruption generation, review and dataset compilation |
| `research/morphology/` | morphological / topological features |
| `research/annotator/` | labelling interface and dataset ingest |

The dataset and checkpoints live outside this repository. Every path in the
dataset is relative to a configurable images root, so the corpus can be released
and re-rooted anywhere.
