# TactileExpert — Status Report

**Next meeting: 2026-08-21**  ·  previous: 2026-08-06  ·  cadence: bi-weekly

A human-in-the-loop pipeline for tactile graphics. Generation → automatic
evaluation by a five-judge fleet → expert edit → re-score. Thesis Chapter 5.

> **Reporting rules for this document.** Tables keep the same headers at every
> meeting so rows stay comparable across dates. Results are per option — a single
> accuracy figure is degenerate against a ~66% always-clean baseline. AUC is the
> primary metric; F1 at a fixed threshold rewards conservatism. Every figure is a
> 12-seed mean ± 2σ, and a change smaller than 2σ is not a result.

---

# Meeting 2026-08-21

## 1. Status at a glance

| Item | State for this meeting | Change since 2026-08-06 |
|---|---|---|
| Dataset | **1,350 pairs, 6,750 decisions, 100% human-verified** | 1,275 → 1,350 |
| Evaluation | one dataset, one test split, all five judges | consolidated — no per-subset arms |
| Retrains | all five judges at **12 seeds** | new |
| Best judge | **VLM Fine-tuned (7B) — macro AUC 0.826 ±0.018** | unchanged as leader |
| Effect of the larger dataset | **1 of 60 comparisons exceeds 2σ** | §4 |
| `too_thick` / `broken_lines` | still the two weakest options for the fleet | standing open problem |
| Supervisor's literature question | answered, with a first implemented result | §5 |
| Proposal Ch. 1–4 | **sent** | was ready to send |

**Headline.** The dataset is final and every judge is retrained on it at 12 seeds.
Growing the corpus by 6% did not measurably move the fleet. The two options that
concern fine structure — `too_thick` and `broken_lines` — remain the weakest, and
§5 reports the first modelling result aimed directly at them: a purely geometric
feature set that reaches 0.808 on `too_thick`, second in the fleet, using no GPU.

---

## 2. The dataset

One dataset from here on. All results in this document are measured on it, and
no result is computed on a subset.

**1,350 pairs / 6,750 decisions, 100% human-verified**, split **train 974 / val
183 / test 193**. The five options are independent binary questions, not mutually
exclusive, so each pair carries five labels. On the test split 34.2% of decisions
are positive, which puts the always-clean baseline at **65.8%** — the number any
judge has to beat to be worth deploying.

The classes are strongly unbalanced, and this bounds what is measurable. In the
**test** split `missing_texture` has 136 positives, `missing_parts` 71,
`extra_parts` 53, `broken_lines` 39 and `too_thick` 31. **Training** positives run
the same order: 617, 369, 361, 183 and 259. The two thinnest evaluation classes
are exactly the two the fleet performs worst on, and with 31 test positives a
`too_thick` effect below roughly 0.10 F1 cannot be distinguished from noise.

---

## 3. Evaluator results — all five judges

**Test split, 193 pairs, 965 decisions. Always-clean baseline 635/965 = 65.8%.**
12 seeds per judge, mean ± 2σ.

### 3a. AUC — primary metric

| Model | Macro AUC | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.826 ±0.018** | 0.794 ±0.057 | **0.780 ±0.057** | 0.817 ±0.038 | **0.963 ±0.016** | **0.776 ±0.036** | 12 |
| CLIP Probe | 0.790 ±0.014 | 0.704 ±0.023 | 0.737 ±0.019 | **0.832 ±0.037** | 0.914 ±0.015 | 0.762 ±0.039 | 12 |
| ViT-B/16 | 0.780 ±0.026 | **0.842 ±0.056** | 0.699 ±0.056 | 0.798 ±0.037 | 0.901 ±0.037 | 0.659 ±0.074 | 12 |
| ResNet-50 | 0.778 ±0.018 | 0.757 ±0.051 | 0.731 ±0.044 | 0.807 ±0.037 | 0.921 ±0.036 | 0.672 ±0.044 | 12 |
| DINOv2 Probe | 0.767 ±0.016 | 0.742 ±0.068 | 0.763 ±0.024 | 0.770 ±0.038 | 0.887 ±0.010 | 0.675 ±0.039 | 12 |

### 3b. F1 — same headers, secondary

| Model | Macro F1 | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.623 ±0.028** | 0.435 ±0.108 | 0.478 ±0.064 | 0.671 ±0.049 | **0.961 ±0.030** | **0.571 ±0.060** | 12 |
| CLIP Probe | 0.594 ±0.018 | 0.367 ±0.048 | **0.480 ±0.049** | **0.684 ±0.045** | 0.906 ±0.019 | 0.534 ±0.026 | 12 |
| ViT-B/16 | 0.584 ±0.025 | **0.495 ±0.088** | 0.404 ±0.091 | 0.662 ±0.045 | 0.892 ±0.032 | 0.468 ±0.095 | 12 |
| ResNet-50 | 0.584 ±0.023 | 0.419 ±0.068 | 0.444 ±0.069 | 0.667 ±0.058 | 0.908 ±0.020 | 0.481 ±0.058 | 12 |
| DINOv2 Probe | 0.569 ±0.016 | 0.403 ±0.062 | 0.467 ±0.045 | 0.628 ±0.034 | 0.878 ±0.008 | 0.471 ±0.054 | 12 |

**Reading these.** The VLM leads both metrics and is best or joint-best on four of
five options. `missing_texture` is easy for everyone; `too_thick` and
`broken_lines` are the two options where the fleet is weakest and where the error
bars are widest — `too_thick` F1 carries a 2σ of up to 0.108, which is the direct
consequence of having 31 test positives. ViT-B/16 is the exception on `too_thick`
and the reason is examined in §5.

---

## 4. Comparison with the previous dataset state

The previous corpus was **1,275 pairs / 6,375 decisions** (train 918 / val 174 /
test 183). The current one adds 75 collected pairs. Both states were swept at 12
seeds with an identical recipe, so the tables below are directly comparable in
form. Note that the evaluation split grew with the corpus, from 183 to 193 pairs,
so this is a comparison of two complete measurements rather than a controlled
before/after on a fixed test set.

### 4a. Previous state — AUC

| Model | Macro AUC | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.814 ±0.027** | 0.759 ±0.077 | **0.773 ±0.027** | 0.820 ±0.043 | **0.955 ±0.020** | 0.765 ±0.061 | 12 |
| CLIP Probe | 0.783 ±0.011 | 0.678 ±0.037 | 0.723 ±0.017 | **0.839 ±0.019** | 0.906 ±0.014 | **0.768 ±0.027** | 12 |
| ViT-B/16 | 0.783 ±0.025 | **0.824 ±0.044** | 0.705 ±0.054 | 0.800 ±0.061 | 0.894 ±0.044 | 0.690 ±0.035 | 12 |
| ResNet-50 | 0.777 ±0.020 | 0.720 ±0.063 | 0.741 ±0.050 | 0.820 ±0.039 | 0.907 ±0.042 | 0.697 ±0.048 | 12 |
| DINOv2 Probe | 0.765 ±0.023 | 0.694 ±0.102 | 0.758 ±0.023 | 0.780 ±0.036 | 0.886 ±0.019 | 0.707 ±0.035 | 12 |

### 4b. Previous state — F1

| Model | Macro F1 | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.621 ±0.027** | 0.398 ±0.077 | **0.483 ±0.116** | 0.694 ±0.031 | **0.958 ±0.030** | **0.573 ±0.059** | 12 |
| CLIP Probe | 0.584 ±0.015 | 0.314 ±0.044 | 0.460 ±0.070 | **0.701 ±0.040** | 0.910 ±0.022 | 0.535 ±0.040 | 12 |
| ViT-B/16 | 0.582 ±0.039 | **0.450 ±0.101** | 0.424 ±0.078 | 0.654 ±0.074 | 0.892 ±0.043 | 0.488 ±0.079 | 12 |
| ResNet-50 | 0.581 ±0.031 | 0.359 ±0.086 | 0.452 ±0.080 | 0.670 ±0.051 | 0.916 ±0.029 | 0.509 ±0.050 | 12 |
| DINOv2 Probe | 0.565 ±0.017 | 0.342 ±0.074 | 0.470 ±0.031 | 0.625 ±0.028 | 0.886 ±0.023 | 0.503 ±0.041 | 12 |

### 4c. What changed

**One of 60 comparisons exceeds its 2σ bar** — 5 judges × 6 axes (five options
plus the macro) × 2 metrics. The exception is CLIP Probe `too_thick` F1, 0.314 →
0.367, a change of +0.053 against a bar of ±0.048, which is marginal.

Macro AUC moves by +0.011, +0.007, −0.003, +0.001 and +0.002 against 2σ of
0.014–0.027. Macro F1 moves by +0.002 to +0.010 against 2σ of 0.017–0.039.

**Conclusion: growing the corpus by 6% did not measurably improve the fleet.**
Every judge is nominally equal or slightly better and not one of those movements
clears the noise floor. This is the expected outcome at this scale rather than a
surprise — 56 additional training pairs is a 6% increase, and with 31 `too_thick`
test positives the detection bar on that option is far above what 6% more data
can deliver. It does mean that further collection at the current rate is a slow
lever, and it is the reason the work has moved to the modelling question in §5.

---

## 5. Answering the supervisor's comment

> *"Look into literature on defect detection and histology/patch-based
> classification methods to improve handling of fine-grained line continuity
> issues."*

### 5a. What the literature offers, and which part transfers

Three families were reviewed, and they transfer to different degrees.

**Industrial / visual defect detection.** PatchCore and the anomaly-detection
line represent an image as a bank of local patch descriptors and score a test
patch by distance to the nearest normal patch. The representation is right — a
defect is local and a global vector destroys it — but the **supervision regime
does not transfer**: these methods train on normal examples only and score
deviation, whereas we hold supervised pair-level labels for five specific
defects. Adopting the method wholesale would discard our labels.

**Computational pathology and multiple-instance learning.** This is the closest
structural match. A whole-slide image carries **one slide-level label** that is
explained by **sparse local evidence** — a few malignant cells in an enormous
field. That is precisely our situation: a pair-level `broken_lines` label
explained by a three-pixel gap. ABMIL and CLAM-style attention pooling replace
the global vector with patch tokens plus a learned attention head, so only the
head trains and n = 1,350 is workable. The attention weights also **localise** the
defect, which feeds the editing half of the pipeline directly.

**Curvilinear-structure segmentation.** clDice and Skeleton Recall Loss are built
exactly around line continuity — they measure and optimise the topology of thin
structures rather than pixel overlap. They are **losses that require segmentation
masks**, which we do not have. What transfers is the *measurement*: skeletons,
endpoints and connectivity as the quantities that describe a broken line.

### 5b. What was implemented, and what it shows

The measurement half was implemented first because it is CPU-only and costs
hours rather than a training cycle. Defects that are geometric are measured
geometrically: the **distance transform** of the ink mask gives stroke width at
every pixel, and its ratio to the nearest gap is the BANA separation criterion
directly; **skeleton endpoints and connected components** give line continuity.
Thirty scale-invariant features, a small classifier, selected on val and scored
once on test.

**Test AUC, same 193-pair split as §3.**

| Feature set | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` |
|---|---|---|---|---|---|
| Morphological probe | **0.808 ±0.031** | 0.745 ±0.038 | 0.624 ±0.030 | 0.856 ±0.015 | 0.632 ±0.052 |
| Ink fraction — one untrained scalar | 0.686 | 0.473 | 0.504 | 0.286 | 0.432 |

Two findings, and the second is the more useful one.

**A purely geometric feature set is the second-best judge on `too_thick`.** At
0.808 it sits above the VLM (0.794), ResNet-50 (0.757), DINOv2 (0.742) and CLIP
(0.704), below only ViT-B/16 (0.842) — with no GPU, no pretrained backbone and
about nine minutes of CPU.

**The fleet's `too_thick` score is largely a proxy for ink density.** A single
untrained scalar — the fraction of dark pixels — reaches 0.686 on `too_thick`.
Asking whether a judge beats that requires a paired test, which needs per-pair
scores; the 12-seed sweeps store only aggregates, so this one comparison is run
against the deployed fleet's own scored output rather than the sweeps.† On it,
**only ViT-B/16 significantly beats ink fraction on `too_thick`** — CLIP is
statistically indistinguishable from it. On `broken_lines` the same scalar is at
chance (0.473) and all five judges beat it decisively. So the two weak options are weak for
**different** reasons: `broken_lines` is genuinely being learned but not well
enough, while `too_thick` is, for four of five judges, being approximated by
"how much ink is on the page". The morphological features are measurably not
that proxy — endpoint density correlates with ink fraction at r = +0.097.

**Stated plainly: the pre-registered target was not met.** The bar was fixed
before extraction — beat ink fraction by ≥ +0.13 on `too_thick` — and the probe
reached +0.122. The improvement is statistically significant but falls under the
threshold set in advance, and it is reported as a miss rather than re-argued.
Two further caveats: `broken_lines` at 0.745 does not beat the best judge on that
option, and the morphological features carry enough information about image
provenance that the `missing_texture` figure above should not be read as texture
detection.

† Deployed fleet, single seed, on the 183 test pairs it was scored on. It is the
only artifact carrying per-pair probabilities. The direction of the finding is
not in question — the gap between ink fraction and four of the five judges is
smaller than the gap between judges — but the exact deltas would need a re-scoring
pass to quote at 12 seeds.

### 5c. What this leaves for the patch-based half

The MIL half of the supervisor's question — patch tokens with attention pooling
in place of the global CLS vector — **has not been run yet**, and it is the
direct answer to "fine-grained line continuity". The morphological result makes
it more interesting rather than less: geometry captures something the neural
judges demonstrably do not, so the open question is whether patch-level attention
recovers the same signal end-to-end, and whether a combination beats either
alone. It reuses the existing feature cache and trains only the head.

---

## 6. Next

| # | Task | Cost | Rationale |
|---|---|---|---|
| 1 | **Patch tokens + attention pooling (MIL)** in place of the global CLS vector | ~1 day, reuses the feature cache | The unanswered half of §5. Attention weights also localise the defect, which feeds the editing loop. |
| 2 | **Morphology + ViT combination** for `too_thick` | hours | The two are measurably not using the same signal; whether they are complementary is a new question and needs its own bar set in advance. |
| 3 | **Ceiling study** — double-annotate ~100 pairs with a time gap | annotation only | Re-verification changed a fifth of the labels it re-examined. If the target itself is unstable, no model can exceed that agreement, and "AUC 0.75 on `too_thick`" reads very differently against a ceiling of 0.80 than against 1.00. |
| 4 | Continue collection, seeded for `broken_lines` and `too_thick` | annotation time | 39 and 31 test positives; the thin classes bound what is measurable, and §4 shows why volume alone is slow. |
| 5 | Rework the VLM question set to the BANA wording | a VLM retrain | Drafted in `research/vlm/questions_v4_bana.py`. |
| 6 | Expert rating exercise on sampled outputs | protocol design | RQ4's second half. |

### 6a. Open question for discussion

Is a measured ceiling on human agreement acceptable as a **primary** contribution
for Chapter 5, or does the chapter need a positive modelling result alongside it?

---

## Appendix — where things live

| Path | Contents |
|---|---|
| `evaluators/` | the five judges; `constants.py` is the single source of truth for the options and the BANA-grounded annotation criteria |
| `app.py` | Gradio collection and editing interface |
| `models/` | deployed fleet (gitignored) |
| `research/baselines/` | the four vision judges, seed sweeps, sweep summarizer |
| `research/vlm/` | VQA build, LoRA fine-tune |
| `research/morphology/` | §5 — feature extraction, trivial baselines, the probe |
| `research/experiments/` | scoring, significance, calibration |
| `project_notes/SESSION_NOTES.md` | working handoff: state, rules, traps, commands |

Data and checkpoints live outside the repo at `$TACTILE_DATA_ROOT`
(default `~/vlm_finetune`). Every mutating script is dry-run by default and
requires `--apply`.
