# TactileExpert — Status Report

**Next meeting: 2026-08-21**  ·  previous: 2026-08-06  ·  cadence: bi-weekly

A human-in-the-loop pipeline for tactile graphics. Generation → automatic
evaluation by a five-judge fleet → expert edit → re-score. Thesis Chapter 5.

> **Reporting rules for this document.** Every table keeps the same headers at
> every meeting so rows are comparable across dates. Results are per option — a
> single accuracy figure is degenerate against a 65% always-clean baseline. AUC is
> the primary comparison metric; F1 at a fixed threshold rewards conservatism.
> Every model number is quoted with the 12-seed 2σ it must clear (§6), and every
> results table is stamped with the **label state** it was measured at.

---

# Meeting 2026-08-21

Work completed since 2026-08-06.

## 1. Status at a glance

| Item | State for this meeting | Change since 2026-08-06 |
|---|---|---|
| Train label verification | **complete — 918/918 pairs** | 407 trained-on → **918**; the last 511 merged this period |
| Dataset v1 | **1,275 pairs, 6,375 decisions, 100% verified** | train closed; val/test already at 100% |
| Label corrections merged | 531 flips over 2,555 decisions (20.7%) | new |
| Label-noise finding | **replicated** on an independent 511-pair set | was a single-split observation |
| Retrain at 918-verified labels | **complete** — all 5 judges × 12 seeds | new |
| Retrain result (AUC) | `broken_lines` and `extra_parts` gain on 3 of 5 each, **`too_thick` on none** | new |
| Retrain result (decisions) | **all 5 judges now beat always-clean**, up from 3 of 5 | new |
| `broken_lines` vs a constant | significantly worse on **2 of 5**, down from **4 of 5** | the §5d prediction, confirmed |
| Best judge | **VLM, macro AUC 0.814 ±0.027 / F1 0.621 ±0.027** | now leads both metrics, with error bars |
| Deployed fleet | 918-verified installed, thresholds re-derived on val | app was on the 407 fleet |
| Collection round 2 | **paused at 4 labelled pairs** — resumes when more samples come in | v2 corpus was empty |
| Proposal Ch. 1–4 | ready to send | unchanged |

**Headline:** the training set is now fully verified. The label-noise finding
replicated, and the retrain confirms it **selectively** — the two options whose
labels were corrected in a consistent direction improved; `too_thick` did not.

---

## 2. Data labelling — train verification completed

### 2a. What was labelled

| | Pairs | Note |
|---|---|---|
| Train queue annotated | **814** | one pass over every fully-labelled train pair |
| Already merged before this period | 303 | part of the 407 the fleet trained on |
| **Merged this period** | **511** | took train coverage 407 → 918 |
| Verified from earlier passes | 104 | pre-queue verification |
| **Train coverage now** | **918 / 918 (100%)** | dataset v1 is closed |

Annotation ran 2026-08-05 → 2026-08-07 at a median of **13 s per pair**
(minimum 4 s, no sub-2 s gaps) — consistent with genuine pair-by-pair review
rather than bulk defaults.

### 2b. Labels corrected, per option

531 flips over the 2,555 decisions re-examined = **20.7%**, matching the 18–20%
measured earlier on val and test.

| Option | Positives before | Positives after | Change | Labels corrected | Direction |
|---|---|---|---|---|---|
| `extra_parts` | 230 | 346 | **+116** | 142 | 129 × 0→1, 13 × 1→0 — **under-flagged** |
| `too_thick` | 166 | 224 | **+58** | 136 | 97 × 0→1, 39 × 1→0 — **under-flagged** |
| `missing_parts` | 342 | 363 | +21 | 97 | 59 × 0→1, 37 × 1→0 — mixed |
| `missing_texture` | 629 | 611 | −18 | 68 | 25 × 0→1, 43 × 1→0 — mixed |
| `broken_lines` | 237 | 173 | **−64** | 88 | 12 × 0→1, 76 × 1→0 — **over-flagged** |
| **total** | **1,604** | **1,717** | **+113** | **531** | 322 × 0→1, 209 × 1→0 |

**The noise is directional, not symmetric.** This is new. Earlier reporting
treated label noise as a magnitude; the train pass shows each option is wrong in a
characteristic direction — `extra_parts` and `too_thick` systematically missed,
`broken_lines` systematically over-called.

### 2c. Effect on the training target

| | Before | After |
|---|---|---|
| Train positive rate | 34.9% | **37.4%** |
| Train always-clean baseline | 65.1% | **62.6%** |
| Record set (pair × option) | 4,590 rows | **4,590 rows — unchanged** |

The record set did not change, only the labels on it. Feature caches key on record
identity, so they stayed valid and the retrain skipped feature extraction.

---

## 3. Dataset state

Re-issued whenever the dataset changes. Unchanged tables are carried forward as-is.

| Split | Pairs | Decisions | Positive | Always-clean baseline | Human-verified |
|---|---|---|---|---|---|
| train | 918 | 4,590 | 37.4% | 62.6% | **918/918 (100%)** |
| val | 174 | 870 | 34.3% | 65.7% | **174/174 (100%)** |
| test | 183 | 915 | 34.6% | 65.4% | **183/183 (100%)** |
| **total** | **1,275** | **6,375** | — | — | **100%** |

### 3a. Positives per option

| Option | train | val | test | test as % |
|---|---|---|---|---|
| `missing_texture` | 611 | 127 | 135 | 73.8% |
| `missing_parts` | 363 | 62 | 68 | 37.2% |
| `extra_parts` | 346 | 48 | 50 | 27.3% |
| `too_thick` | 224 | 25 | 26 | 14.2% |
| `broken_lines` | 173 | 36 | 38 | 20.8% |

The five options are independent binary questions, not mutually exclusive.
Test = 152 library + 31 generated pairs.

---

## 4. Which labels each result was measured at

Results are not comparable across label states, so every table below is stamped.

| Label state | Train pairs verified | Used by | Reported at |
|---|---|---|---|
| **407-verified** | 407 / 918 (44.3%) | the deployed fleet in `models/` | 2026-08-06 |
| **918-verified** | 918 / 918 (100%) | the retrain in §5 | **2026-08-21** |

**Clarification on the previous meeting.** The fleet presented on 2026-08-06 was
trained on **407** verified train pairs. By the meeting date **506** train pairs
had been labelled, so roughly **99 labels existed that training had not used**.
Those, plus the rest of the queue, are the 511 merged this period.

---

## 5. Evaluator results — test split

183 pairs, 915 decisions. Always-clean baseline **598/915 = 65.4%**.

### 5a. AUC at 918-verified labels — primary comparison metric

12-seed sweep, mean ± 2σ. **Label state: 918-verified.**

| Model | Macro AUC | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.814 ±0.027** | 0.759 ±0.077 | **0.773 ±0.027** | 0.820 ±0.043 | **0.955 ±0.020** | **0.765 ±0.061** | 12 |
| CLIP Probe | 0.783 ±0.011 | 0.678 ±0.037 | 0.723 ±0.017 | **0.839 ±0.019** | 0.906 ±0.014 | 0.768 ±0.027 | 12 |
| ViT-B/16 | 0.783 ±0.025 | **0.824 ±0.044** | 0.705 ±0.054 | 0.800 ±0.061 | 0.894 ±0.044 | 0.690 ±0.035 | 12 |
| ResNet-50 | 0.777 ±0.020 | 0.720 ±0.063 | 0.741 ±0.050 | 0.820 ±0.039 | 0.907 ±0.042 | 0.697 ±0.048 | 12 |
| DINOv2 Probe | 0.765 ±0.023 | 0.694 ±0.102 | 0.758 ±0.023 | 0.780 ±0.036 | 0.886 ±0.019 | 0.707 ±0.035 | 12 |

**All five judges now carry a 12-seed noise floor.** The VLM sweep
(`vlm_sweep_tv918`, 12 seeds × ~2h40m) closed the last gap. Its macro AUC lead over
CLIP is +0.031 against a pooled 2σ of 0.027 — detectable, but only just.

Note the deployed VLM is **seed 43**, which scored macro AUC 0.823 against the
12-seed mean of 0.814, and `too_thick` 0.786 against a mean of 0.759. A single seed
flatters this judge; the sweep is what should be quoted.

### 5b. F1 at 918-verified labels — same headers, secondary

| Model | Macro F1 | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.621 ±0.027** | 0.398 ±0.077 | **0.483 ±0.116** | 0.694 ±0.031 | **0.958 ±0.030** | **0.573 ±0.059** | 12 |
| CLIP Probe | 0.584 ±0.015 | 0.314 ±0.044 | 0.460 ±0.070 | **0.701 ±0.040** | 0.910 ±0.022 | 0.535 ±0.040 | 12 |
| ViT-B/16 | 0.582 ±0.039 | **0.450 ±0.101** | 0.424 ±0.078 | 0.654 ±0.074 | 0.892 ±0.043 | 0.488 ±0.079 | 12 |
| ResNet-50 | 0.581 ±0.031 | 0.359 ±0.086 | 0.452 ±0.080 | 0.670 ±0.051 | 0.916 ±0.029 | 0.509 ±0.050 | 12 |
| DINOv2 Probe | 0.565 ±0.017 | 0.342 ±0.074 | 0.470 ±0.031 | 0.625 ±0.028 | 0.886 ±0.023 | 0.503 ±0.041 | 12 |

VLM thresholds are calibrated on **val** and applied fixed to test — nothing is
fitted on what is reported.

**The VLM is the best judge on both metrics, and now with error bars.** Macro F1
+0.037 over CLIP against a pooled 2σ of 0.027 — detectable. At 407-verified it led
on AUC but trailed CLIP on macro F1; it leads on both at 918-verified.

### 5c. The same table at 407-verified labels — as presented 2026-08-06

Carried forward unchanged for comparison. Single seed (42), not a sweep.

| Model | Macro AUC | Macro F1 | `too_thick` AUC | `broken_lines` AUC | `missing_parts` AUC | `missing_texture` AUC | `extra_parts` AUC |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.786** | 0.551 | 0.732 | 0.728 | 0.806 | **0.952** | 0.715 |
| CLIP Probe | 0.779 | **0.578** | 0.753 | 0.690 | 0.811 | 0.914 | 0.726 |
| DINOv2 Probe | 0.750 | 0.558 | 0.717 | 0.705 | 0.766 | 0.892 | 0.670 |
| ResNet-50 | 0.731 | 0.511 | 0.716 | 0.706 | 0.745 | 0.907 | 0.579 |
| ViT-B/16 | 0.713 | 0.546 | **0.776** | 0.625 | 0.712 | 0.813 | 0.638 |

### 5d. Effect of the label correction — Δ AUC, 918 vs 407, gated on 2σ

**A delta is called only if it exceeds the larger of the two sweeps' 2σ.** All four
vision judges at 12 seeds each. **detected** in bold; everything else is n.s.

| Model | Macro | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` |
|---|---|---|---|---|---|---|
| ViT-B/16 | **+0.055** | +0.022 | **+0.080** | +0.051 | +0.040 | **+0.082** |
| VLM Fine-tuned (7B) † | **+0.028** | +0.027 | **+0.045** | +0.014 | +0.004 | +0.050 |
| ResNet-50 | +0.026 | −0.020 | +0.031 | +0.034 | −0.007 | **+0.092** |
| DINOv2 Probe | +0.015 | −0.038 | **+0.067** | +0.016 | +0.004 | +0.023 |
| CLIP Probe | +0.013 | −0.035 | +0.033 | +0.027 | −0.003 | **+0.045** |
| **models detecting a gain** | **2 / 5** | **0 / 5** | **3 / 5** | **0 / 5** | **0 / 5** | **3 / 5** |

On F1: ResNet-50 `extra_parts` **+0.092** and DINOv2 `broken_lines` **+0.060** are
detected; everything else n.s.

† The VLM row compares its 12-seed mean at 918 against its **single** 407 run —
there is no 407 VLM sweep, so that side has no error bar and the comparison is
one-sided against the 918 noise floor. Its `too_thick` +0.027 sits well inside a 2σ
of **0.077**, the widest error bar of any judge-option pair in the fleet.

### 5e. Reading of §5d — what predicts a gain is the *direction* of the correction

| Option | Corrections | Dominant direction | Models detecting a gain |
|---|---|---|---|
| `extra_parts` | 142 | **91%** one way (0→1) | **3 / 4** |
| `broken_lines` | 88 | **86%** one way (1→0) | **2 / 4** |
| `too_thick` | 136 | 71% one way | **0 / 4** |
| `missing_parts` | 97 | 61% | 0 / 4 |
| `missing_texture` | 68 | 63% (fewest corrections) | 0 / 4 |

Among the three heavily-corrected options, **how consistently the labels were wrong
in one direction predicts the gain better than how many were wrong.**
`too_thick` had the second-most corrections and the least directional consistency,
and gained nothing. `extra_parts` had the most one-sided correction and gained on
three of four judges.

| Prediction made before the retrain | Outcome |
|---|---|
| `broken_lines` improves (over-flagged, corrected down) | **confirmed** — detected on 3 of 5 |
| `extra_parts` improves (under-flagged, corrected up) | **confirmed** — detected on 3 of 5 |
| `too_thick` improves (136 corrections) | **not confirmed** — 0 of 5, wrong sign on 3 |
| `missing_texture` unchanged (cleanest labels) | **confirmed** — 0 of 5 |

**`too_thick` is not explained by label noise, on all five judges.** The VLM was the
last open question: at a single seed it appeared to gain +0.055 AUC, but its 12-seed
sweep puts the mean at 0.759 with a 2σ of 0.077, so the gain is inside seed noise
and the deployed seed 43 was simply a favourable draw. Nothing in the fleet improved
on `too_thick` from a fully verified training set.

That makes it the one fine-structure failure that survives clean labels, and points
at the representation rather than at more annotation. Standing caveat: it is also the
thinnest class — 26 test positives, F1 2σ up to 0.139 — so an effect below that bar
would be invisible either way.

### 5f. Against the always-clean baseline — paired McNemar

Always-clean = **598/915**. Net = decisions gained over answering "clean" to
everything. **Bold = p<0.05.** Each fleet is scored at **its own val-fitted
per-option thresholds** — see §5h for why that is not optional.

**Label state: 918-verified (current).**

| Model | Net | p | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **+138** | <0.0001 | −10 | +1 | **+21** | **+121** | +5 |
| ViT-B/16 | **+81** | <0.0001 | −9 | **−30** | +13 | **+105** | +2 |
| CLIP Probe | **+75** | <0.0001 | **−34** | −7 | **+21** | **+112** | −17 |
| ResNet-50 | **+70** | <0.0001 | −7 | −13 | **+23** | **+112** | **−45** |
| DINOv2 Probe | **+53** | 0.0021 | **−32** | **−17** | **+21** | **+101** | **−20** |

**Label state: 407-verified (previous meeting), same thresholds treatment.**

| Model | Net | p | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **+126** | <0.0001 | −12 | −5 | **+23** | **+123** | −3 |
| CLIP Probe | **+91** | <0.0001 | **−12** | **−24** | **+18** | **+116** | −7 |
| ResNet-50 | **+43** | 0.0004 | **−16** | **−23** | +8 | **+112** | **−38** |
| DINOv2 Probe | +33 | 0.0745 n.s. | **−18** | **−22** | −1 | **+104** | **−30** |
| ViT-B/16 | +13 | 0.5340 n.s. | **−16** | **−51** | −14 | **+104** | −10 |

### 5g. What the relabelling changed at the decision level

| | 407-verified | 918-verified |
|---|---|---|
| Judges beating always-clean significantly | 3 of 5 | **5 of 5** |
| Judges significantly worse on `broken_lines` | **4 of 5** | **2 of 5** |
| Judges with **no** significant per-option loss | 1 (VLM) | 1 (VLM) |
| `missing_parts` significant gains | 2 of 5 | **5 of 5** |

Count of options where each judge is significantly *worse* than a constant:

| Model | 407 | 918 | Change |
|---|---|---|---|
| ResNet-50 | 3 | 1 | `too_thick`, `broken_lines` cleared |
| ViT-B/16 | 2 | 1 | `too_thick` cleared |
| CLIP Probe | 2 | 1 | `broken_lines` cleared |
| VLM Fine-tuned | 0 | 0 | still clean |
| DINOv2 Probe | 3 | 3 | unchanged |

**The `broken_lines` prediction in §5d is confirmed at the decision level.**
`broken_lines` net improved for **all five** judges, and three of the four that
were significantly worse than a constant no longer are. `too_thick` remains mixed
— cleared for ResNet-50 and ViT-B/16, worse for CLIP and DINOv2 — consistent with
§5e: it is the one option the relabelling did not fix.

### 5h. Why the operating point had to be re-derived first

Thresholds are fitted to a fleet's score distribution and do not survive a
retrain. Correcting the labels raised train `too_thick` positives 166 → 224, which
shifted the scores upward:

| Val-fitted `too_thick` threshold | 407 | 918 |
|---|---|---|
| ResNet-50 | 0.30 | **0.65** |
| ViT-B/16 | 0.50 | **0.90** |

Gated at a flat t=0.50 instead, the new fleet looks **worse** on `too_thick` than
the fleet it replaced on all four vision judges — while its AUC there is
essentially unchanged. That is a calibration artifact, not lost discrimination,
and it is the same fixed-threshold trap that motivates preferring AUC: at a low
base rate, a judge that flags more sheds true negatives faster than it gains true
positives. Both tables above therefore use val-fitted thresholds, and
`mcnemar_gate.py --thresholds` refuses a threshold file fitted on the split being
reported.

### 5i. Ensemble

| Comparison | 407-verified | 918-verified | Verdict |
|---|---|---|---|
| Ensemble vs best member (VLM) | 729 vs 743, net −14, p=0.2819 | 720 vs 748, net **−28, p=0.0272** | now **significantly worse** |
| Combiner weights | 0 of 32 vote patterns differ from unweighted 3-of-5 majority | — | inert |

At 407-verified the ensemble merely failed to beat its best member. At
918-verified it is significantly worse than it. The case for dropping it is
stronger than last meeting, not weaker.

Recommendation: drop the ensemble from the story. One judge with a stated scope is
a cleaner claim than five with none.

> The ensemble's *macro* AUC/F1 figures quoted at earlier meetings are not
> reproducible from the current evaluation artifacts — only the decision-level
> numbers above are sourced. They are omitted rather than carried forward.

---

## 6. Noise floor — the bar any delta must clear

12 seeds at a fixed label state, mean ± 2σ, test split. A change smaller than 2σ
is not a result.

| Model | Macro AUC | Macro F1 | F1 `too_thick` | F1 `extra_parts` | Label state |
|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | 0.814 ±0.027 | 0.621 ±0.027 | 0.398 ±0.077 | 0.573 ±0.059 | **918** |
| CLIP Probe | 0.783 ±0.011 | 0.584 ±0.015 | 0.314 ±0.044 | 0.535 ±0.040 | **918** |
| ViT-B/16 | 0.783 ±0.025 | 0.582 ±0.039 | 0.450 ±**0.101** | 0.488 ±0.079 | **918** |
| ResNet-50 | 0.777 ±0.020 | 0.581 ±0.031 | 0.359 ±0.086 | 0.509 ±0.050 | **918** |
| DINOv2 Probe | 0.765 ±0.023 | 0.565 ±0.017 | 0.342 ±0.074 | 0.503 ±0.041 | **918** |
| CLIP Probe | 0.769 ±0.016 | 0.581 ±0.025 | 0.352 ±0.041 | 0.531 ±0.041 | 407 |
| DINOv2 Probe | 0.751 ±0.015 | 0.558 ±0.023 | 0.401 ±0.052 | 0.457 ±0.054 | 407 |
| ResNet-50 | 0.751 ±0.029 | 0.565 ±0.030 | 0.377 ±**0.139** | 0.416 ±0.074 | 407 |
| ViT-B/16 | 0.728 ±0.027 | 0.547 ±0.032 | 0.443 ±**0.114** | 0.421 ±0.094 | 407 |

**Consequence:** on the thin classes, effects below ~0.10 are undetectable at 183
test pairs. `too_thick` has 26 positives and a 2σ of up to 0.139 — wider than most
effects worth reporting. The VLM's `too_thick` 2σ of 0.077 on AUC is the widest in
the fleet and is why its apparent gain there could not be called (§5e).

Regenerate with `research/baselines/summarize_sweep.py <sweep_root> --markdown`;
add `--vs <previous_root>` to gate deltas on 2σ automatically.

---

## 7. Label quality — the central finding, now replicated

**Noise:signal = labels corrected ÷ true positives.**

| Option | Test (183 pairs) | | Train (511 new pairs) | | Direction of train error |
|---|---|---|---|---|---|
| | positives | noise:signal | positives | noise:signal | |
| `broken_lines` | 38 | 0.95 | 83 | **1.06** | over-flagged |
| `too_thick` | 26 | **1.27** | 131 | **1.04** | under-flagged |
| `extra_parts` | 50 | 0.66 | 206 | 0.69 | under-flagged |
| `missing_parts` | 68 | 0.47 | 230 | 0.42 | mixed |
| `missing_texture` | 135 | 0.13 | 320 | 0.21 | mixed |

Same rank order on 511 pairs and 2,555 decisions — disjoint from test and 2.8× its
size. Previously a single-split observation; now **replicated**.

### 7a. Noise rank is the inverse of performance rank

| Option | Noise rank (worst first) | AUC rank at 407 (best first) | Beat always-clean at 407? |
|---|---|---|---|
| `broken_lines` | 1 | 5 | **no — 3 of 5 significantly worse** |
| `too_thick` | 2 | 4 | no |
| `extra_parts` | 3 | 3 | no |
| `missing_parts` | 4 | 2 | yes, 2 of 5 |
| `missing_texture` | 5 | 1 | **yes, all 5 decisively** |

The option with the cleanest labels is the one every judge does well on; the
noisiest is the one every judge fails.

### 7b. How §5d refines this

| Before this period | After |
|---|---|
| "The fleet's weakness on fine structure is a data problem, not an architecture problem." | Partly. Fixing labels gained `extra_parts` on 3 of 4 judges and `broken_lines` on 2 of 4. It gained **nothing** on `too_thick`, which survives a fully verified training set. |
| Noise magnitude (noise:signal) was the predictor | **Directional consistency of the correction is the better predictor** (§5e). `too_thick` has the worst noise:signal and gained nothing; `extra_parts` has the most one-sided error and gained most. |

That is a sharper claim than the original and it splits the remaining work: labels
explain part of the gap, and `too_thick` is now the one clean open question —
either the representation is wrong or the target is irreducibly noisy. The ceiling
study distinguishes those two, which is why it moved up the list in §10.

---

## 8. Ruled out — closed with measurement

| Question | Test | Result | Status |
|---|---|---|---|
| Is input resolution the bottleneck? | 448px vs 896px, matched recipe and seed | val macro-F1 **0.6568 vs 0.6569** | falsified |
| Is model capacity the bottleneck? | 2B vs 7B, matched recipe | dead tie in-domain | falsified |
| Do ensemble weights help? | accuracy- and AUC-derived weights | 0 of 32 vote patterns change | inert |
| Does the ensemble beat its best member? | paired McNemar on decisions | net −14, p=0.2819 | no |
| Do panel scores predict which defect a human fixes? | GroupKFold on 155 edit steps | 0.42 vs 0.406 constant; 2 of 5 actions never predicted | no signal |
| Is the RL framing viable? | four prerequisites tested | no learnable state; reward measures 1 of 5 options; action space collapses to a lookup; n=155 | **retired** |
| Is `too_thick` explained by label noise? | retrain at 918-verified, 4 judges × 12 seeds | 0 of 4 detect a gain; wrong sign on 3 | **not on current evidence** (§5e) |

Full rationale for the RL retirement, the metric glossary, and the theory removed
when this document became a status report: `git show 4b84e04:README.md`.

---

## 9. In flight

**This period is closed — everything below COMPLETED, exit 0, on 2026-08-07/08.**

| Job(s) | What | Requested | Actual |
|---|---|---|---|
| 19279792 | `backbones` 12-seed sweep @ 918 | 1:05 | 44:31 |
| 19279793 | `clip` 12-seed sweep @ 918 | 0:45 | 16:01 |
| 19279794 | `dino` 12-seed sweep @ 918 | 0:25 | 12:49 |
| 19280190 | Qwen2-VL-7B LoRA @ 918 (deployed seed 43) | 4:00 | 2:37:59 |
| 19290283/85/86 | seed-42 vision judges @ 918, for deployment | 0:20 ea | 4m07s / 2m11s / 1m01s |
| 19297548 | score the installed fleet on test | 0:30 | 2:45 |
| 19298127 | re-derive val thresholds for the new fleet | 0:30 | 2:33 |
| 19290309–19290315 | **VLM 12-seed sweep**, 6 parallel jobs | 6:00 ea | 5:20–5:52 |

Outputs: `sweep_tv918/`, `vlm_sweep_tv918/`, `output_7b_tv918/`, `models_tv918/`.
Per-seed summaries exported to `experiments_out/sweep_summaries/*.json`.

Walltime was right-sized from measured runtimes rather than left at script defaults.
That is the only lever on start time at fair-share 0.16, and the effect is stark:
every job at ≤1h05 started within ~15 minutes, while a 6h request was scheduled
**~42 hours out**. The VLM sweep ran as 6 concurrent 2-seed jobs rather than one
serial job — with 2h40m per seed, compute dominates queue latency and the usual
"loop seeds inside one job" rule inverts.

**Deployed:** `models/` is the 918-verified fleet (vision seed 42, VLM seed 43),
provenance in `FLEET_MANIFEST.json`, val thresholds re-derived for it.

---

## 10. Next

Modelling work on dataset v1 is **complete and closed**. Everything remaining is
either annotation effort or waits on more collected samples.

| # | Task | Blocked on | Value |
|---|---|---|---|
| 1 | **Collection round 2** — more samples | annotation time | paused at 4 pairs. **Seed images likely to produce thick or broken lines**: `too_thick` is the thinnest class, the noisiest, and the one clean labels did not fix. Class balance beats volume. |
| 2 | **Ceiling study** — double-annotate ~100 pairs with a time gap | annotation effort only, no GPU | the decisive experiment: separates "wrong representation" from "irreducibly noisy target" for `too_thick` |
| 3 | Dense patch features vs global CLS vector | — | the last modelling explanation for `too_thick` now that labels are ruled out on all five judges. Falsifiable: should move `too_thick`/`broken_lines` and leave `missing_texture` alone. |
| 4 | Rework the VLM question set to the BANA wording | a VLM retrain | drafted and gated in `research/vlm/questions_v4_bana.py`, parked because `broken_lines` overshot. That file says to revisit once the holdout has >8 positives — i.e. after this collection round. |
| 5 | Expert rating exercise on sampled outputs | protocol design | RQ4's second half; light protocol, no recruitment |
| 6 | Supervisor's suggested experiments from 2026-08-06 | to be specified | — |

### 10a. Why the ceiling study is the highest-value item

| | |
|---|---|
| What it measures | per-option agreement when a careful annotator revisits the same pairs |
| Why it matters | re-verification changed 14.6% of evaluation decisions and 20.7% of the train decisions re-examined. If the target itself is unstable, no model can exceed that agreement. |
| What it changes | "AUC 0.75 on `too_thick`" is weak against a perfect target and respectable against one humans reproduce ~80% of the time |
| Why it is now sharper | §5e shows `too_thick` did not improve with better labels. Either the representation is wrong, or the target is irreducibly noisy — the ceiling study distinguishes these. |
| Cost | annotation time; no compute |
| Precedent | to our knowledge no tactile-graphics QA work has established such a ceiling |

One accidental instance already exists: a pair re-annotated 2 days after its first
pass disagreed on 1 of 5 options. That is n=1 — not a measurement, but it confirms
the protocol runs on existing tooling.

### 10b. Open question for discussion

Is a measured ceiling on human agreement acceptable as a **primary** contribution,
or does Chapter 5 need a positive modelling result alongside it?

---

# Previous meetings

## 2026-08-06

| Item | State as presented |
|---|---|
| Dataset | 1,275 pairs, 6,375 decisions; val and test 100% verified, train 44.3% |
| Fleet | 5 judges trained at **407-verified** labels, seed 42 |
| Results | §5c and §5f above |
| Central finding | label noise rank is the inverse of performance rank (test split only) |
| Labelling in progress | 506 train pairs labelled; ~99 not yet used by training |
| Ruled out | resolution, capacity, ensemble weights, ensembling, RL prerequisites |

---

## Appendix — where things live

| Path | Contents |
|---|---|
| `evaluators/` | the five judges; `constants.py` is the single source of truth for the options |
| `app.py` | Gradio collection and editing interface |
| `models/` | deployed fleet (gitignored) |
| `research/annotator/` | verification queue, annotator UI, merge-back, collection ingest |
| `research/baselines/` | four vision judges, seed sweeps, sweep summarizer |
| `research/vlm/` | VQA build, LoRA fine-tune |
| `research/experiments/` | scoring, paired significance, calibration |
| `research/_retired/` | pre-v1 scripts — provenance only, do not run |
| `project_notes/SESSION_NOTES.md` | working handoff: state, traps, commands |
| `git show 4b84e04:README.md` | the pre-status-report README — theory, RL-retirement rationale, metric glossary |

Data and checkpoints live outside the repo at `$TACTILE_DATA_ROOT`
(default `~/vlm_finetune`). Every mutating script is dry-run by default and
requires `--apply`.
