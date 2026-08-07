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
| Retrain at 918-verified labels | 2 of 5 judges complete at 12 seeds | new |
| First retrain result | `broken_lines` and `extra_parts` improve; `too_thick` does not | new |
| Collection round 2 | starting | v2 corpus holds 1 pair |
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
| CLIP Probe | **0.783 ±0.011** | 0.678 ±0.037 | 0.723 ±0.017 | **0.839 ±0.019** | **0.906 ±0.014** | **0.768 ±0.027** | 12 |
| DINOv2 Probe | 0.765 ±0.023 | **0.694 ±0.102** | **0.758 ±0.023** | 0.780 ±0.036 | 0.886 ±0.019 | 0.707 ±0.035 | 12 |
| ResNet-50 | *pending* | | | | | | 6 / 12 |
| ViT-B/16 | *pending* | | | | | | 5 / 12 |
| VLM Fine-tuned (7B) | *pending* | | | | | | training |

### 5b. F1 at 918-verified labels — same headers, secondary

| Model | Macro F1 | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| CLIP Probe | **0.584 ±0.015** | 0.314 ±0.044 | 0.460 ±0.070 | **0.701 ±0.040** | **0.910 ±0.022** | **0.535 ±0.040** | 12 |
| DINOv2 Probe | 0.565 ±0.017 | **0.342 ±0.074** | **0.470 ±0.031** | 0.625 ±0.028 | 0.886 ±0.023 | 0.503 ±0.041 | 12 |
| ResNet-50 | *pending* | | | | | | 6 / 12 |
| ViT-B/16 | *pending* | | | | | | 5 / 12 |
| VLM Fine-tuned (7B) | *pending* | | | | | | training |

**Rows are published only at 12 seeds.** ResNet-50 and ViT-B/16 are mid-sweep;
their figures moved measurably between two readings taken minutes apart, which is
the whole reason the 2σ rule exists. Two rounds of conclusions were previously
retracted for trusting a low-seed variance estimate, so partial rows are shown as
*pending* rather than as numbers that will change.

### 5c. The same table at 407-verified labels — as presented 2026-08-06

Carried forward unchanged for comparison. Single seed (42), not a sweep.

| Model | Macro AUC | Macro F1 | `too_thick` AUC | `broken_lines` AUC | `missing_parts` AUC | `missing_texture` AUC | `extra_parts` AUC |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.786** | 0.551 | 0.732 | 0.728 | 0.806 | **0.952** | 0.715 |
| CLIP Probe | 0.779 | **0.578** | 0.753 | 0.690 | 0.811 | 0.914 | 0.726 |
| DINOv2 Probe | 0.750 | 0.558 | 0.717 | 0.705 | 0.766 | 0.892 | 0.670 |
| ResNet-50 | 0.731 | 0.511 | 0.716 | 0.706 | 0.745 | 0.907 | 0.579 |
| ViT-B/16 | 0.713 | 0.546 | **0.776** | 0.625 | 0.712 | 0.813 | 0.638 |

### 5d. Effect of the label correction — 918 vs 407, gated on 2σ

**A delta is called only if it exceeds the larger of the two sweeps' 2σ.**
Complete rows only; ResNet-50 and ViT-B/16 are withheld until 12 seeds.

| Model | Option | Δ AUC | Verdict |
|---|---|---|---|
| DINOv2 Probe | `broken_lines` | **+0.067** | **detected** |
| CLIP Probe | `extra_parts` | **+0.045** | **detected** |
| CLIP Probe | macro | +0.013 | n.s. |
| DINOv2 Probe | macro | +0.015 | n.s. |
| CLIP Probe | `too_thick` | −0.035 | n.s. |
| DINOv2 Probe | `too_thick` | −0.038 | n.s. |
| both | `missing_texture` | −0.003 / +0.004 | n.s. |

On F1, DINOv2 `broken_lines` **+0.060 detected**; everything else n.s.

### 5e. Reading of §5d — the prediction was half right

| Prediction | Outcome |
|---|---|
| `broken_lines` improves (labels were over-flagged, corrected down) | **confirmed** — +0.067 AUC detected on DINOv2, +0.060 F1 detected |
| `extra_parts` improves (labels under-flagged, corrected up) | **confirmed** — +0.045 AUC detected on CLIP |
| `too_thick` improves (labels under-flagged, 136 corrections) | **not confirmed** — −0.035 / −0.038 AUC, both n.s. |
| `missing_texture` unchanged (cleanest labels) | **confirmed** — n.s. on both |

The two options corrected in a *consistent direction* improved. `too_thick` had
136 corrections and did not, despite being the option with the worst measured
noise:signal. Its labels moved in both directions (97 up, 39 down), and it is the
thinnest class in test at 26 positives — where 2σ on F1 reaches 0.139. **On
current evidence `too_thick` is not explained by label noise alone**, which points
at the representation argument (§7) rather than more annotation.

### 5f. Against the always-clean baseline — paired McNemar

**Label state: 407-verified.** To be re-run at 918-verified before the meeting.
Net = decisions gained over answering "clean" to everything. **Bold = significant
at α=0.05. Negative = significantly worse than a constant.**

| Model | Net | p | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **+145** | <0.0001 | −3 | 0 | **+23** | **+122** | +3 |
| ResNet-50 | **+87** | <0.0001 | −10 | **−18** | +8 | **+112** | −5 |
| CLIP Probe | **+82** | 0.0001 | −12 | **−33** | **+18** | **+116** | −7 |
| DINOv2 Probe | **+49** | 0.0231 | **−18** | **−39** | +12 | **+109** | −15 |
| ViT-B/16 | +34 | 0.0971 n.s. | **−16** | **−33** | +12 | **+90** | **−19** |

| Claim at 407-verified | Evidence |
|---|---|
| The fleet detects **one** defect reliably | every win is carried by `missing_texture`; remove it and 4 of 5 are net negative |
| Only the VLM never loses to a constant | it is the only model with no significant negative above |
| Three of five are worse than "clean" on `broken_lines` | CLIP −33, DINOv2 −39, ViT −33, all p<0.05 |

**§5d suggests this last row will change at 918-verified** — `broken_lines` was
the most over-flagged option, and it improved. Re-running the gate is the test.

### 5g. Ensemble

| Comparison | Result | Verdict |
|---|---|---|
| Ensemble vs best member (VLM) | 729 vs 743 decisions, net −14, p=0.2819 | not significant — buys nothing |
| Ensemble on `broken_lines` | net −19, p=0.0094 | **significantly worse** |
| Combiner weights | 0 of 32 vote patterns differ from unweighted 3-of-5 majority | inert |

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
| CLIP Probe | 0.769 ±0.016 | 0.581 ±0.025 | 0.352 ±0.041 | 0.531 ±0.041 | 407 |
| DINOv2 Probe | 0.751 ±0.015 | 0.558 ±0.023 | 0.401 ±0.052 | 0.457 ±0.054 | 407 |
| ResNet-50 | 0.751 ±0.029 | 0.565 ±0.030 | 0.377 ±**0.139** | 0.416 ±0.074 | 407 |
| ViT-B/16 | 0.728 ±0.027 | 0.547 ±0.032 | 0.443 ±**0.114** | 0.421 ±0.094 | 407 |

**Consequence:** on the thin classes, effects below ~0.10 are undetectable at 183
test pairs. `too_thick` has 26 positives and a 2σ of up to 0.139 — wider than most
effects worth reporting.

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
| "The fleet's weakness on fine structure is a data problem, not an architecture problem." | Partly. Fixing labels measurably improved `broken_lines` and `extra_parts`. It did **not** improve `too_thick`, so at least one fine-structure failure survives clean labels. |

That is a sharper claim than the original, and it splits the remaining work:
labels explain some of the gap, representation is the candidate for the rest.

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
| Is `too_thick` explained by label noise? | retrain at 918-verified | −0.035 / −0.038 AUC, n.s. | **not on current evidence** (§5e) |

Full rationale for the RL retirement: `project_notes/README_prose_archive_20260807.md`.

---

## 9. In flight

| Job | What | Walltime | Status |
|---|---|---|---|
| 19279793 | `clip` 12-seed sweep @ 918-verified | 0:45 | **complete** |
| 19279794 | `dino` 12-seed sweep @ 918-verified | 0:25 | **complete** |
| 19279792 | `backbones` 12-seed sweep @ 918-verified | 1:05 | running — 5/12 seeds |
| 19280190 | Qwen2-VL-7B LoRA @ 918-verified | 4:00 | running |

Output: `TactileEval_Context_And_Data/sweep_tv918/`, `output_7b_tv918/`.
Walltime right-sized from measured runtimes rather than left at defaults — at
fair-share 0.16 it is the only lever on start time; all four started within
~15 minutes of submission.

---

## 10. Next

| # | Task | Blocked on | Value |
|---|---|---|---|
| 1 | Complete §5a/§5b — 12 seeds for ResNet-50, ViT-B/16, and the VLM | jobs above | makes every row callable against 2σ |
| 2 | Re-run §5f (McNemar vs always-clean) at 918-verified | same | tests whether `broken_lines` still loses to a constant |
| 3 | Re-derive val thresholds, re-score the fleet | same | thresholds are fitted parameters — val only, never test |
| 4 | Collection round 2 in the app | starting now | **seed images likely to produce thick or broken lines** — class balance beats volume |
| 5 | **Ceiling study** — double-annotate ~100 pairs with a time gap | annotation effort only, no GPU | per-option human agreement; reframes every number in §5 |
| 6 | Expert rating exercise on sampled outputs | protocol design | RQ4's second half; light protocol, no recruitment |
| 7 | Supervisor's suggested experiments from 2026-08-06 | to be specified | — |

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
| `project_notes/README_prose_archive_20260807.md` | theory and rationale removed from this document |

Data and checkpoints live outside the repo at `$TACTILE_DATA_ROOT`
(default `~/vlm_finetune`). Every mutating script is dry-run by default and
requires `--apply`.
