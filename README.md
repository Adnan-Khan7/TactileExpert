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
| Dataset | **1,350 pairs, 6,750 decisions, 100% human-verified** | 1,275 → 1,350; train verification closed at 918/918, then +75 collected |
| Label verification | complete on all splits | 511 pairs merged, 531 flips (20.7%) |
| Label-noise finding | **replicated** on an independent 511-pair set | was a single-split observation |
| Collection round 2 | **75 pairs / 52 sessions**, 48% deliberate corruptions | v2 corpus was empty |
| Retrains completed | **3 full 12-seed sweeps** of all 5 judges (918, tv974, tv974-frozen-test) | new |
| Best judge | **VLM — macro AUC 0.826 ±0.018, F1 0.623 ±0.028** | leads both metrics, with error bars |
| Did relabelling help? | **yes** — `broken_lines` and `extra_parts` gained on 3 of 5 judges | §5e |
| Did the +75 new pairs help? | **no measurable effect yet** — 0 of 48 comparisons detected | §5c, controlled |
| `too_thick` | still unimproved by any data change, on all 5 judges | now the standing open problem |
| Proposal Ch. 1–4 | ready to send | unchanged |

**Headline.** Correcting labels helped and is measurable. Adding 6% more data did
not, and a controlled experiment shows the apparent gain was the test set, not
learning. `too_thick` resists every data-side intervention tried so far, which
moves the open question from annotation to architecture — see §10.

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

Re-issued whenever the dataset changes. **Changed this period:** 75 collected pairs
were ingested on 2026-08-17, split by session at v1's own 72/14/14 ratio.

| Split | Pairs | Decisions | Positive | Always-clean baseline | Human-verified | of which v2 |
|---|---|---|---|---|---|---|
| train | 974 | 4,870 | 36.7% | 63.3% | **974/974 (100%)** | 56 |
| val | 183 | 915 | 33.9% | 66.1% | **183/183 (100%)** | 9 |
| test | 193 | 965 | 34.2% | 65.8% | **193/193 (100%)** | 10 |
| **total** | **1,350** | **6,750** | — | — | **100%** | **75** |

Previous state for comparison: 1,275 pairs / 6,375 decisions, test 183, always-clean
65.4%. The whole corpus remains 100% human-verified.

### 3a. Positives per option

| Option | train | val | test | test as % |
|---|---|---|---|---|
| `missing_texture` | 617 | 131 | 136 | 70.5% |
| `missing_parts` | 369 | 63 | 71 | 36.8% |
| `extra_parts` | 361 | 48 | 53 | 27.5% |
| `broken_lines` | 183 | 36 | 39 | 20.2% |
| `too_thick` | 259 | 32 | 31 | 16.1% |

The five options are independent binary questions, not mutually exclusive.

### 3b. Composition of the 75 new pairs

| | |
|---|---|
| Sessions | 52 |
| Deliberate corruptions | 36 of 75 (48%) |
| `too_thick` positives | 47 (63% of the new pairs, against 14% in v1 test) |
| `broken_lines` positives | 11 (was 0 at the previous meeting) |
| All-clean pairs | 13 |

Collection deliberately targeted the two thin classes. Train `too_thick` positives
went 224 → 259 and `broken_lines` 173 → 183. Because the new material is far more
`too_thick`-heavy than v1, the corpus balance shifts as collection grows — the mix
is recorded here at every ingest.

## 4. Which dataset each result was measured on

Results are not comparable across dataset states, so every table is stamped.

| State | Train | Val | Test | Reported at |
|---|---|---|---|---|
| 407-verified | 407 / 918 | 174 | 183 | 2026-08-06 |
| 918-verified | 918 / 918 | 174 | 183 | interim |
| **tv974 (current)** | **974** | **183** | **193** | **2026-08-21** |

tv974 is the first state whose **evaluation splits changed**. The frozen pre-ingest
test pair list is preserved at `data/frozen_v1_test_pairs.json`, and all 183 survive
inside the new 193-pair test — which is what makes §5c possible.

## 5. Evaluator results — test split

**Dataset: tv974.** 193 pairs, 965 decisions. Always-clean baseline
**635/965 = 65.8%**. All five judges, 12 seeds each, mean ± 2σ.

### 5a. AUC — primary comparison metric

| Model | Macro AUC | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.826 ±0.018** | 0.794 ±0.057 | **0.780 ±0.057** | 0.817 ±0.038 | **0.963 ±0.016** | **0.776 ±0.036** | 12 |
| CLIP Probe | 0.790 ±0.014 | 0.704 ±0.023 | 0.737 ±0.019 | **0.832 ±0.037** | 0.914 ±0.015 | 0.762 ±0.039 | 12 |
| ViT-B/16 | 0.780 ±0.026 | **0.842 ±0.056** | 0.699 ±0.056 | 0.798 ±0.037 | 0.901 ±0.037 | 0.659 ±0.074 | 12 |
| ResNet-50 | 0.778 ±0.018 | 0.757 ±0.051 | 0.731 ±0.044 | 0.807 ±0.037 | 0.921 ±0.036 | 0.672 ±0.044 | 12 |
| DINOv2 Probe | 0.767 ±0.016 | 0.742 ±0.068 | 0.763 ±0.024 | 0.770 ±0.038 | 0.887 ±0.010 | 0.675 ±0.039 | 12 |

### 5b. F1 — same headers, secondary

| Model | Macro F1 | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` | Seeds |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.623 ±0.028** | 0.435 ±0.108 | 0.478 ±0.064 | 0.671 ±0.049 | **0.961 ±0.030** | **0.571 ±0.060** | 12 |
| CLIP Probe | 0.594 ±0.018 | 0.367 ±0.048 | **0.480 ±0.049** | **0.684 ±0.045** | 0.906 ±0.019 | 0.534 ±0.026 | 12 |
| ViT-B/16 | 0.584 ±0.025 | **0.495 ±0.088** | 0.404 ±0.091 | 0.662 ±0.045 | 0.892 ±0.032 | 0.468 ±0.095 | 12 |
| ResNet-50 | 0.584 ±0.023 | 0.419 ±0.068 | 0.444 ±0.069 | 0.667 ±0.058 | 0.908 ±0.020 | 0.481 ±0.058 | 12 |
| DINOv2 Probe | 0.569 ±0.016 | 0.403 ±0.062 | 0.467 ±0.045 | 0.628 ±0.034 | 0.878 ±0.008 | 0.471 ±0.054 | 12 |

**The VLM leads both metrics** and is best or joint-best on four of five options.
`too_thick` remains the weakest option for every judge except ViT-B/16, and carries
the widest error bars in the fleet (F1 2σ up to 0.108).

### 5c. Did the new annotations help? A controlled decomposition

The naive comparison against the previous state is confounded: the retrain and the
test set changed in the same step. To separate them, the tv974 models were
**re-evaluated on the frozen pre-ingest 183-pair test** with train and val held
bit-identical — same seeds, same weights (verified), only the evaluation set moves.

| Model | `too_thick` AUC 918/183 | 974/183 | 974/193 | **Learning effect** | **Test-set effect** | Naive total |
|---|---|---|---|---|---|---|
| ResNet-50 | 0.720 | 0.738 | 0.757 | +0.018 | +0.020 | +0.037 |
| ViT-B/16 | 0.824 | 0.834 | 0.842 | +0.010 | +0.008 | +0.018 |
| CLIP Probe | 0.678 | 0.679 | 0.704 | **+0.001** | **+0.025** | +0.025 |
| DINOv2 Probe | 0.694 | 0.714 | 0.742 | +0.020 | +0.028 | +0.048 |

**Learning effect, test held fixed: 0 detected out of 48** model × option × metric
comparisons. Macro AUC moves +0.003, +0.004, −0.001, +0.000 against 2σ of
0.015–0.025.

| Question | Answer |
|---|---|
| Did the 75 new pairs measurably improve the judges? | **No, not yet.** |
| Then what produced the apparent `too_thick` gain? | The test set. 4 of the 5 new `too_thick` test positives are deliberate corruptions, which are easier than naturally occurring ones. For CLIP that accounts for the entire effect. |
| Is this evidence that labelling does not help? | **No.** +56 training pairs is +6%. The measured floor says roughly 4× the positives are needed to halve the detection bar; a null result at this volume is the expected result. |
| Is anything moving in the right direction? | Yes — the learning effect is positive on `too_thick` for 3 of 4 judges, just below the bar. |

This control cost ~1.5 h of GPU and no new modelling code. **It should ship with
every future ingest**, or evaluation-split growth will keep inflating comparisons
in the same direction.

### 5d. Against the always-clean baseline — paired McNemar

> **Measured on the 918-verified state (test = 183, always-clean 598/915).** Not
> yet re-run at tv974: that needs a deployed fleet, and none was installed this
> period pending these results. Carried forward unchanged.

Always-clean = **598/915**. Net = decisions gained over answering "clean" to
everything. **Bold = p<0.05.** Each fleet is scored at **its own val-fitted
per-option thresholds** — see §5f for why that is not optional.

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

### 5e. What the relabelling changed at the decision level

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
the earlier finding: it is the one option the relabelling did not fix.

### 5f. Why the operating point had to be re-derived first

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

### 5g. Ensemble

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

12 seeds per judge, mean ± 2σ, **tv974 test (193 pairs)**. A change smaller than
2σ is not a result.

| Model | Macro AUC | Macro F1 | F1 `too_thick` | F1 `extra_parts` |
|---|---|---|---|---|
| VLM Fine-tuned (7B) | 0.826 ±0.018 | 0.623 ±0.028 | 0.435 ±**0.108** | 0.571 ±0.060 |
| CLIP Probe | 0.790 ±0.014 | 0.594 ±0.018 | 0.367 ±0.048 | 0.534 ±0.026 |
| ViT-B/16 | 0.780 ±0.026 | 0.584 ±0.025 | 0.495 ±0.088 | 0.468 ±0.095 |
| ResNet-50 | 0.778 ±0.018 | 0.584 ±0.023 | 0.419 ±0.068 | 0.481 ±0.058 |
| DINOv2 Probe | 0.767 ±0.016 | 0.569 ±0.016 | 0.403 ±0.062 | 0.471 ±0.054 |

**Consequence.** `too_thick` has 31 test positives and an F1 2σ up to 0.108, so
effects below ~0.10 on it are invisible. This is why §5c can report a null and why
collection alone is a slow lever: the 75 new pairs moved the `too_thick` positive
count from 26 to 31.

Regenerate with `research/baselines/summarize_sweep.py <root> --markdown`; add
`--vs <previous_root>` to gate deltas on 2σ automatically.

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

## 9. Compute this period

All jobs COMPLETED, exit 0. Nothing is running.

| Jobs | What | Requested | Actual |
|---|---|---|---|
| 19279792–19280190 | 918-verified retrain, 4 vision sweeps + VLM | — | 12m–2h38m |
| 19290283–19298127 | deploy seed-42 fleet, score, re-derive thresholds | 0:20–0:30 | 1m–4m |
| 19290309–19290315 | VLM 12-seed sweep @ 918, 6 parallel jobs | 6:00 | 5:20–5:52 |
| 19991210–19991212 | tv974 vision sweeps, cold cache | 0:35–1:10 | 16m–49m |
| 19991213–19991218 | tv974 VLM 12-seed sweep, 6 parallel | 7:00 | 5:38–6:01 |
| 19994706–19994708 | **frozen-test control**, identical models | 0:35–1:10 | 16m–45m |

Three complete 12-seed sweeps of all five judges this period. Walltime is sized
from measured runtimes: everything at ≤1h10 started within ~15 minutes, while a 6h
request queued behind concurrent jobs was once scheduled ~42 h out.

Outputs: `sweep_tv974/`, `vlm_sweep_tv974/`, `sweep_tv974_frozentest/`,
`sweep_tv918/`, `vlm_sweep_tv918/`. Per-seed summaries cached in
`experiments_out/sweep_summaries/`.

**Deployed fleet is still the 918-verified one.** No tv974 fleet was installed —
§5c says there is nothing to gain from redeploying yet.

---

## 10. Next

The data-side levers are now measured. Correcting labels helped; adding 6% more
data did not. `too_thick` and `broken_lines` remain the failures, and §5c shows
more of the same collection will not move them quickly.

| # | Task | Cost | Rationale |
|---|---|---|---|
| 1 | **Morphological / topological features** for `too_thick` + `broken_lines` | hours, CPU only | The defects are geometric. A distance transform of the ink mask *is* stroke width; skeleton endpoints and connected components *are* line continuity. These encode what a global embedding provably cannot. |
| 2 | **Patch-level features + attention pooling (MIL)** in place of the global CLS vector | ~1 day, reuses the feature cache | Borrowed from computational pathology, where a slide-level label must be explained by sparse local evidence — structurally identical to a 3-pixel gap under a pair-level label. Attention weights also localize the defect, which feeds the editing loop. |
| 3 | Continue collection, seeded for `broken_lines` | annotation time | 39 test positives; the thin classes still bound what is measurable. |
| 4 | **Ceiling study** — double-annotate ~100 pairs with a time gap | annotation only | Separates "wrong representation" from "irreducibly noisy target" for `too_thick`. |
| 5 | Rework the VLM question set to the BANA wording | a VLM retrain | Drafted in `research/vlm/questions_v4_bana.py`; unblocked now the holdout has more positives. |
| 6 | Expert rating exercise on sampled outputs | protocol design | RQ4's second half. |

### 10a. Why the architecture direction, and why now

| Evidence | Implication |
|---|---|
| `too_thick` did not improve from correcting 136 labels (0 of 5 judges) | not label noise |
| `too_thick` did not improve from +35 training positives (0 of 48 comparisons) | not data volume, at this scale |
| Judges flag `too_thick` 4/5 on a dense keyboard drawing but **0/5** on a pencil whose strokes tripled | they appear to encode **ink density**, not line-width-to-spacing ratio |
| `too_thick` and `broken_lines` criteria never reference the photograph | both are properties of the rendering alone — the pair comparison, and its registration problem, is not needed for them |

The last row is the cheapest win available: two of the five options can be modelled
as single-image problems, which removes the unregistered-pair difficulty from
exactly the two options that are failing.

**Falsifiable prediction.** Patch-level or morphological features should move
`too_thick` and `broken_lines` and leave `missing_texture` unchanged. If they do
not, the remaining explanation is that the target itself is unstable — which is
what the ceiling study measures.

### 10b. The ceiling study, and why it is now decisive

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


### 10c. Open question for discussion

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
