# TactileExpert — Status Report

**Meeting: 2026-08-07**  ·  previous: 2026-08-06  ·  cadence: bi-weekly

A human-in-the-loop pipeline for tactile graphics. Generation → automatic
evaluation by a five-judge fleet → expert edit → re-score. Thesis Chapter 5.

> **Reporting rules for this document.** Every table below keeps the same headers
> at every meeting so rows are comparable across dates. Results are per option —
> a single accuracy figure is degenerate against a 65% always-clean baseline. AUC
> is the primary comparison metric; F1 at a fixed threshold rewards conservatism.
> Every model number is quoted with the 12-seed 2σ it must clear (§4).

---

## 1. Status at a glance

| Item | State | Since last meeting |
|---|---|---|
| Dataset v1 | **1,275 pairs, 6,375 decisions, 100% human-verified** | train closed 44.3% → **100%** |
| Label verification | complete on all three splits | +511 pairs merged, 531 flips |
| Evaluator fleet (deployed) | 5 judges, trained at 407-verified labels | unchanged — retrain in flight |
| Results in §3 | measured at **407-verified** labels | unchanged; §3 is re-measured next meeting |
| Retrain at 918-verified | **in flight** — 3 sweeps running, VLM queued | submitted 2026-08-07 |
| Collection app | offline; v2 corpus holds 1 pair | new collection round starting |
| Proposal Ch. 1–4 | ready to send | unchanged |

**Headline this period:** the dataset is closed and fully verified, and the
label-noise finding **replicated** on an independent 511-pair set (§5).

---

## 2. Dataset state

Re-issued whenever the dataset changes. Unchanged tables are carried forward as-is.

| Split | Pairs | Decisions | Positive | Always-clean baseline | Human-verified |
|---|---|---|---|---|---|
| train | 918 | 4,590 | 37.4% | 62.6% | **918/918 (100%)** |
| val | 174 | 870 | 34.3% | 65.7% | **174/174 (100%)** |
| test | 183 | 915 | 34.6% | 65.4% | **183/183 (100%)** |
| **total** | **1,275** | **6,375** | — | — | **100%** |

### 2b. Positives per option

| Option | train | val | test | test as % |
|---|---|---|---|---|
| `missing_texture` | 611 | 127 | 135 | 73.8% |
| `missing_parts` | 363 | 62 | 68 | 37.2% |
| `extra_parts` | 346 | 48 | 50 | 27.3% |
| `too_thick` | 224 | 25 | 26 | 14.2% |
| `broken_lines` | 173 | 36 | 38 | 20.8% |

The five options are independent binary questions, not mutually exclusive.
Test = 152 library + 31 generated pairs.

### 2c. Change log for this period

| Event | Count | Effect |
|---|---|---|
| Verified train pairs merged | 511 | train coverage 44.3% → 100% |
| Label flips | 531 / 2,555 decisions (20.7%) | train positive rate 34.9% → 37.4% |
| Direction | 322 × 0→1, 209 × 1→0 | always-clean baseline 65.1% → 62.6% |
| Record set | **unchanged** (identical fingerprint) | feature caches stayed valid |

---

## 3. Evaluator results — test split

**Label state: 407-verified.** 183 pairs, 915 decisions. Always-clean baseline
**598/915 = 65.4%**. This table is re-measured at 918-verified next meeting; the
headers do not change.

### 3a. AUC — primary comparison metric

| Model | Macro AUC | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` |
|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **0.786** | 0.732 | 0.728 | 0.806 | **0.952** | 0.715 |
| CLIP Probe | 0.779 | 0.753 | 0.690 | **0.811** | 0.914 | **0.726** |
| DINOv2 Probe | 0.750 | 0.717 | 0.705 | 0.766 | 0.892 | 0.670 |
| ResNet-50 | 0.731 | 0.716 | 0.706 | 0.745 | 0.907 | 0.579 |
| ViT-B/16 | 0.713 | **0.776** | 0.625 | 0.712 | 0.813 | 0.638 |

### 3b. F1 — same headers, secondary

| Model | Macro F1 | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` |
|---|---|---|---|---|---|---|
| CLIP Probe | **0.578** | 0.387 | 0.413 | 0.627 | 0.932 | **0.529** |
| DINOv2 Probe | 0.558 | 0.371 | **0.430** | 0.646 | 0.909 | 0.435 |
| VLM Fine-tuned (7B) | 0.551 | 0.293 | 0.387 | **0.662** | **0.951** | 0.460 |
| ViT-B/16 | 0.546 | **0.447** | 0.403 | 0.594 | 0.821 | 0.465 |
| ResNet-50 | 0.511 | 0.333 | 0.429 | 0.605 | 0.920 | 0.267 |

Macro AUC and macro F1 disagree on the ranking. AUC is the one to read: F1 sits at
a fixed threshold, so a judge that flags less sheds false positives faster than
true positives and looks better than it is.

### 3c. Against the always-clean baseline — paired McNemar

Net = decisions gained over answering "clean" to everything. **Bold = significant
at α=0.05. Negative = significantly worse than a constant.**

| Model | Net | p | `too_thick` | `broken_lines` | `missing_parts` | `missing_texture` | `extra_parts` |
|---|---|---|---|---|---|---|---|
| VLM Fine-tuned (7B) | **+145** | <0.0001 | −3 | 0 | **+23** | **+122** | +3 |
| ResNet-50 | **+87** | <0.0001 | −10 | **−18** | +8 | **+112** | −5 |
| CLIP Probe | **+82** | 0.0001 | −12 | **−33** | **+18** | **+116** | −7 |
| DINOv2 Probe | **+49** | 0.0231 | **−18** | **−39** | +12 | **+109** | −15 |
| ViT-B/16 | +34 | 0.0971 n.s. | **−16** | **−33** | +12 | **+90** | **−19** |

**Two readings that matter:**

| Claim | Evidence |
|---|---|
| The fleet detects **one** defect reliably | every model's win is carried by `missing_texture`; remove it and 4 of 5 are net negative |
| Only the VLM never loses to a constant | it is the only model with no significant negative in the row above |
| Three of five are worse than "clean" on `broken_lines` | CLIP −33, DINOv2 −39, ViT −33, all p<0.05 |

### 3d. Ensemble

| Comparison | Result | Verdict |
|---|---|---|
| Ensemble vs best member (VLM) | 729 vs 743 decisions, net −14, p=0.2819 | not significant — buys nothing |
| Ensemble on `broken_lines` | net −19, p=0.0094 | **significantly worse** |
| Combiner weights | 0 of 32 vote patterns differ from unweighted 3-of-5 majority | inert |

Recommendation: drop the ensemble from the story. One judge with a stated scope is
a cleaner claim than five with none.

> Note: the ensemble's *macro* AUC/F1 figures quoted at earlier meetings are not
> reproducible from the current evaluation artifacts — only the decision-level
> numbers above are sourced. They are omitted rather than carried forward.

---

## 4. Noise floor — the bar any delta must clear

12 seeds at a fixed label state, `sweep_tv407`. **Mean ± 2σ, test split.**
A change smaller than 2σ is not a result.

| Model | Macro AUC | Macro F1 | F1 `too_thick` | F1 `extra_parts` |
|---|---|---|---|---|
| CLIP Probe | 0.769 ±0.016 | 0.581 ±0.025 | 0.352 ±0.041 | 0.531 ±0.041 |
| DINOv2 Probe | 0.751 ±0.015 | 0.558 ±0.023 | 0.401 ±0.052 | 0.457 ±0.054 |
| ResNet-50 | 0.751 ±0.029 | 0.565 ±0.030 | 0.377 ±**0.139** | 0.416 ±0.074 |
| ViT-B/16 | 0.728 ±0.027 | 0.547 ±0.032 | 0.443 ±**0.114** | 0.421 ±0.094 |

**Consequence:** on the thin classes, effects below ~0.10 are undetectable at 183
test pairs. `too_thick` has 26 positives and a 2σ of up to 0.139 — wider than most
effects worth reporting. Two rounds of conclusions were retracted for ignoring this.

Regenerate with `research/baselines/summarize_sweep.py <sweep_root> --markdown`.

---

## 5. Label quality — the central finding, now replicated

Re-verification measured how wrong the original labels were. **Noise:signal =
labels corrected ÷ true positives.**

| Option | Test (183 pairs) | | Train (511 new pairs) | | Direction of train error |
|---|---|---|---|---|---|
| | positives | noise:signal | positives | noise:signal | |
| `broken_lines` | 38 | 0.95 | 83 | **1.06** | over-flagged (76 of 88 were 1→0) |
| `too_thick` | 26 | **1.27** | 131 | **1.04** | under-flagged (97 of 136 were 0→1) |
| `extra_parts` | 50 | 0.66 | 206 | 0.69 | under-flagged (129 of 142 were 0→1) |
| `missing_parts` | 68 | 0.47 | 230 | 0.42 | mixed |
| `missing_texture` | 135 | 0.13 | 320 | 0.21 | mixed |

### 5a. Noise rank is the exact inverse of performance rank

| Option | Noise rank (worst first) | AUC rank (best first) | Beats always-clean? |
|---|---|---|---|
| `broken_lines` | 1 | 5 | **no — 3 of 5 significantly worse** |
| `too_thick` | 2 | 4 | no |
| `extra_parts` | 3 | 3 | no |
| `missing_parts` | 4 | 2 | yes, 2 of 5 |
| `missing_texture` | 5 | 1 | **yes, all 5 decisively** |

The option with the cleanest labels is the one every judge does well on; the
noisiest is the one every judge fails. **The fleet's weakness on fine structure is
a data problem before it is an architecture problem.**

### 5b. New this period

| Finding | Detail |
|---|---|
| **Replication** | Same rank order on 511 pairs / 2,555 decisions, disjoint from test and 2.8× its size. Previously a single-split observation. |
| **Noise is directional** | Not symmetric magnitude. `extra_parts` and `too_thick` were systematically under-flagged; `broken_lines` over-flagged. |
| **Testable consequence** | `too_thick` train positives 166→224, `broken_lines` 237→173. Both are served by one 2-output head (`QL`), so it was being pulled in opposite directions. Prediction: **QL moves most in the retrain**, judged against §4. |

---

## 6. Ruled out — closed with measurement

| Question | Test | Result | Status |
|---|---|---|---|
| Is input resolution the bottleneck? | 448px vs 896px, matched recipe and seed | val macro-F1 **0.6568 vs 0.6569** | falsified |
| Is model capacity the bottleneck? | 2B vs 7B, matched recipe | dead tie in-domain | falsified |
| Do ensemble weights help? | accuracy- and AUC-derived weights | 0 of 32 vote patterns change | inert |
| Does the ensemble beat its best member? | paired McNemar on decisions | net −14, p=0.2819 | no |
| Do panel scores predict which defect a human fixes? | GroupKFold on 155 edit steps | 0.42 vs 0.406 constant; 2 of 5 actions never predicted | no signal |
| Is the RL framing viable? | four prerequisites tested | no learnable state; reward measures 1 of 5 options; action space collapses to a lookup; n=155 | **retired** |

Full rationale for the RL retirement: `project_notes/README_prose_archive_20260807.md`.

---

## 7. In flight

| Job | What | Walltime | Status |
|---|---|---|---|
| 19279792 | `backbones` 12-seed sweep @ 918-verified | 1:05 | running |
| 19279793 | `clip` 12-seed sweep @ 918-verified | 0:45 | running |
| 19279794 | `dino` 12-seed sweep @ 918-verified | 0:25 | running |
| 19280190 | Qwen2-VL-7B LoRA @ 918-verified | 4:00 | queued |

Output: `TactileEval_Context_And_Data/sweep_tv918/`, `output_7b_tv918/`.
Walltime right-sized from measured runtimes rather than left at defaults — at
fair-share 0.16 it is the only lever on start time.

---

## 8. Next

| # | Task | Blocked on | Value |
|---|---|---|---|
| 1 | Re-measure §3 and §4 at 918-verified labels | jobs above | makes this meeting's tables comparable to next meeting's |
| 2 | Test the QL prediction (§5b) against 2σ | same | says whether label noise, not architecture, caused the `broken_lines` failure |
| 3 | New collection round in the app | starting now | class balance on `too_thick` / `broken_lines`, not raw volume |
| 4 | **Ceiling study** — double-annotate ~100 pairs with a time gap | annotation effort only, no GPU | per-option human agreement; reframes every number in §3 |
| 5 | Expert rating exercise on sampled outputs | protocol design | RQ4's second half; light protocol, no recruitment |

### 8a. Why the ceiling study is the highest-value item

| | |
|---|---|
| What it measures | per-option agreement when a careful annotator revisits the same pairs |
| Why it matters | re-verification changed 14.6% of evaluation decisions and 18–20% on fine structure. If the target itself is unstable, no model can exceed that agreement. |
| What it changes | "AUC 0.75 on `too_thick`" is weak against a perfect target and respectable against one humans reproduce ~80% of the time |
| Cost | annotation time; no compute |
| Precedent | to our knowledge no tactile-graphics QA work has established such a ceiling |

One accidental instance already exists: a pair re-annotated 2 days after its first
pass disagreed on 1 of 5 options. That is n=1 — not a measurement, but it confirms
the protocol runs on existing tooling.

### 8b. Open question for discussion

Is a measured ceiling on human agreement acceptable as a **primary** contribution,
or does Chapter 5 need a positive modelling result alongside it?

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
