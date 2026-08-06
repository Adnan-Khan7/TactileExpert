# TactileExpert dataset v1 — unified splits

**Status: v1 under construction.** The splits are built; label verification is in
progress. Nothing here should be cited until verification completes and the
evaluators are retrained.

## What changed from the v2 design

v2 kept two frozen evaluation sets — a library `test.jsonl` and a GPT-domain
`gpt_test.jsonl` — and gated deployment on the second. That produced two
problems that no amount of modelling could fix:

* the GPT holdout had **8 positives in 150 decisions**, so a degenerate judge
  answering *clean* on everything scored 142/150 and no member of the fleet
  cleared it;
* the gate rule was ambiguous — the script docstring said "improve both sets",
  the thesis said in-domain only.

v1 merges both corpora into a single train/val/test split.

```
                     pairs   decisions   positive   always-clean baseline
train                  918        4174      37.0%          63.0%
val                    174         870      34.0%          66.0%
test                   183         915      33.6%          66.4%
```

The headline: **the always-clean baseline on the evaluation set falls from
94.7% to 66.4%.** That came from unification alone — no examples were moved out
of train to balance classes.

## Composition

| split | library pairs | generated pairs |
|---|---|---|
| train | 688 | 230 |
| val | 146 | 28 |
| test | 153 | 30 |

Per-option positives in **test**:

| option | positives | share |
|---|---|---|
| missing_texture | 121 | 66.1% |
| missing_parts | 71 | 38.8% |
| broken_lines | 56 | 30.6% |
| too_thick | 35 | 19.1% |
| extra_parts | 24 | 13.1% |

`extra_parts` is the thinnest and `missing_texture` is skewed high — report
per-option metrics, never a single accuracy figure.

## Build rules

`research/build_unified_splits.py`, seed 42, generated split 70/15/15 by session.

1. **Library rows keep their v2 assignment.** Re-splitting them would invalidate
   every previously reported library-test number for no benefit.
2. **Generated sessions split by session, never by row.**
3. **The 30 original GPT holdout sessions are pinned to test.** They have already
   gated deployment decisions; returning any to train would be the worse leak.
4. **Finals only reach val/test.** The inferred positives (iteration images) are
   single-option, positive-only, and labelled by the *action* the annotator took
   rather than the defect observed — training signal, not ground truth. Inferred
   rows whose session landed in val/test were dropped, not moved.

## Known issues carried into v1

**Val contains 28 sessions the deployed fleet trained on**, tagged
`was_trained_on: true`. This is structural, not a mistake: every generated
session is either in train or in the pinned holdout, so a clean generated val set
cannot be built from the current 184 sessions. It resolves on retrain — the new
models will not have seen val. **Test has zero tainted pairs**, so the current
fleet can be baselined on it honestly before retraining.

**Labels carry noise.** That is the reason for the verification pass; the test
set is being confirmed pair by pair before any result is derived from it.

**104 partial-label images.** Iteration images with one option known and four
unknown. These are first in the verification queue; each one completed converts a
1-option row into a 5-option row.

## Provenance fields kept on every row

| field | meaning |
|---|---|
| `category` | `gpt_generated` vs library source — kept so per-domain breakdowns stay possible |
| `synthetic_corruption` | defect was introduced deliberately; never allowed into the holdout |
| `was_trained_on` | row was inside the corpus the *deployed* fleet trained on |

## Order of operations

1. `research/build_unified_splits.py` — done
2. `research/annotator/build_verification_queue.py` — done, 461 records
3. **label verification in `annotator.py`** — in progress
4. apply verified labels back into the splits
5. retrain the fleet on `splits_v3_unified/train`
6. evaluate on test with `research/experiments/mcnemar_gate.py` for paired
   significance — not a raw accuracy delta

Steps 5 and 6 must happen in that order. Any number from the current checkpoints
on val is optimistic until step 5 completes.
