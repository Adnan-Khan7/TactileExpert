# TactileExpert

A human-in-the-loop pipeline for producing and checking **tactile graphics** —
raised-line images that blind and low-vision readers explore by touch.

A designer supplies a photograph. The system generates a candidate tactile
rendering, a fleet of automatic judges scores it against five defect criteria,
and the designer edits and re-scores until it is clean. This repository holds
the pipeline, the judges, and the dataset used to train and measure them.

---

## The problem

A tactile graphic is not a picture you look at. It is a surface you read with a
fingertip, and that changes what counts as a defect. A line 0.4mm too thick
merges with its neighbour and the shape becomes unreadable. A 2mm gap in an
outline stops a finger tracing an edge. A texture that fills every region
uniformly gives the reader no way to tell one surface from another.

None of that is visible to standard image metrics. SSIM and FID compare pixels;
they have nothing to say about whether a fingertip can follow a contour. So
quality control is done by hand, by a small number of trained transcribers, and
it is the bottleneck in getting tactile material to readers.

This project asks a narrow, testable question: **can any part of that judgement
be automated well enough to be worth a transcriber's time?**

---

## The five criteria

Defined once in [`evaluators/constants.py`](evaluators/constants.py); every
other component imports them.

| option | what it means | test positives |
|---|---|---|
| `missing_texture` | no texture differentiation — all regions read the same by touch | 135 / 183 |
| `missing_parts` | a structural part visible in the photo is absent | 68 / 183 |
| `extra_parts` | invented structure not present in the photo | 50 / 183 |
| `broken_lines` | outlines discontinuous — a tracing finger hits gaps | 38 / 183 |
| `too_thick` | outlines so heavy that adjacent parts merge | 26 / 183 |

They are **not mutually exclusive**. A stray fragment is both a discontinuity
and something that should not be there. The panel answers five independent
binary questions.

---

## Dataset v1

`1,275` image pairs, `6,375` labelled decisions.

| split | pairs | decisions | positive | always-clean baseline | human-verified |
|---|---|---|---|---|---|
| train | 918 | 4,590 | 34.9% | 65.1% | 44.3% |
| val | 174 | 870 | 34.3% | 65.7% | **100%** |
| test | 183 | 915 | 34.6% | 65.4% | **100%** |

Two properties matter more than the totals.

**Both evaluation splits are fully re-verified.** Every test and val pair was
re-examined pair-by-pair through a confirm-or-correct interface. This was not a
formality: **286 labels changed**, 152 of them in test alone.

**The splits are balanced against each other.** After verification the val/test
boundary was re-derived by bucketing pairs on their 5-bit label signature and
splitting each bucket proportionally, which cut the largest per-option gap
between the two from 9.4% to **1.5%**.

### The label-noise finding

Re-verification measured how wrong the original labels had been, and the answer
reframes everything else in this repository:

| option | test positives | labels corrected | noise : signal |
|---|---|---|---|
| `too_thick` | 26 | 33 | **1.27×** |
| `broken_lines` | 38 | 36 | 0.95× |
| `extra_parts` | 50 | 33 | 0.66× |
| `missing_parts` | 68 | 32 | 0.47× |
| `missing_texture` | 135 | 18 | 0.13× |

On `too_thick` there were **more wrong labels than true positives**. And the
ranking of label noise is the exact inverse of the ranking of judge
performance — the option with the cleanest labels is the one every judge does
well on, and the option with the noisiest labels is the one every judge fails.

This is the central result. The fleet's weakness on fine-structure defects is
not primarily an architecture problem; it is a data problem that was invisible
until the labels were audited.

---

## The judges

Five independent scorers, each emitting a probability per option.

| judge | architecture | trained |
|---|---|---|
| CLIP probe | frozen CLIP ViT-L/14, MLP head per dimension | head only |
| DINOv2 probe | frozen DINOv2 ViT-L/14, MLP head per dimension | head only |
| ResNet-50 | two-stream, shared backbone | end-to-end |
| ViT-B/16 | two-stream, shared backbone | end-to-end |
| Qwen2-VL-7B | LoRA r=4, vision encoder frozen | adapter only |

All five take the **pair** — photograph and tactile rendering — because every
criterion is relational. The pairs are not pixel-registered: the tactile image
is a re-rendering, not an overlay, and 278 of 300 sampled pairs have different
aspect ratios.

---

## What we can and cannot claim

### The fleet clears the degenerate baseline

A judge that answers "clean" to everything scores **598/915 (65.4%)**. Under
paired McNemar tests, four of five judges beat it significantly overall.

That is a genuine improvement on the previous dataset design, where an
always-clean judge scored 142/150 on the evaluation set and **no judge could
clear it at all**. Unifying the corpora is what made the question answerable.

### But almost all of the win is one option

Per option, against always-clean:

- `missing_texture` — every judge wins decisively (+100 to +128 decisions)
- `missing_parts` — only the VLM wins significantly (+21, p=0.013)
- `too_thick`, `broken_lines` — **every vision judge is significantly *worse*
  than answering "clean"**
- `extra_parts` — nobody is significantly better

The honest summary: **the fleet reliably detects one defect out of five.**

### The ensemble does not earn its place

The weighted ensemble is **significantly worse than its own best member**
(−47 decisions, p=0.0004). And the weights are decorative: across all 32
possible vote patterns, the weighted vote is **identical to an unweighted
3-of-5 majority**, under both accuracy- and AUC-derived weights. Majority
voting lets four weaker judges outvote the one strong one.

### Negative results worth recording

- **Input resolution is not the bottleneck.** Quadrupling the VLM's pixel
  budget (448px → 896px, matched recipe, same seed) changed nothing: both arms
  peaked at epoch 4 with val macro-F1 of **0.6568 vs 0.6569**.
- **Panel scores carry no triage signal** — 41.9% against a 40.6% constant.
- **Combiner weights cannot move a decision** (above).

### What the test set cannot resolve

With 183 test pairs — 26 positives for `too_thick`, 50 for `extra_parts` — and
measured seed variance of σ≈0.02–0.07 on per-option F1, **effects smaller than
roughly 0.10 are not detectable** on the thin classes. Twelve-seed sweeps at a
fixed label state establish this floor, and any claim in this repository is
quoted against it.

---

## Layout

```
evaluators/        the five judges + constants.py (single source of truth)
app.py             the Gradio collection and editing interface
models/            trained weights (gitignored — large)
research/
  build_unified_splits.py    built v1 from the earlier two-corpus design
  restratify_v1.py           val/test rebalance on label signature
  annotator/                 verification queue, annotator UI, merge-back
  baselines/                 the four vision judges + seed sweeps
  vlm/                       VQA build + LoRA fine-tune
  experiments/               scoring, paired significance, calibration
  _retired/                  pre-v1 scripts, provenance only — do not run
```

Data, checkpoints and experiment outputs live **outside** the repository at
`$TACTILE_DATA_ROOT` (default `~/vlm_finetune`); `research/_paths.py` resolves
both roots. Nothing under `research/` hardcodes a home directory.

---

## Reproducing

```bash
export TACTILE_DATA_ROOT=~/vlm_finetune

# 1. verification queue -> annotate -> merge back
python research/annotator/build_verification_queue.py --train-finals
python research/annotator/annotator.py --annotations-file <queue> --share
python research/annotator/apply_verified_labels.py --queue <queue>          # dry run
python research/annotator/apply_verified_labels.py --queue <queue> --apply

# 2. train (seed is explicit; vary it to measure the noise floor)
sbatch research/baselines/train_backbones_v2.sh  --seed 42
sbatch research/baselines/train_clip_heads_v2.sh --seed 42
sbatch research/baselines/train_dino_probe_v2.sh --seed 42
sbatch research/vlm/train_qwen2vl_7b.sh

# 3. score and test
sbatch research/experiments/run_eval_full.sh --splits test \
    --base-model Qwen/Qwen2-VL-7B-Instruct --out experiments_out/eval_v1_fleet.json
python research/experiments/mcnemar_gate.py --gate experiments_out/eval_v1_fleet.json \
    --vs-always-clean --all-models
```

Every mutating script is **dry-run by default** and requires `--apply`.

---

## Methodological notes

These cost real time to learn and are enforced in code rather than left to
discipline.

**Report per option, never in aggregate.** The always-clean constant scores
~65%. A single accuracy figure cannot distinguish a useful judge from one that
never flags.

**Prefer AUC to F1 for comparisons.** F1 at a fixed threshold rewards
conservatism: a judge that flags less sheds false positives faster than true
positives and appears to improve. We caught exactly this twice.

**Never fit on what you report.** Calibrators and thresholds are fitted
parameters. They are fit on train-split sessions only, and `rederive_fleet.py`
refuses to do otherwise without an explicit override.

**Cache features, never labels.** Labels change with every verification pass
while an image-derived cache key does not — so a cache holding labels silently
trains a relabelled split on stale ones. Caches store features plus a record
fingerprint; labels are rebuilt every run.

**Augment the pair, not the image.** Sampling crop and flip per image
decorrelates a pair whose label is a statement about their relationship. Verified:
independent sampling agreed on orientation 190/400 times; joint sampling, 400/400.

**`set -e` in every job script.** Without it a crashed trainer still exits 0 and
SLURM reports COMPLETED — a silent failure indistinguishable from a result.

---

## Status

Dataset v1 is complete and both evaluation splits are fully verified. Train
verification is partial (44.3%) and ongoing; the measured label-error rate
suggests the remaining 511 pairs still limit what the judges can learn.

The immediate research direction follows from the negative results above: since
resolution is ruled out and label noise is quantified, the leading hypothesis
for the fine-structure failures is **representation** — the probes compress each
image to a single global CLS token and are then asked about a 3-pixel gap.
Dense patch features or a spatially-aligned comparison would address that
directly.
