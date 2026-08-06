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
- **Panel scores carry no triage signal** — 0.42 accuracy against a 0.406
  constant, macro F1 0.22, and two of the five actions never predicted at all.
  See the RL section below.
- **Combiner weights cannot move a decision** (above).

### What the test set cannot resolve

With 183 test pairs — 26 positives for `too_thick`, 50 for `extra_parts` — and
measured seed variance of σ≈0.02–0.07 on per-option F1, **effects smaller than
roughly 0.10 are not detectable** on the thin classes. Twelve-seed sweeps at a
fixed label state establish this floor, and any claim in this repository is
quoted against it.

---

## Why the reinforcement-learning framing is retired

An earlier design proposed learning an **agent** that watches the editing loop
and decides what to fix next. This section records why that was abandoned, and
on what evidence, so the decision is auditable rather than a change of subject.

### What the proposal was, in plain terms

Reinforcement learning needs four pieces:

| piece | plain meaning | here |
|---|---|---|
| **policy** | the thing that decides | a model choosing the next repair |
| **state** | what it sees before deciding | the five judges' scores for the current image |
| **action** | what it picks | which of the five defects to fix next |
| **reward** | the score saying whether that was good | how much the judges' defect scores dropped after the edit |

Formally the reward was `r = φ(qₜ) − φ(qₜ₊₁)`, where `φ` is the mean defect
probability across the fleet. In words: *the edit is rewarded in proportion to
how much the automatic judges think it improved the image.*

A policy can only be learned if **the state predicts the action** and **the
reward tracks real quality**. Both were tested. Both failed.

### Failure 1 — the state carries no signal

*What it means:* if what the model sees before deciding tells you nothing about
what a human decides, no policy can learn the mapping, however good the model.

*Experiment:* [`research/policy/test_triage_signal.py`](research/policy/test_triage_signal.py)
takes the panel's scores for an image and tries to predict which defect the
annotator actually repaired next, over 155 recorded edit steps. Validation is
`GroupKFold` on trajectory id, because steps within one session share an image
lineage and a naive split would leak.

*Result:*

```
accuracy 0.42   vs   0.406 for always guessing the most common action
macro F1 0.22
broken_lines   F1 0.00   (16 cases — never once predicted correctly)
missing_parts  F1 0.00   ( 4 cases)
```

1.4 points over a constant. The classifier learned to name the two most frequent
actions and ignore the rest. **The observation an agent would condition on does
not predict the behaviour it is meant to imitate.**

### Failure 2 — the reward measures one defect

*What it means:* the reward defines what the agent optimises. If it is only
sensitive to part of the problem, the agent optimises that part.

*Evidence:* `φ` averages the fleet's five per-option probabilities. Measured
against always-clean on the verified test set, the fleet detects
`missing_texture` decisively, `missing_parts` for one judge only, and is
**significantly worse than answering "clean"** on `too_thick` and
`broken_lines`. So `φ` is close to a `missing_texture` detector with noise
attached, and an agent maximising it would learn to chase texture.

The reward is also only meaningful if the probabilities are **calibrated** —
a score of 0.30 should mean a 30% chance the defect is present. The fleet was
trained with asymmetric and BCE losses and never calibrated, and the calibration
constants that did exist were fitted against a superseded fleet.

### Failure 3 — the action space collapses

*What it means:* if the "decision" is a lookup, learning it is not a policy
problem.

*Evidence:* the annotator's real action is choosing **which defect to repair**,
after which the instruction text is a near-deterministic template for that
choice. Behaviour cloning on the text therefore learns a five-entry lookup
table, and DPO — which learns from preferred-versus-rejected pairs — would be
comparing five hand-written strings. The scaffold was built
(`research/policy/`) and confirmed the training machinery runs; it demonstrated
nothing about the task.

### Failure 4 — sample size

155 edit steps from 82 sessions. Small for supervised learning on five classes;
far too small for policy learning.

### Conclusion

Any one of these is survivable. Together they say the formulation does not fit
the problem: **no learnable state, a reward that measures one fifth of the
objective, a degenerate action space, and n=155.** The honest framing is a
pre-registered negative — the prerequisites were specified, tested, and two of
them failed outright.

---

## Research directions

Three, in the order the evidence supports them.

### 1. Label stability as the object of study

Re-verification changed **260 of the 1,785 decisions** on the two evaluation
splits — 14.6% overall, and 18–20% on the fine-structure criteria. (286 labels
changed in total once the then-verified train pairs are included.) If a careful annotator revisiting the same images disagrees with the
earlier judgement that often, then **the target itself is not stable**, and no
model can exceed that agreement.

That reframes every model number in this repository. "AUC 0.75 on `too_thick`"
is weak against a perfect target and respectable against a target humans
reproduce ~80% of the time. Making the claim requires measuring the ceiling
properly: double-annotate a subset with a deliberate time gap, compute per-option
agreement, and report every result against it.

This turns the project's most awkward finding into its headline, and to our
knowledge no tactile-graphics QA work has established such a ceiling.

### 2. Representation, not capacity or resolution

The obvious explanations for the fine-structure failures have been eliminated:

- **capacity** — 2B versus 7B at matched recipe was a dead tie in-domain
- **resolution** — 448px versus 896px, matched recipe and seed, produced val
  macro-F1 of 0.6568 versus 0.6569

What remains untested is the representation. The probes reduce each image to a
**single global CLS vector**, and that vector is then asked whether a 3-pixel
gap exists somewhere in the outline. Global semantic embeddings are the right
tool for "does this region have texture" and the wrong tool for "is this line
continuous" — which is exactly the pattern in the per-option results.

The experiment is cheap and makes a falsifiable prediction: dense patch features
or a spatially-aligned comparison should improve `broken_lines` and `too_thick`
and leave `missing_texture` unchanged. If it does not, the last obvious
modelling door is closed and the remaining explanation is label noise.

### 3. Reposition the system around what it can actually do

The current claim — an automatic evaluator fleet — is broader than the evidence.
The defensible claim is **pre-screening for one high-prevalence defect**.
`missing_texture` is 74% positive in library material, and the VLM reaches
library-only AUC **0.941** on it, without the source-shortcut inflation that
affects the vision judges. Catching the most common defect reliably is real
workflow value for a transcriber, and it is a claim that survives scrutiny.

The five-judge ensemble should be dropped from that story: it is arithmetically
a 3-of-5 majority vote, and it performs significantly *worse* than its own best
member.

---

## Chapter 5: how to tackle it, as of now

Working notes for discussion — not a settled plan.

### Where the proposal currently commits us

The proposal's RQ4 asks whether the integrated system *"yields a workflow that
measurably improves with use and that tactile experts find usable and
trustworthy."* Two halves, and they are in different shape:

- *"measurably improves with use"* — this is the policy-learning claim, and the
  evidence above says it does not hold in that form.
- *"experts find it usable and trustworthy"* — untouched, and still legitimately
  future work behind the CUREB timeline.

The phrase appears in three places in Chapter 1 (thesis statement, RQ4,
contribution 4) and nowhere in Chapters 2–4. So the exposure is small and
localised, but it is worth resolving deliberately rather than discovering it
during a defence.

### A framing the evidence can carry

> When generation, evaluation and editing are integrated into a human-in-the-loop
> workflow, **which parts of expert quality judgement can be reliably automated,
> and what bounds the rest?**

This keeps the system — it exists, it is deployed, it collected the corpus — and
replaces a promise of a positive result with a question the data answers. It is
also the safer thing to put in front of a committee: a proposal that promises to
*determine* a limit cannot fail in the way one that promises an improvement can.

### A possible section skeleton

| section | what it contains | status |
|---|---|---|
| The deployed system | interface, collection protocol, the v1 dataset | **done** |
| What is automatable | per-option, against always-clean, paired significance | **done** |
| What bounds the rest | label stability; noise rank inversely predicts performance rank | measured, needs the ceiling study |
| What was ruled out | resolution, capacity, ensembling, RL prerequisites | **done** |
| Expert validation | transcriber study | future work, CUREB-gated |

Four of five rest on results already in hand. The gap is the ceiling
measurement, which is annotation work rather than compute.

### Chapter 4 hands this chapter a thread worth pulling

Chapter 4 lists as a limitation that *"re-scoring edits with the diagnosing probe
is a proxy rather than an independent quality measure."* That circularity is
exactly what breaks the RL reward — `φ` averages the same fleet that made the
diagnosis. Chapter 5 can report that the limitation Chapter 4 anticipated was
measured, and its consequence quantified. That is a stronger transition than
introducing the negative result cold.

### Open questions for discussion

- Reword RQ4 now, or write Chapter 5 and reconcile afterwards?
- Is a measured ceiling on human agreement an acceptable *primary* contribution,
  or does the chapter need a positive modelling result alongside it?
- Does the retired RL framework belong in Chapter 5 as a pre-registered negative,
  or in Chapter 6 as a closed direction?

---

## Growing the dataset through continued use

The collection interface is still live, and every session it runs adds data.
This is an ongoing, low-effort task that can accumulate alongside the writing.
Notes on what it is worth, and what to watch.

### What one session yields

A photograph, a generated tactile rendering, the fleet's scores, the expert's
chosen repair, the edited image, and a final human label across all five
criteria — plus every intermediate iteration.

### What more of it actually buys

**Measurement resolution, mainly.** The binding constraint on almost every
question in this repository is that test has 183 pairs and only 26 `too_thick`
positives. Nothing detects an effect below ~0.10 on the thin classes. Roughly
four times the positives would halve that bar. That is the difference between
"we could not measure it" and an answer.

**A defensible ceiling.** Double-annotating a subset with a deliberate gap gives
per-option human agreement. That number reframes every model result and does not
depend on collecting much at all — it is the highest value per unit of effort.

**Usability evidence.** Sessions record what the expert actually did.
[`research/policy/audit_override_rate.py`](research/policy/audit_override_rate.py)
measures how often the panel's top flag matches the defect they chose to repair.
That is a direct trust measure, it needs no GPU, and it survives the retirement
of the policy framing — the agreement rate is interesting whether or not anything
learns from it.

**What it does not buy:** trajectories for policy learning. That direction is
closed on grounds that more of the same data does not reopen.

### Things to watch

- **Class balance beats volume.** Sessions sample the natural defect
  distribution, so `too_thick` positives arrive slowly. Deliberately seeding
  images likely to produce thick or broken lines is far more efficient than
  collecting generally.
- **Source drift.** Library and generated material have very different base
  rates — `missing_texture` is 88% positive in library pairs and 3% in generated
  ones. New sessions are all generated, so the corpus balance shifts as it grows,
  and pooled metrics shift with it. Report per source when it matters.
- **In-flow labels are noisier than reviewed ones.** The label-noise finding came
  from labels collected during editing. New sessions inherit that. Budget a
  verification pass for anything destined for an evaluation split, or treat
  fresh sessions as training data only.
- **Inferred positives are action labels, not diagnoses.** A pre-edit image
  marked `too_thick` records what the expert chose to fix, not the only thing
  wrong with it. 37.5% of them were corrected on review.
- **Keep sessions whole across splits.** Iterations share a photograph with their
  final, so a session split across train and test leaks. Enforced in
  `build_unified_splits.py` and in the calibration session filter.

### How this could complement Chapter 5

Loosely, and without committing to any of it:

If enough sessions accumulate to grow the evaluation set, the dose-response
question becomes answerable — does verified label fraction predict evaluator
quality? Right now that is invisible at this test size, and a clean answer would
turn the label-stability argument from an inference into a demonstration.

If the override rate turns out to be high, the pre-screening claim strengthens
considerably: the panel's top flag agreeing with expert judgement is close to
what a transcriber would actually want from the tool. If it turns out low, that
is worth knowing before an expert study is designed around it.

And if the corpus grows enough to support a genuinely held-out *second* test set,
every result here could be replicated rather than merely reported — which for a
chapter built substantially on negative results would be the single most
strengthening thing available.

None of this needs deciding now. The collection runs, the data accumulates, and
these become answerable in the order the sample size allows.

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

Dataset v1 is complete; both evaluation splits are 100% human-verified. Train
verification is partial (44.3%) and paused. The measured label-error rate
suggests the remaining pairs still limit what the judges can learn — but see the
power note above: at this test-set size that limitation is not directly
observable, which is itself part of the finding.

The fleet is trained, installed and provenance-stamped (seed, split directory
and augmentation mode are recorded in every checkpoint). Thresholds are fitted
on val, never on test, and `derive_val_thresholds.py` refuses `--split test`.

Open questions for discussion are in **Research directions** above.
