# Edit-policy scaffold (Ch5 Loop 3)

Learns the **edit-suggestion policy**: given a natural photo, the current tactile
graphic, the panel's defect scores and a target defect, write the repair
instruction a human would have written.

> **This is pipeline validation, not an H2 result.** 115 SFT records from 57
> trajectories and 44 DPO pairs cannot support a claim about policy quality.
> H2 is gated for December. Report these runs as evidence the loop closes.

## Pipeline

```
generated_training_data/trajectories.jsonl
        │  extract_policy_data.py        (already run — 157 BC / 57 DPO)
        ▼
experiments_out/policy_data/{bc,dpo}.jsonl
        │  build_policy_dataset.py       (CPU)
        ▼
experiments_out/policy_data/{sft,dpo_pairs}/{train,val,test}.jsonl
        │  train_bc_policy.py            (GPU — LoRA SFT)
        ▼
output_policy_bc/best_checkpoint
        │  train_dpo_policy.py --init-from …   (GPU — DPO)
        ▼
output_policy_dpo/best_checkpoint
```

Run the whole chain:

```bash
cd "$TACTILE_DATA_ROOT" && sbatch ~/TactileExpert/research/policy/train_policy.sh
```

Validate first, on a login node, without touching a GPU or downloading weights:

```bash
python research/policy/build_policy_dataset.py
python research/policy/train_bc_policy.py  --dry-run
python research/policy/train_dpo_policy.py --dry-run
```

## Splits

| | train | val | test |
|---|---|---|---|
| trajectories | 57 | 12 | 13 |
| SFT records | 115 | 17 | 25 |
| DPO pairs | 44 | 8 | 5 |

Splits are assigned **per trajectory, once, over the union of the BC and DPO
trajectory sets**, and both datasets inherit that assignment. This matters: 157
BC records come from only 82 trajectories, 57 DPO pairs from 36, and the DPO
trajectories are a strict subset of the BC ones. Splitting by row — or splitting
the two datasets independently — puts the same editing session on both sides of
the line. `build_policy_dataset.py` asserts against both failure modes and exits
non-zero if either occurs.

The test sets are small enough (25 records / 5 pairs) that they support a
sanity check and nothing more. Do not report a test number from 5 pairs as a
finding.

## Decisions already made

**Reward filter — keep all 57 DPO pairs.** 19 of 57 have negative calibrated
reward on the chosen (human) instruction. DPO learns the *preference* between
chosen and rejected, which does not require the panel's defect mass to have
fallen; filtering would cut 44 training pairs to ~29 on an already-thin set.
`--min-reward` runs the alternative. Note that exactly one pair flips sign
between raw and calibrated reward, so raw-vs-calibrated is immaterial here.

**Separate system prompt.** The policy writes instructions; the judge answers
yes/no. Reusing `evaluators/constants.py:SYSTEM_PROMPT` would train the policy
to speak as a judge, so `build_policy_dataset.py` defines its own.

**No `trl`.** The cluster venv has `peft` but not `trl`, `datasets` or
`bitsandbytes`. The DPO objective is written out directly and the reference
policy is the same weights with the LoRA adapter disabled, so only one model is
ever resident.

## Data shape you need to know about

Measured 2026-07-29 against the real artifacts.

- **Era mixing.** 102 of 157 BC records are `era="ensemble"`, **55 are
  `"pre-ensemble"`** — collected before the ensemble existed, so their prefill
  provenance differs. All 57 DPO pairs are ensemble-era, so DPO is clean; SFT is
  not. `--era ensemble` drops the 55 if you decide they do not belong.
- **History is mostly empty.** 82 of 157 records have `history: []`. Half the
  set gives the policy no session context.
- **`target_option` has a 6th value `"multiple"`** (2 records) that is not in
  `ALL_OPTIONS`. It is carried as a literal string; nothing one-hots it.
- **`missing_texture` is the largest target class** (63 of 157) and is exactly
  the option whose training and inference prompts encode *different concepts*
  (backlog item C1). Treat conclusions drawn from those records with care.

## First run — job 18704731, 2026-07-29 (2B, 9m17s, MaxRSS 6.7G)

The loop closes. Both stages train, checkpoint and early-stop correctly.

```
BC    best epoch 8 (of 8)   val_loss 2.6235   test_loss 2.5418  ppl 12.70
DPO   best epoch 3 (of 4)   val_loss 1.1207   val margin −0.4044
      baseline −1.2287 → −0.9053 → −0.6689 → −0.4044   (epoch 4 regressed)
```

**Read the val margin, not the test line.** Three findings qualify these numbers:

**1. BC is trained on the text DPO must reject.** 13 of 115 SFT training records
(11.3%) are *verbatim* `EDIT_INSTRUCTIONS` templates — they enter the BC set
because when the annotator accepted a prefill unchanged, `extract_policy_data.py`
still emits a BC record whose `instruction` is that template. Those records also
average 239 chars against 114 for the rest, so they carry outsized token weight.
This is why the DPO baseline margin starts at **−1.2287**: BC has already raised
the likelihood of the template language that DPO scores as `rejected`. The two
objectives are partly fighting. DPO recovers about two-thirds of it but does not
cross zero within 4 epochs.

**2. The test split does not measure the same task.** 3 of 5 test pairs are
template-vs-template (60%) against 3 of 44 in train (7%), and in all three the
chosen text is the *correct* template for the stated target option while the
rejected is another option's template. A policy conditioned on `Target defect to
repair: X` only has to match X to its template. Hence `pref_accuracy 1.0`,
`margin +12.8076` — an artifact, not generalization. **Do not cite the test
numbers.** With 5 pairs the split cannot be rebalanced; it can only be reported
as uninformative.

**3. `pref_accuracy` on val is 4 of 8.** It cannot distinguish a working policy
from a coin. Ignore it.

The rejected-vs-chosen length imbalance also differs sharply by split
(mean `len(rejected) − len(chosen)`: train +103, val +89, **test +0.8**), so a
length heuristic that helps on train/val does nothing on test.

Incidentally, the test pairs are a clean illustration of backlog item **C3**: the
prefill shown was the `broken_lines` template while the annotator was addressing
`missing_texture`, because the card and the dropdown sort flagged options by
different keys.

## Controlled variant — job 18710250, `--drop-accepted-prefill` (2B, 10m10s)

Tests finding 1 directly. The builder drops SFT records whose instruction is a
verbatim template (13 of 115 train, 19 across all splits); **the DPO pairs and
the trajectory split assignment are held identical**, so any change in the DPO
numbers is attributable to the BC training data alone.

```
                              baseline        --drop-accepted-prefill
SFT train records                 115                    102
DPO pairs (unchanged)          44/8/5                 44/8/5

DPO baseline mean_margin      −1.2287                +1.4983      <- sign flips
DPO baseline loss              1.1754                 0.9344
best val_loss             1.1207 (ep3)           0.8905 (ep4)
val margin at best            −0.4044                +2.2720
margin trajectory   −0.91 −0.67 −0.40 −0.58   +1.89 +2.16 +2.23 +2.27
```

**Conclusion: confirmed.** Training BC on accepted-prefill records is what drove
the preference margin negative. Removed, the BC policy already prefers the human
instruction over the template before DPO starts, and DPO builds on that instead
of spending its budget reversing it. Every comparable DPO metric improves.

**What this does NOT show.** `pref_accuracy` is 0.5 — 4 of 8 — in *both* runs and
did not move despite the 2.7 margin swing. The mean margin is therefore being
carried by magnitude on a few pairs, not by more pairs being ranked correctly.
On 8 val pairs, mean margin is not a robust statistic. The defensible claim is
about sign and direction under a controlled change; the claim that the variant
policy ranks human instructions better pair-for-pair is NOT supported.

**The test split inverts.** The variant scores *worse* on test (accuracy 0.8,
margin 5.80) than the baseline (1.0, 12.81) — because the test split is 60%
template-vs-template and so rewards exactly the template bias the variant
removes. This is independent confirmation that the test numbers measure the
wrong thing. Cite neither.

BC losses are **not** comparable across the two runs: dropping records shrinks
the BC val and test sets too (17→15, 25→21), so 2.6235 vs 2.6929 is a different
denominator, not a regression.

Both DPO runs hit their 4-epoch cap (the variant was still improving at epoch 4),
as did both BC runs at 8. Nothing here was selected by early stopping.

## Open, not decided

- **Whether `--drop-accepted-prefill` becomes the default.** The variant above
  settles the mechanism but not the policy question: dropping the records costs
  11% of a thin SFT set to remove a bias whose measured effect is on an 8-pair
  val margin. My read is that it should be the default — the two objectives
  demonstrably fight otherwise — but it is a judgement call, and the supporting
  statistic is weak. Re-decide once collection resumes and the sets are larger.
- **Every run hit its epoch cap** — BC at 8, DPO at 4, in both configurations,
  and the variant's DPO was still improving at epoch 4. Nothing was selected by
  early stopping, so none of these is a converged number. `--epochs 12 / 8`
  would settle it, at roughly 15 GPU-minutes.
- Whether to condition on `era`, or drop the 55 pre-ensemble records.
- Whether the reward belongs in the SFT objective at all (currently it is
  metadata only; `--min-reward` on the builder would filter instead).
- No generation-quality metric yet. Both trainers early-stop on loss, which
  says nothing about whether the produced instructions are *good*. The honest
  next step is scoring generated instructions with the deployed fleet — i.e.
  does applying the policy's instruction reduce defect mass — which needs
  gpt-image-1 calls and therefore a collection session.
