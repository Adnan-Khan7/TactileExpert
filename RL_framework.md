# The RL-Inspired Framework of TactileExpert — Reference Notes

*Reinforcment Learning Inspiration for TactileExpert.*

---

## 0. About this document

- **Section 1** is the 90-second pitch — read it first; it's what you say out loud.
- **Section 2** is a gentle RL primer, every concept tied to *our* system.
- **Section 3** is the honest "why *RL-inspired*, not *RL*" framing — this is
  the single most important section for surviving questions.
- **Sections 4–8** are the technical meat: the formalization, the reward, the
  three learning loops, and the two learning algorithms (behavior cloning, DPO).
- **Section 9** is anticipated questions with crisp answers — your safety net.
- **Section 10** is the glossary.

The through-line to keep in your head: **the system turns every human editing
session into training signal, and learns from that logged experience to (a)
judge better, (b) combine judges better, and (c) eventually suggest edits
better.** That "improve a decision-making policy from logged experience of
acting and being rewarded" is the reinforcement-learning idea we borrow.

---

## 1. The 90-second pitch (say this first)

> Chapter 5 is a human-in-the-loop system, TactileExpert, that generates a
> tactile graphic, has a panel of models evaluate it for defects, turns those
> defects into a plain-language edit instruction, applies the edit, and repeats
> until a human accepts the result. The research contribution is not the
> plumbing — it's that **the system learns from its own use.** Every session is
> logged as a sequence of (state, action, outcome) steps, exactly the structure
> reinforcement learning uses to describe an agent improving through experience.
> We exploit that structure in three learning loops: the evaluators retrain on
> collected data behind a deployment gate; the way we combine the evaluators is
> re-derived from where they agree or disagree with human labels; and — the main
> proposed work — the edit-suggestion policy is trained to imitate and then
> exceed the human's own instructions using the preference data we log every
> time a user rewrites a suggestion. We call it *RL-inspired* because we borrow
> RL's sequential-decision framing and its "learn a policy from rewarded
> experience" goal, while using modern offline learning methods (imitation
> learning and Direct Preference Optimization) rather than classical online RL.

---

## 2. Reinforcement learning in five minutes (for a non-RL audience)

Reinforcement learning studies an **agent** that makes a **sequence of
decisions** in an **environment** to maximize a **reward**. The vocabulary:

| RL term | Plain meaning | In TactileExpert |
|---|---|---|
| **Agent** | The decision-maker being trained | The edit-suggestion policy (today: a human; proposed: a learned policy) |
| **Environment** | The world the agent acts in, which changes in response | The image-editing model + the evaluator panel that scores results |
| **State** `s` | What the agent observes before deciding | The reference photo, the current tactile drawing, and the panel's defect scores |
| **Action** `a` | The decision the agent makes | A natural-language edit instruction (e.g. "close the broken outlines") |
| **Transition** | How the state changes after an action | Applying the instruction produces a new drawing with new scores |
| **Reward** `r` | A number saying how good that step was | How much the assessed defect level dropped after the edit |
| **Policy** `π` | The agent's strategy: state → action | The rule that turns "what's wrong" into "what to type" |
| **Trajectory** (episode) | One full run from start to finish | One editing session: generate → edit → edit → accept |
| **Return** | Total reward over an episode | Total defect reduction from first draft to accepted graphic |

The **goal of RL** is to find a policy that collects the most reward over time.
Classic RL does this **online**: the agent tries actions, sees rewards, and
updates itself in a continuous loop, deliberately **exploring** to discover good
actions it hasn't tried.

**The one picture to remember:** *state → (policy chooses) action → environment
gives reward and a new state → repeat.* That loop is all RL is, at heart.

---

## 3. Why we say "RL-inspired," not "reinforcement learning"

This is the framing that protects the whole chapter. Be precise and confident
about it; do not overclaim.

**What we take from RL:**
- The **sequential-decision framing.** A tactile edit is naturally a
  multi-step decision process — you rarely fix everything in one edit — so
  state/action/reward/policy/trajectory is the *right language* for it.
- The **goal**: learn a decision policy that improves from logged **experience
  of being rewarded**, rather than from a fixed labeled dataset alone.
- The **reward-driven view of quality**: progress is measured by defect
  reduction per step, not just a final label.

**What we do *not* do (and shouldn't claim):**
- We do **not** run **online** RL with live exploration and policy-gradient
  updates in the loop. That would need thousands of trial edits per policy
  update and a fast, trustworthy automatic reward — neither is safe when each
  action is a paid image-generation call and the reward is itself a model.
- We do **not** train a separate reward model and optimize it with PPO (the
  classic RLHF recipe). Instead we use **offline** methods on **logged human
  interaction** — imitation learning and Direct Preference Optimization (DPO).

**Why this is a strength, not a hedge.** Offline learning from logged human
feedback is exactly where the field has moved (offline RL; RLHF-via-DPO in
modern LLM post-training). Our human-in-the-loop app is a *preference-data
generator*: every time a user rewrites a suggested instruction, that is a
labeled "the human's version is better than the machine's" pair — the same
signal that aligns frontier models. So "RL-inspired, realized with offline
preference learning" is a modern, defensible position, not a consolation prize.

**The honest one-liner if pushed:** *"We use the reinforcement-learning framing
to structure the problem and the reward, and offline preference-learning
methods — which are provably connected to RL from human feedback — to do the
actual policy learning. We are deliberately not doing online policy-gradient RL,
because the action is an expensive generative call and the reward is a learned
judge; online exploration there would be costly and unsafe."*

---

## 4. TactileExpert as a sequential decision process (the formalization)

Here is the framing written out with our concrete instances. This is the
skeleton of Chapter 5 §5.3.

An **episode** starts from a natural image `I`. The generator `G` produces the
first tactile candidate under a standards prompt `p`:

```
x_0 = G(I, p)
```

At each step `t`:

- **State** `s_t = (I, x_t, q_t)` where
  - `I` — the reference photo,
  - `x_t` — the current tactile drawing,
  - `q_t ∈ [0,1]^5` — the evaluator panel's defect probabilities for the five
    options (too_thick, broken_lines, missing_parts, missing_texture,
    extra_parts). This is the "what's wrong right now" vector.
- **Action** `a_t` — a natural-language repair instruction (free text).
- **Transition** — the editing model `E` applies the instruction:
  `x_{t+1} = E(x_t, a_t)`, and the panel re-scores to give `q_{t+1}`.
- **Reward** `r_t = φ(q_t) − φ(q_{t+1})`, where `φ(q) = mean of the five
  probabilities` (an overall "defect mass"). Positive reward = the edit reduced
  assessed defects; negative = it made things worse.
- **Terminal signal** — when the human accepts, they tick the final defect
  checkboxes, giving ground-truth labels `y ∈ {0,1}^5` for the finished graphic.

**Worked example (a real pattern from our logs):**
> A `horse` graphic is generated. The panel flags `missing_texture` at high
> probability (the limbs are untextured). State = (horse photo, current drawing,
> scores). The human's action: *"Texture the hind limbs and forelimbs with
> different patterns so depth is perceived by touch."* The editor applies it;
> the panel re-scores and `missing_texture` probability drops from 0.74 to 0.20.
> Reward for that step ≈ the drop in mean defect mass. The session continues
> until the human accepts and checks the final labels.

**What gets logged per step** (this is the raw material for all learning):
the two images, the panel's per-model scores, the instruction the human
actually sent, the instruction the system had *pre-filled* (suggested), whether
the human accepted it verbatim, and the final human labels. We already store
all of this in `generated_training_data/trajectories.jsonl`.

---

## 5. The reward function — and why it's trustworthy

**Design.** Reward = reduction in the panel's mean defect probability. It is
**dense** (available every step, not only at the end) and **automatic** (no
human needed to score each intermediate edit), which is what makes learning
from many sessions feasible.

**The obvious objection — and it's a good one:** *"The reward is the model
grading its own homework. Couldn't the edit policy learn to fool the judge
instead of actually fixing the graphic?"* This is the **reward-hacking / proxy
reward** problem, and a sharp committee member will raise it. Our four defenses:

1. **Calibrated probabilities.** A reward built on probabilities is only
   meaningful if a "0.30" really means a 30% chance of the defect. We calibrated
   the whole panel (per-model Platt scaling on collected labels); out-of-fold
   calibration error dropped from ~0.25 to ~0.03. So the reward now reflects
   genuine probability changes, not arbitrary score wobble.
2. **Independent human ground truth.** Every finished graphic gets human defect
   labels `y`. These are *independent* of the panel's self-assessment and let us
   check whether reward-driven "improvement" agrees with human judgment.
3. **The judge keeps improving (Loop 1).** The evaluators are retrained on new
   human labels behind a deployment gate, so the reward signal gets harder to
   game over time rather than staying a fixed target.
4. **The real metric is human acceptance, not reward.** When we evaluate the
   learned edit policy (Loop 3), the headline number is whether *humans* accept
   its suggestions and reach an acceptable graphic faster — the proxy reward is
   only the training-time signal, not the success criterion.

**Honest caveat to state proactively:** re-scoring an edit with the same panel
that diagnosed it is a proxy, not an independent quality measure. That is
exactly why terminal human labels and the acceptance-rate evaluation exist. (We
inherited this limitation explicitly from the TactileEval chapter and address it
here.)

---

## 6. The three learning loops

The system has three things it can improve from logged experience. Two are
already operating; one is the main proposed work. A useful mental model: the
**edit policy** is the "agent," and the **evaluator panel** is both the reward
signal and part of the environment — so we improve the agent *and* the machinery
that rewards it.

### Loop 1 — Judge adaptation (OPERATING)
*"Make the evaluators better on our actual domain, safely."*

- **What it is.** Collected pairs (images + human labels) are merged into the
  training set; all five evaluators are retrained; a candidate is **deployed
  only if it beats the current one on BOTH a frozen original-domain test set and
  a frozen, growing in-domain holdout.** This gate has both *accepted* upgrades
  (three evaluators improved at one retrain) and *rejected* them (a VLM
  candidate whose validation score rose but whose holdout accuracy fell — we
  kept the incumbent).
- **Which ML idea is this?** Closest to **continual / iterative supervised
  learning with a guardrail**, not classic RL. In RL terms, it's improving the
  *environment's reward machinery* from experience. The deployment gate is the
  key methodological piece — it makes "the system learns" a controlled claim,
  not a hope.
- **What it demonstrates (Hypothesis H1).** In-domain accuracy of the panel
  improves as collected data grows. Preliminary support: a 2B judge went from
  9/45 to 33/45 correct on the holdout after fine-tuning on in-loop data, and a
  weak defect category recovered (F1 0.24 → 0.41) when its in-domain data
  doubled.

### Loop 2 — Judge aggregation from disagreement (OPERATING)
*"Learn how much to trust each evaluator, from where they agree with humans."*

- **What it is.** We combine the five evaluators with a per-defect **weighted
  vote**. The weights were originally each model's accuracy on the old test set;
  we now **re-derive them from live records** — for each saved pair we know every
  model's decision and the human's label, so we weight each model by its actual
  per-defect accuracy on our real domain.
- **Which ML idea is this?** **Ensemble weighting / stacking** informed by
  logged outcomes — a light form of meta-learning. Not RL, but it *is* "learn
  the reward machinery from experience," so it belongs to the same story.
- **What it demonstrates (Hypothesis H3).** Whether a disagreement-informed
  ensemble can match or beat its single strongest member under domain shift.
  Honest open problem: currently our best single model (the fine-tuned VLM)
  sometimes beats the weighted ensemble on live data — a vote cannot exceed its
  dominant member — so H3 may push us toward *learned* aggregation (e.g.
  per-instance routing) rather than fixed weights. Reporting this openly is a
  strength.

### Loop 3 — Learning the edit policy (PROPOSED — the main contribution)
*"Learn to suggest the edit instruction a human would actually write."*

- **The problem today.** The system pre-fills a suggested instruction from
  hand-written templates. Users rewrite them often — but the acceptance rate is
  rising as we formalize the templates from collected language, which gives us a
  strong baseline to beat and clean imitation targets.
- **Stage (i): Behavior Cloning (imitation).** Train a policy
  `π_θ(a | s)` to reproduce the *human's* instruction given the state, using the
  high-reward steps as demonstrations. (Section 7.)
- **Stage (ii): Direct Preference Optimization (DPO).** Every time a user
  rewrote a suggestion, we logged a pair: *human's instruction ≻ machine's
  suggestion*, in that state. DPO tunes the policy to prefer the better one.
  (Section 8.)
- **Stage (iii): Online A/B in the live app.** Compare the learned policy vs the
  templates on **human acceptance rate**, **reward per edit**, and
  **iterations-to-accept**.
- **What it demonstrates (Hypothesis H2).** The learned policy beats the
  template policy on acceptance and reward per edit. This is the open,
  RL-flavored experiment at the heart of the chapter.

**One-line summary of the three:** Loop 1 sharpens the *reward*, Loop 2 sharpens
how the *reward is combined*, Loop 3 learns the *policy* — together, a system
that improves along every axis from its own logged use.

---

## 7. Behavior cloning, explained simply

**Behavior cloning (BC)** is the simplest form of **imitation learning**: treat
"what action did the expert take in this state?" as an ordinary supervised
prediction problem. State goes in, the expert's action is the target, train with
a standard loss.

- **Here:** state `s = (photo, drawing, defect scores)` → predict the human's
  edit instruction. The humans are the experts; their instructions are the
  labels. We train on the steps where the edit *worked* (high reward), so we
  imitate good corrections, not mistakes.
- **Why start here.** It's stable, needs no reward model, and gives a competent
  first policy to then refine. It answers "can a model reproduce expert repair
  language from the state at all?"
- **Its known weakness (mention if asked):** *distribution shift* — a cloned
  policy can drift into states the expert never demonstrated and compound errors.
  In our setting this is softened because a human is still in the loop to catch
  bad suggestions, and because we then refine with preferences (DPO), which
  teaches the policy from its *own* rejected suggestions, not just imitation.

---

## 8. DPO (Direct Preference Optimization), explained simply

**The setup.** For each rewrite we have a **preference pair** in a given state:
a **chosen** answer (the human's instruction) and a **rejected** answer (the
machine's suggestion the human discarded).

**What DPO does.** It nudges the policy to raise the probability of the chosen
answer and lower the probability of the rejected one — a classification-style
loss over pairs, with a term that keeps the policy from drifting too far from
the imitation-trained starting point.

**Why it matters / why it's the practical choice.**
- The classic way to learn from preferences is **RLHF**: train a separate reward
  model on the pairs, then use online RL (PPO) to optimize the policy against it.
  It works but is heavy and finicky.
- **DPO** (Rafailov et al., 2023) proves you can skip the separate reward model
  and the online RL: there's a closed-form relationship between the preference
  data and the optimal policy, so you can optimize the policy **directly** from
  the pairs with a simple, stable loss. It is the method behind much of modern
  LLM alignment.
- For us this is ideal: our app **naturally emits preference pairs** (every
  rewrite), and DPO turns them into policy improvement without the cost and risk
  of online RL — which closes the circle back to Section 3's "RL-inspired,
  realized offline."

**The honest connection to RL to state out loud:** *"DPO is derived from the
RLHF objective — the same preference-optimization goal as RL from human feedback
— but solved in closed form offline. So the policy learning is RL-derived in its
mathematics even though we never run an online RL loop."*

---

## 9. Anticipated questions and crisp answers

**Q: Is this really reinforcement learning?**
A: We use the RL *framing* (state, action, reward, policy, trajectories) and an
RL-*derived* objective (DPO comes from the RLHF objective), but we deliberately
learn **offline** from logged human interaction rather than running online
policy-gradient RL. It's honestly labeled "RL-inspired." (See §3.)

**Q: Why not just do full online RL / PPO?**
A: Each action is an expensive generative edit call and the reward is itself a
learned model. Online exploration there is costly and risks reward-hacking a
fallible judge. Offline preference learning (DPO) gets the alignment benefit
from data we already collect, safely.

**Q: The reward is the model grading itself — isn't that circular?**
A: Four defenses: calibrated probabilities (ECE ~0.25 → ~0.03), independent
terminal human labels, the judge itself improving behind a deploy gate, and —
crucially — evaluating the policy on *human acceptance*, not on the proxy
reward. (See §5.)

**Q: What's actually novel? Fine-tuning and DPO aren't new.**
A: The novelty is the **closed-loop system that manufactures its own training
signal from human-in-the-loop use** in a real accessibility task, and improves
three components from it under **deployment gates** — plus the honest finding
that domain shift can make a single strong judge beat an ensemble. The
contribution is the *system and its learning framework in this domain*, not a
new optimizer.

**Q: How do you know the system is improving and not drifting?**
A: The deployment gate (Loop 1) requires beating the incumbent on two frozen
test sets before anything ships; nothing is deployed on a hope. Hypotheses
H1–H3 are each measured against frozen data.

**Q: What if the learned edit policy (H2) doesn't beat the templates?**
A: We have a decision gate (planned December). If H2 shows no advantage, the
contribution reframes around the two operating loops (judge adaptation and
aggregation) and the negative result is reported. The chapter cannot be sunk by
one hypothesis failing. (This is Risk R1 in the chapter.)

**Q: Where do the preference pairs come from, and how many?**
A: Every time a user rewrites a suggested instruction, that's one pair
(human ≻ machine). They accumulate automatically as collection continues; the
extraction tooling reports the current usable count.

**Q: Why five separate evaluators instead of one?**
A: Different model families catch different defects (structure vs. texture vs.
semantics). The ensemble is the strongest judge on the original domain; the
open question (H3) is whether that survives domain shift, which is itself a
finding.

**Q: Isn't Loop 1 just normal retraining?**
A: The retraining is standard; the **gate** is the point — a controlled,
falsifiable deployment rule that turns "the system learns" into a testable,
guarded claim, and has already *rejected* candidates.

---

## 10. Glossary

- **Agent** — the decision-maker being trained (our edit policy).
- **Policy `π`** — a mapping from state to action (what to type given what's wrong).
- **State / Action / Reward** — see §2 table.
- **Trajectory / Episode** — one full editing session, start to accepted graphic.
- **Return** — total reward over an episode (total defect reduction).
- **Online vs Offline RL** — online: learn while acting and exploring; offline:
  learn from a fixed log of past interactions. We are offline.
- **Behavior Cloning (BC)** — supervised imitation of expert actions (§7).
- **Imitation learning** — the family BC belongs to: learn to act like a demonstrator.
- **RLHF** — Reinforcement Learning from Human Feedback: reward model + online RL
  on human preferences; the heavy classic recipe.
- **DPO** — Direct Preference Optimization: learn the same preference objective
  directly from pairs, offline, no reward model (§8).
- **Reward hacking / proxy reward** — a policy scoring well on a flawed reward
  without truly solving the task; guarded against in §5.
- **Distribution shift** — the policy encountering states unlike its training
  data; BC's main weakness (§7).
- **Calibration / ECE** — whether predicted probabilities match real
  frequencies; Expected Calibration Error measures the gap. Ours: ~0.25 → ~0.03.
- **Deployment gate** — the rule that a new model ships only if it beats the
  incumbent on frozen test sets (Loop 1).
- **Ensemble weighting / stacking** — combining several models' outputs, with
  weights learned from data (Loop 2).

---

## 11. Suggested three-slide arc for the meeting

1. **The loop and the framing.** The pipeline figure (generate → evaluate →
   edit → accept) relabeled with state/action/reward/policy. Message: *this is a
   sequential decision process, and every session is logged experience.*
2. **What the system learns (the three loops).** One line each: judge adaptation
   (gated), aggregation from disagreement, edit-policy BC+DPO. Mark two as
   operating with preliminary numbers, one as the proposed core. Message: *the
   system improves from its own use, under guardrails.*
3. **Claims, evidence, and honesty.** Hypotheses H1–H3, what's already
   supported, the H2 decision gate, and the reward-trust defenses (calibration,
   human labels, acceptance-rate evaluation). Message: *falsifiable claims,
   honest limitations, a fallback if the main hypothesis fails.*
