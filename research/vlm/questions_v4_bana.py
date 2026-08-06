"""PARKED — BANA-aligned question set (C1 candidate, 2026-07-30). NOT ACTIVE.

Nothing imports this. It exists so the wording and its rationale survive the
revert described below.

WHY IT IS PARKED
----------------
These questions were written to close C1: the deployed VLM was trained with one
set of questions (research/vlm/build_vqa_v2.py) and asked a different set at
inference (evaluators/constants.py). Three of five differed, as did the system
prompt, and two differences changed what was being measured rather than just the
wording.

A 7B judge was retrained on them (job 18753124 -> output_c1_7b, dataset
data/vqa_v4_bana) as a controlled comparison against the deployed output_7b:
same corpus, same hyperparameters, same base model, prompts the only variable.

RESULT (experiments_out/eval_c1_gate.json, gated at the deployed t=0.50):

    in-domain test  macro F1  0.597 -> 0.624   net +15/765   p=0.091
    GPT holdout     correct    132  ->  138    net  +6/150   p=0.109
    always-clean baseline                      142/150

Neither improvement is significant. And the holdout gain is CONSERVATISM, not
better detection — flags raised fell 16 -> 8, false positives 13 -> 6, but true
positives ALSO fell 3/8 -> 2/8. The judge moved toward the always-clean baseline
by behaving more like it, and still does not clear it.

Per-option, `broken_lines` OVERSHOT: it now flags 0 times on the holdout against
2 positives. The carve-out for dashed styles and crossing breaks worked and then
went too far. Fix that wording before any future attempt.

So the deployed configuration was left untouched: adopting a change with no
measurable benefit, at the cost of the weights/calibration cascade plus a known
broken_lines regression, is churn. Revisit when the holdout has more than 8
positives — i.e. after the synthetic-corruption collection — so a difference can
actually be measured.

TO REACTIVATE
-------------
1. Fix the broken_lines wording (it under-flags).
2. Copy QUESTIONS_V4 and SYSTEM_PROMPT_V4 over QUESTIONS and SYSTEM_PROMPT in
   evaluators/constants.py.
3. Re-apply the build_vqa_v2.py change that IMPORTS them from constants.py
   rather than restating them — that is what makes the drift structurally
   impossible, and it is the part of C1 worth keeping regardless of wording.
4. Retrain (research/vlm/train_qwen2vl_c1_7b.sh) and re-gate
   (research/experiments/gate_c1_vlm.sh, which now runs McNemar automatically).
5. If it wins, run the cascade: rederive_fleet.py then recalibrate.

BANA sections cited are from the Guidelines and Standards for Tactile Graphics
(2010), https://www.brailleauthority.org/tg/web-manual/tgmanual.html
"""

QUESTIONS_V4 = {
    # BANA 3.4.3.3 (dashed lines are a legitimate style) and 2.11 (a break where
    # lines cross improves clarity) — so only UNINTENDED gaps count.
    # !! THIS ONE OVERSHOT — 0 flags against 2 holdout positives. Revise. !!
    "broken_lines": (
        "Ignoring texture fills, and ignoring deliberate design choices such as "
        "a dashed line style or a small break where one line crosses another, "
        "does any outline of the object have unintended gaps or missing "
        "segments, so that a fingertip tracing that edge would lose the line?"
    ),
    # BANA 2.11 (clutter) and 5.9.3.2 / 3.4.3.12 (minimum 1/8 inch separation).
    "too_thick": (
        "Are any lines or filled areas so thick, or so close together, that "
        "adjacent components merge and can no longer be told apart by touch, or "
        "that fine detail collapses into a solid mass?"
    ),
    # BANA 2.11 (adjacent areas must be distinguishable), 3.2.2 (up to five area
    # textures), 3.4.3.1 (raising small areas for tactual contrast). Phrased on
    # ADJACENCY, which covers both no-fill-anywhere and
    # fill-present-but-undifferentiated in one condition — the annotator's actual
    # criterion, which is BANA depth-compliance rather than presence/absence.
    "missing_texture": (
        "Do any two adjacent regions of the object that represent different "
        "surfaces, materials, or depths carry the same fill, or no fill at all, "
        "so that a fingertip cannot tell where one region ends and the next "
        "begins?"
    ),
    # BANA 2.8 (simplify if intent is not compromised), 3.6.1 (do not
    # over-simplify), 3.7.1. The v3 question asked for parts "present in the
    # reference photograph", which penalised the deliberate background removal
    # the annotator performs — hence 0/15 follow-through on strong flags.
    "missing_parts": (
        "Is any part essential to recognising and understanding the object "
        "absent, so that its omission compromises the intent of the graphic? "
        "Omitted background, scenery, and non-essential detail are expected "
        "simplifications, not defects."
    ),
    # BANA 2.5 (decorative borders omitted), 2.6 (frames omitted), 3.7.1.
    # Deliberately does NOT say "not present in the reference": background IS in
    # the reference and must still be removed, and that clause excluded the case
    # the annotator flags most often.
    "extra_parts": (
        "Does the graphic include anything that does not belong to the object "
        "itself — background, ground, scenery, frames or borders, text or logos, "
        "or stray clutter lines — adding tactile clutter without serving the "
        "concept?"
    ),
}

SYSTEM_PROMPT_V4 = (
    "You are an expert evaluator of tactile graphics for blind and visually "
    "impaired readers, working to the BANA (Braille Authority of North America) "
    "Guidelines and Standards for Tactile Graphics. Tactile graphics are embossed "
    "line drawings explored by touch. You are shown two images: the first is a "
    "natural reference photograph and the second is a tactile graphic derived "
    "from it. Simplification of the reference is expected and correct — omitted "
    "background, scenery and non-essential detail are not defects. "
    "Answer the quality assessment question with 'yes' or 'no' only."
)
