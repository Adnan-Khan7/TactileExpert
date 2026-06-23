#!/usr/bin/env python3
"""
TactileExpert — Human-in-the-Loop Tactile Graphic Pipeline

Gradio interface for generating, evaluating, and iteratively refining
tactile graphics using GPT-image-1 and BANA-trained quality evaluators.

Usage:
    python app.py                        # local, port 7860
    python app.py --port 7861
    python app.py --model-path /path/to/Qwen2-VL-2B-Instruct
"""

import argparse
import base64
import io
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

from evaluators import CLIPProbeEvaluator, VLMEvaluator
from evaluators.constants import (
    ALL_OPTIONS, OPTION_DISPLAY, DEFAULT_THRESHOLD_MODE, build_result,
)
from generation.gpt_generator import build_prompt, generate_tactile, edit_tactile, BANA_TEMPLATE

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Colour scheme (two models only)
# ---------------------------------------------------------------------------

MODEL_COLOR = {
    "CLIP Probe":     "#e67e22",
    "VLM Fine-tuned": "#3498db",
}

MODE_LABELS = {
    "Balanced (t = 0.50)":          "balanced",
    "Calibrated (val F1-max)":      "calibrated",
}
MODE_DISPLAY = {v: k for k, v in MODE_LABELS.items()}
DEFAULT_MODE_LABEL = MODE_DISPLAY[DEFAULT_THRESHOLD_MODE]

ALL_CLEAR_KEYWORDS = {
    "too_thick":       ["lines too thick", "strokes merge", "reduce line width"],
    "broken_lines":    ["outline gap", "broken line", "missing edge segment"],
    "missing_parts":   ["missing body part", "incomplete structure", "omitted feature"],
    "missing_texture": ["add texture", "surface pattern missing", "region needs hatching"],
}

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _thumb_uri(pil: Image.Image, max_side: int = 220) -> str:
    """Small base64 JPEG data-URI for inline strip thumbnails."""
    img = pil.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _prob_bar(prob: float, thresh: float, flagged: bool) -> str:
    """Probability bar with a dark tick mark at the decision threshold."""
    pct   = int(prob * 100)
    tick  = int(thresh * 100)
    color = "#e74c3c" if flagged else "#27ae60"
    return (
        f"<div style='position:relative;background:#eee;border-radius:4px;height:10px;margin:3px 0'>"
        f"<div style='width:{pct}%;background:{color};height:10px;border-radius:4px'></div>"
        f"<div style='position:absolute;left:{tick}%;top:-2px;width:2px;height:14px;"
        f"background:#2c3e50' title='threshold {thresh}'></div>"
        f"</div>"
    )


def _eval_panel(model_name: str, result: dict | None) -> str:
    color = MODEL_COLOR.get(model_name, "#555")
    if result is None:
        return (
            f"<div style='border-left:4px solid {color};padding:10px;background:#fafafa;"
            f"border-radius:6px;font-family:sans-serif;color:#aaa;font-size:13px'>"
            f"<b style='color:{color}'>{model_name}</b><br>Not run yet.</div>"
        )
    n_flag  = len(result["flagged"])
    verdict = (
        "<span style='color:#27ae60;font-weight:700'>all clear</span>" if n_flag == 0
        else f"<span style='color:#e74c3c;font-weight:700'>{n_flag} issue{'s' if n_flag > 1 else ''} flagged</span>"
    )
    rows = []
    for opt in ALL_OPTIONS:
        d      = result["options"][opt]
        prob   = d["prob_yes"]
        thresh = d["threshold"]
        flag   = d["flagged"]
        label  = OPTION_DISPLAY[opt]
        badge  = "<span style='color:#e74c3c;font-weight:700'>⚠ FLAGGED</span>" if flag \
                 else "<span style='color:#27ae60;font-weight:700'>✓ clear</span>"
        rows.append(
            f"<div style='margin-bottom:8px'>"
            f"<div style='display:flex;justify-content:space-between'>"
            f"<span style='font-size:12px;font-weight:600'>{label}</span>{badge}</div>"
            f"{_prob_bar(prob, thresh, flag)}"
            f"<span style='font-size:11px;color:#555'>p(defect) = {prob:.2f}"
            f" &nbsp;·&nbsp; threshold = {thresh:.2f}"
            f" &nbsp;·&nbsp; flag if p &gt; t</span>"
            f"</div>"
        )
    body = "".join(rows)
    mode_note = MODE_DISPLAY.get(result.get("mode", ""), "")
    return (
        f"<div style='border-left:4px solid {color};padding:10px;background:#fafafa;"
        f"border-radius:6px;font-family:sans-serif;font-size:13px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
        f"<b style='color:{color};font-size:14px'>{model_name}</b>{verdict}</div>"
        f"<div style='color:#999;font-size:11px;margin-top:2px'>thresholds: {mode_note}</div>"
        f"<div style='margin-top:8px'>{body}</div></div>"
    )


def _strip_card(thumb_uri: str, label: str, badges_html: str) -> str:
    return (
        f"<div style='min-width:120px;max-width:150px;text-align:center;"
        f"font-family:sans-serif;font-size:11px'>"
        f"<img src='{thumb_uri}' style='width:110px;height:110px;object-fit:contain;"
        f"border:1px solid #ddd;border-radius:6px;background:#fff'/>"
        f"<div style='font-weight:600;margin:4px 0 3px'>{label}</div>"
        f"<div>{badges_html}</div></div>"
    )


def _iteration_strip(state: list[dict], mode: str) -> str:
    """Natural | Iter 0 | Iter 1 | ... with thumbnails and per-mode flag badges."""
    if not state:
        return ("<div style='color:#aaa;font-family:sans-serif;font-size:13px;"
                "padding:8px'>No iterations yet.</div>")
    cards = []

    nat_pil = state[-1].get("nat_pil")
    if nat_pil is not None:
        ref_badge = ("<span style='background:#7f8c8d;color:#fff;border-radius:3px;"
                     "padding:1px 5px;font-size:10px'>reference</span>")
        cards.append(_strip_card(_thumb_uri(nat_pil), "Natural", ref_badge))

    for it in state:
        clip_res, vlm_res = _results(it, mode)
        flagged = sorted(set(clip_res["flagged"]) | set(vlm_res["flagged"]),
                         key=ALL_OPTIONS.index)
        badges = " ".join(
            f"<span style='background:#e74c3c;color:#fff;border-radius:3px;"
            f"padding:1px 5px;font-size:10px;margin:1px'>{OPTION_DISPLAY[o]}</span>"
            for o in flagged
        ) or ("<span style='background:#27ae60;color:#fff;border-radius:3px;"
              "padding:1px 5px;font-size:10px'>all clear</span>")
        cards.append(_strip_card(_thumb_uri(it["pil"]), it["label"], badges))

    return (
        "<div style='display:flex;gap:12px;align-items:flex-start;"
        "flex-wrap:nowrap;overflow-x:auto;padding:6px 0'>"
        + "".join(cards) + "</div>"
    )


def _all_clear_guidance(user_keywords: str = "") -> str:
    keyword_examples = []
    for opts in ALL_CLEAR_KEYWORDS.values():
        keyword_examples.extend(opts)
    examples = " · ".join(f'"{k}"' for k in keyword_examples[:6])
    user_note = (
        f"<div style='margin-top:6px;font-size:12px;color:#555'>"
        f"Your input: <i>\"{user_keywords}\"</i></div>" if user_keywords else ""
    )
    return (
        "<div style='padding:12px 16px;background:#f0fff4;border:2px solid #27ae60;"
        "border-radius:8px;font-family:sans-serif'>"
        "<div style='font-size:15px;font-weight:700;color:#27ae60'>✓ Both models see no defects</div>"
        "<div style='margin-top:8px;font-size:13px;color:#1a1a1a;font-weight:600'>"
        "If you have a specific concern, describe it in the box below.</div>"
        "<div style='margin-top:4px;font-size:12px;color:#333'>"
        f"Useful keywords: {examples}</div>"
        f"{user_note}</div>"
    )


def _api_error_msg(stage: str, e: Exception) -> str:
    msg = str(e)
    if "billing" in msg.lower():
        return (
            f"⚠ {stage} unavailable: the OpenAI account has hit its billing "
            f"limit. Switch to <b>Eval only</b> mode (top of the left panel) "
            f"to evaluate uploaded tactile graphics without API calls. "
            f"<span style='color:#999;font-size:11px'>({msg})</span>"
        )
    return f"⚠ {stage} failed: {msg}"


# ---------------------------------------------------------------------------
# Threshold-mode plumbing
# ---------------------------------------------------------------------------
# Each iteration stores RAW probabilities plus the calibrated-threshold
# snapshots, so switching mode re-renders instantly without re-inference.

def _results(it: dict, mode: str) -> tuple[dict, dict]:
    clip_res = build_result(it["clip_probs"], it["clip_cal"], mode)
    vlm_res  = build_result(it["vlm_probs"],  it["vlm_cal"],  mode)
    return clip_res, vlm_res


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=True,
                        help="Create a public gradio.live URL (default on — "
                             "cluster nodes are not reachable from outside)")
    parser.add_argument("--no-share", dest="share", action="store_false")
    args = parser.parse_args()

    # Lazy-load evaluators once (heavy — keep alive for the session)
    _cache: dict = {}

    def _clip():
        if "clip" not in _cache:
            _cache["clip"] = CLIPProbeEvaluator()
        return _cache["clip"]

    def _vlm():
        if "vlm" not in _cache:
            _cache["vlm"] = VLMEvaluator(model_path=args.model_path)
        return _cache["vlm"]

    # Session state: list of iteration dicts
    # Each: {label, pil, nat_pil, nat_path, tac_path, clip_probs, vlm_probs,
    #        clip_cal, vlm_cal}
    _state: list[dict] = []

    def _mode(mode_label: str) -> str:
        return MODE_LABELS.get(mode_label, DEFAULT_THRESHOLD_MODE)

    def _evaluate_and_store(label: str, tac_pil, nat_pil, nat_path, tac_path) -> dict:
        clip_probs = _clip().score_pair(nat_path, tac_path)
        vlm_probs  = _vlm().score_pair(nat_path, tac_path)
        it = {
            "label":      label,
            "pil":        tac_pil,
            "nat_pil":    nat_pil,
            "nat_path":   nat_path,
            "tac_path":   tac_path,
            "clip_probs": clip_probs,
            "vlm_probs":  vlm_probs,
            "clip_cal":   dict(_clip().calibrated_thresholds),
            "vlm_cal":    dict(_vlm().calibrated_thresholds),
        }
        _state.append(it)
        return it

    def _render_latest(mode: str, selected_model: str, user_note: str = ""):
        """Panels, strip, all-clear box, and edit-box prefill for current state."""
        if not _state:
            return (
                _eval_panel("CLIP Probe", None),
                _eval_panel("VLM Fine-tuned", None),
                _iteration_strip([], mode),
                gr.update(value="", visible=False),
                gr.update(value=""),
            )
        clip_res, vlm_res = _results(_state[-1], mode)
        all_clear = clip_res["all_clear"] and vlm_res["all_clear"]
        sel = clip_res if selected_model == "CLIP Probe" else vlm_res
        prefill = sel["top_issue"]["edit_instruction"] if sel["top_issue"] else ""
        return (
            _eval_panel("CLIP Probe", clip_res),
            _eval_panel("VLM Fine-tuned", vlm_res),
            _iteration_strip(_state, mode),
            gr.update(value=_all_clear_guidance(user_note) if all_clear else "",
                      visible=all_clear),
            gr.update(value=prefill),
        )

    # ---- Callbacks --------------------------------------------------------

    def on_generate(nat_pil, object_desc, prompt_text, api_key_gen,
                    mode_label, selected_model):
        mode = _mode(mode_label)
        if nat_pil is None:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                "⚠ Please upload a natural reference image first.",
            )
        if not api_key_gen.strip():
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                "⚠ API key required for generation.",
            )

        prompt = prompt_text.strip() or build_prompt(object_desc or "the object")

        nat_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        nat_pil.save(nat_tmp.name, format="JPEG")
        nat_path = nat_tmp.name

        try:
            tac_pil = generate_tactile(api_key=api_key_gen.strip(), prompt=prompt)
        except Exception as e:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                _api_error_msg("Generation", e),
            )

        tac_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tac_pil.save(tac_tmp.name, format="JPEG")

        _evaluate_and_store(f"Iter {len(_state)}", tac_pil, nat_pil,
                            nat_path, tac_tmp.name)

        clip_html, vlm_html, strip, clear_box, edit_prefill = \
            _render_latest(mode, selected_model)
        return (
            tac_pil, clip_html, vlm_html, strip, clear_box,
            gr.update(interactive=True), edit_prefill, "",
        )

    def on_eval_upload(nat_pil, tac_pil, mode_label, selected_model):
        """Eval-only mode: evaluate an uploaded tactile graphic, no API calls."""
        mode = _mode(mode_label)
        if nat_pil is None:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                "⚠ Please upload a natural reference image first.",
            )
        if tac_pil is None:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                "⚠ Please upload a tactile graphic to evaluate.",
            )

        nat_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        nat_pil.save(nat_tmp.name, format="JPEG")
        tac_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tac_pil = tac_pil.convert("RGB")
        tac_pil.save(tac_tmp.name, format="JPEG")

        _evaluate_and_store(f"Iter {len(_state)}", tac_pil, nat_pil,
                            nat_tmp.name, tac_tmp.name)

        clip_html, vlm_html, strip, clear_box, edit_prefill = \
            _render_latest(mode, selected_model)
        return (
            tac_pil, clip_html, vlm_html, strip, clear_box,
            gr.update(interactive=True), edit_prefill, "",
        )

    def on_edit(edit_text, user_note, api_key_edit, include_nat,
                mode_label, selected_model):
        mode = _mode(mode_label)
        if not _state:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(),
                "⚠ No iteration to edit yet — generate first.",
            )
        if not api_key_edit.strip():
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(),
                "⚠ API key required for editing.",
            )

        last = _state[-1]
        instruction = edit_text.strip()
        if user_note.strip():
            instruction = (instruction + " " + user_note.strip()).strip()
        if not instruction:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(),
                "⚠ Please provide an edit instruction.",
            )

        try:
            new_tac = edit_tactile(
                api_key=api_key_edit.strip(),
                current_tactile=last["pil"],
                edit_instruction=instruction,
                natural_reference=last["nat_pil"] if include_nat else None,
            )
        except Exception as e:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(),
                _api_error_msg("Edit", e),
            )

        tac_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        new_tac.save(tac_tmp.name, format="JPEG")

        _evaluate_and_store(f"Iter {len(_state)}", new_tac, last["nat_pil"],
                            last["nat_path"], tac_tmp.name)

        clip_html, vlm_html, strip, clear_box, edit_prefill = \
            _render_latest(mode, selected_model, user_note)
        return (
            new_tac, clip_html, vlm_html, strip, clear_box, edit_prefill, "",
        )

    def on_view_change(mode_label, selected_model):
        """Re-render everything when threshold mode or trusted model changes."""
        clip_html, vlm_html, strip, clear_box, edit_prefill = \
            _render_latest(_mode(mode_label), selected_model)
        return clip_html, vlm_html, strip, clear_box, edit_prefill

    def on_reset():
        _state.clear()
        return (
            None, None,
            _eval_panel("CLIP Probe", None),
            _eval_panel("VLM Fine-tuned", None),
            _iteration_strip([], DEFAULT_THRESHOLD_MODE),
            gr.update(value="", visible=False),
            gr.update(interactive=False),
            "", "",
        )

    # ---- UI ---------------------------------------------------------------

    with gr.Blocks(
        title="TactileExpert",
        theme=gr.themes.Soft(),
        css=".gradio-container{max-width:1300px!important}",
    ) as demo:

        gr.Markdown("""
# TactileExpert — Tactile Graphic Pipeline
Upload a natural reference photograph, generate a tactile graphic via GPT-image-1,
evaluate it with CLIP Probe and VLM Fine-tuned, then iteratively refine.
""")

        err_box = gr.HTML(value="", visible=True)

        with gr.Row():

            # ---- Left column: inputs ----------------------------------------
            with gr.Column(scale=1, min_width=320):

                ui_mode = gr.Radio(
                    choices=["Full pipeline", "Eval only"],
                    value="Full pipeline",
                    label="Mode",
                    info="Full pipeline: generate + edit via GPT-image-1. "
                         "Eval only: upload tactile graphics and run the "
                         "evaluators — no API calls (testing / development).",
                )

                gr.Markdown("### 1 — Reference Image")
                nat_img = gr.Image(label="Natural Reference Photograph (mandatory)",
                                   type="pil", height=220)

                with gr.Group(visible=True) as gen_group:
                    gr.Markdown("### 2 — Generation prompt")
                    object_desc = gr.Textbox(
                        label="Object name / description",
                        placeholder="e.g. a horse, a bicycle, a potted cactus",
                        lines=1,
                    )
                    prompt_box = gr.Textbox(
                        label="Full BANA prompt (auto-filled from object name — editable)",
                        value=BANA_TEMPLATE.format(object_description="the object"),
                        lines=8,
                    )
                    object_desc.change(
                        fn=lambda d: build_prompt(d) if d.strip() else BANA_TEMPLATE.format(object_description="the object"),
                        inputs=object_desc,
                        outputs=prompt_box,
                    )

                    api_key_gen = gr.Textbox(
                        label="OpenAI API key — Generation (call 1)",
                        placeholder="sk-...",
                        type="password",
                        lines=1,
                    )
                    gen_btn = gr.Button("Generate tactile graphic", variant="primary")

                with gr.Group(visible=False) as upload_group:
                    gr.Markdown("### 2 — Tactile graphic to evaluate")
                    tac_upload = gr.Image(
                        label="Tactile graphic (upload each new iteration here)",
                        type="pil", height=220,
                    )
                    eval_btn = gr.Button("Evaluate tactile graphic",
                                         variant="primary")

                with gr.Group(visible=True) as edit_group:
                    gr.Markdown("### 3 — Edit")

                    model_radio = gr.Radio(
                        choices=["CLIP Probe", "VLM Fine-tuned"],
                        value="CLIP Probe",
                        label="Use verdict from",
                    )
                    edit_box = gr.Textbox(
                        label="Edit instruction (auto-filled — editable)",
                        placeholder="Will populate after generation…",
                        lines=3,
                    )
                    user_note = gr.Textbox(
                        label="Your additional feedback (optional — appended to instruction)",
                        placeholder="e.g. the tail texture is too faint, add diagonal hatching",
                        lines=2,
                    )
                    include_nat_cb = gr.Checkbox(
                        label="Include natural reference photo in the edit call "
                              "(grounds the correction in the real object)",
                        value=True,
                    )
                    api_key_edit = gr.Textbox(
                        label="OpenAI API key — Editing (call 2)",
                        placeholder="sk-...",
                        type="password",
                        lines=1,
                    )
                    edit_btn = gr.Button("Apply edit", variant="secondary",
                                         interactive=False)

                gr.Markdown("---")
                reset_btn = gr.Button("Reset session", variant="stop", size="sm")

            # ---- Right column: outputs --------------------------------------
            with gr.Column(scale=2):

                gr.Markdown("### Current tactile candidate")
                tac_out = gr.Image(label="Latest tactile graphic", height=300)

                gr.Markdown("### Evaluation")
                mode_radio = gr.Radio(
                    choices=list(MODE_LABELS.keys()),
                    value=DEFAULT_MODE_LABEL,
                    label="Decision thresholds",
                    info="Balanced: every option flags at p > 0.50. "
                         "Calibrated: per-option thresholds tuned for max F1 on "
                         "the validation split (tick mark on each bar).",
                )
                with gr.Row():
                    clip_panel = gr.HTML(_eval_panel("CLIP Probe", None))
                    vlm_panel  = gr.HTML(_eval_panel("VLM Fine-tuned", None))

                all_clear_box = gr.HTML(value="", visible=False)

                gr.Markdown("### Iteration history")
                strip_html = gr.HTML(_iteration_strip([], DEFAULT_THRESHOLD_MODE))

        # ---- Wire up --------------------------------------------------------

        gen_btn.click(
            fn=on_generate,
            inputs=[nat_img, object_desc, prompt_box, api_key_gen,
                    mode_radio, model_radio],
            outputs=[tac_out, clip_panel, vlm_panel, strip_html,
                     all_clear_box, edit_btn, edit_box, err_box],
        )

        eval_btn.click(
            fn=on_eval_upload,
            inputs=[nat_img, tac_upload, mode_radio, model_radio],
            outputs=[tac_out, clip_panel, vlm_panel, strip_html,
                     all_clear_box, edit_btn, edit_box, err_box],
        )

        ui_mode.change(
            fn=lambda m: (
                gr.update(visible=m == "Full pipeline"),
                gr.update(visible=m != "Full pipeline"),
                gr.update(visible=m == "Full pipeline"),
            ),
            inputs=ui_mode,
            outputs=[gen_group, upload_group, edit_group],
        )

        edit_btn.click(
            fn=on_edit,
            inputs=[edit_box, user_note, api_key_edit, include_nat_cb,
                    mode_radio, model_radio],
            outputs=[tac_out, clip_panel, vlm_panel, strip_html,
                     all_clear_box, edit_box, err_box],
        )

        for ctrl in (mode_radio, model_radio):
            ctrl.change(
                fn=on_view_change,
                inputs=[mode_radio, model_radio],
                outputs=[clip_panel, vlm_panel, strip_html, all_clear_box, edit_box],
            )

        reset_btn.click(
            fn=on_reset,
            outputs=[nat_img, tac_out, clip_panel, vlm_panel, strip_html,
                     all_clear_box, edit_btn, edit_box, user_note],
        )

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
