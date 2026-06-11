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
import os
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

from evaluators import CLIPProbeEvaluator, VLMEvaluator
from evaluators.constants import ALL_OPTIONS, OPTION_DISPLAY, EDIT_INSTRUCTIONS
from generation.gpt_generator import build_prompt, generate_tactile, edit_tactile, BANA_TEMPLATE

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Colour scheme (two models only)
# ---------------------------------------------------------------------------

MODEL_COLOR = {
    "CLIP Probe":     "#e67e22",
    "VLM Fine-tuned": "#3498db",
}

ALL_CLEAR_KEYWORDS = {
    "too_thick":       ["lines too thick", "strokes merge", "reduce line width"],
    "broken_lines":    ["outline gap", "broken line", "missing edge segment"],
    "missing_parts":   ["missing body part", "incomplete structure", "omitted feature"],
    "missing_texture": ["add texture", "surface pattern missing", "region needs hatching"],
}

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _prob_bar(prob: float, flagged: bool) -> str:
    pct   = int(prob * 100)
    color = "#e74c3c" if flagged else "#27ae60"
    return (
        f"<div style='background:#eee;border-radius:4px;height:8px;margin:3px 0'>"
        f"<div style='width:{pct}%;background:{color};height:8px;border-radius:4px'></div>"
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
            f"<div style='margin-bottom:7px'>"
            f"<div style='display:flex;justify-content:space-between'>"
            f"<span style='font-size:12px;font-weight:600'>{label}</span>{badge}</div>"
            f"{_prob_bar(prob, flag)}"
            f"<span style='font-size:11px;color:#777'>p={prob:.3f} / t={thresh}</span>"
            f"</div>"
        )
    body = "".join(rows)
    macro = sum(d["prob_yes"] for d in result["options"].values()) / len(ALL_OPTIONS)
    return (
        f"<div style='border-left:4px solid {color};padding:10px;background:#fafafa;"
        f"border-radius:6px;font-family:sans-serif;font-size:13px'>"
        f"<b style='color:{color};font-size:14px'>{model_name}</b>"
        f"<span style='color:#999;font-size:11px;margin-left:8px'>avg p={macro:.3f}</span>"
        f"<div style='margin-top:8px'>{body}</div></div>"
    )


def _iteration_strip(iterations: list[dict]) -> str:
    if not iterations:
        return "<div style='color:#aaa;font-family:sans-serif;font-size:13px;padding:8px'>No iterations yet.</div>"
    cards = []
    for it in iterations:
        label   = it["label"]
        flagged = it.get("flagged_opts", [])
        badges  = " ".join(
            f"<span style='background:#e74c3c;color:#fff;border-radius:3px;"
            f"padding:1px 5px;font-size:10px;margin:1px'>{OPTION_DISPLAY[o]}</span>"
            for o in flagged
        ) or "<span style='background:#27ae60;color:#fff;border-radius:3px;" \
              "padding:1px 5px;font-size:10px'>all clear</span>"
        cards.append(
            f"<div style='min-width:100px;max-width:130px;text-align:center;"
            f"font-family:sans-serif;font-size:11px'>"
            f"<div style='font-weight:600;margin-bottom:3px'>{label}</div>"
            f"<div>{badges}</div></div>"
        )
    return (
        "<div style='display:flex;gap:10px;align-items:flex-start;"
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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--port", type=int, default=7860)
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
    # Each: {label, pil, nat_path, tac_path, clip_result, vlm_result, flagged_opts}
    _state: list[dict] = []

    # ---- Callbacks --------------------------------------------------------

    def on_generate(nat_pil, object_desc, prompt_text, api_key_gen):
        if nat_pil is None:
            return (
                None,
                _eval_panel("CLIP Probe", None),
                _eval_panel("VLM Fine-tuned", None),
                _iteration_strip([]),
                gr.update(value="", visible=False),
                gr.update(interactive=False),
                "⚠ Please upload a natural reference image first.",
            )
        if not api_key_gen.strip():
            return (
                None,
                _eval_panel("CLIP Probe", None),
                _eval_panel("VLM Fine-tuned", None),
                _iteration_strip(_state),
                gr.update(value="", visible=False),
                gr.update(interactive=False),
                "⚠ API key required for generation.",
            )

        prompt = prompt_text.strip() or build_prompt(object_desc or "the object")

        # Save natural image to temp file
        nat_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        nat_pil.save(nat_tmp.name, format="JPEG")
        nat_path = nat_tmp.name

        try:
            tac_pil = generate_tactile(api_key=api_key_gen.strip(), prompt=prompt)
        except Exception as e:
            return (
                None,
                _eval_panel("CLIP Probe", None),
                _eval_panel("VLM Fine-tuned", None),
                _iteration_strip(_state),
                gr.update(value="", visible=False),
                gr.update(interactive=False),
                f"⚠ Generation failed: {e}",
            )

        tac_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tac_pil.save(tac_tmp.name, format="JPEG")
        tac_path = tac_tmp.name

        clip_res = _clip().evaluate(nat_path, tac_path)
        vlm_res  = _vlm().evaluate(nat_path, tac_path)

        # Union of flagged options from both models
        flagged_union = list(
            {o for o in clip_res["flagged"]} | {o for o in vlm_res["flagged"]}
        )

        idx = len(_state)
        _state.append({
            "label":       f"Iter {idx}",
            "pil":         tac_pil,
            "nat_path":    nat_path,
            "tac_path":    tac_path,
            "clip_result": clip_res,
            "vlm_result":  vlm_res,
            "flagged_opts": sorted(flagged_union, key=lambda o: ALL_OPTIONS.index(o)),
        })

        all_clear = clip_res["all_clear"] and vlm_res["all_clear"]
        guidance  = _all_clear_guidance() if all_clear else ""

        return (
            tac_pil,
            _eval_panel("CLIP Probe", clip_res),
            _eval_panel("VLM Fine-tuned", vlm_res),
            _iteration_strip(_state),
            gr.update(value=guidance, visible=all_clear),
            gr.update(interactive=True),
            "",
        )

    def on_edit(selected_model, edit_text, user_note, api_key_edit):
        if not _state:
            return (
                None,
                _eval_panel("CLIP Probe", None),
                _eval_panel("VLM Fine-tuned", None),
                _iteration_strip([]),
                gr.update(visible=False),
                "⚠ No iteration to edit yet — generate first.",
            )
        if not api_key_edit.strip():
            return (
                None, None, None,
                _iteration_strip(_state),
                gr.update(visible=False),
                "⚠ API key required for editing.",
            )

        last = _state[-1]
        instruction = edit_text.strip()
        if user_note.strip():
            instruction = (instruction + " " + user_note.strip()).strip()
        if not instruction:
            return (
                None, None, None,
                _iteration_strip(_state),
                gr.update(visible=False),
                "⚠ Please provide an edit instruction.",
            )

        nat_pil = Image.open(last["nat_path"])
        try:
            new_tac = edit_tactile(
                api_key=api_key_edit.strip(),
                current_tactile=last["pil"],
                edit_instruction=instruction,
            )
        except Exception as e:
            return (
                None, None, None,
                _iteration_strip(_state),
                gr.update(visible=False),
                f"⚠ Edit failed: {e}",
            )

        nat_path = last["nat_path"]
        tac_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        new_tac.save(tac_tmp.name, format="JPEG")
        tac_path = tac_tmp.name

        clip_res = _clip().evaluate(nat_path, tac_path)
        vlm_res  = _vlm().evaluate(nat_path, tac_path)

        flagged_union = list(
            {o for o in clip_res["flagged"]} | {o for o in vlm_res["flagged"]}
        )

        idx = len(_state)
        _state.append({
            "label":        f"Iter {idx}",
            "pil":          new_tac,
            "nat_path":     nat_path,
            "tac_path":     tac_path,
            "clip_result":  clip_res,
            "vlm_result":   vlm_res,
            "flagged_opts": sorted(flagged_union, key=lambda o: ALL_OPTIONS.index(o)),
        })

        all_clear = clip_res["all_clear"] and vlm_res["all_clear"]
        guidance  = _all_clear_guidance(user_note) if all_clear else ""

        return (
            new_tac,
            _eval_panel("CLIP Probe", clip_res),
            _eval_panel("VLM Fine-tuned", vlm_res),
            _iteration_strip(_state),
            gr.update(value=guidance, visible=all_clear),
            "",
        )

    def on_model_select(selected_model):
        """Pre-populate edit instructions from the selected model's last result."""
        if not _state:
            return gr.update(value="")
        last = _state[-1]
        result = last["clip_result"] if selected_model == "CLIP Probe" else last["vlm_result"]
        if result["all_clear"]:
            return gr.update(value="")
        top = result["top_issue"]
        return gr.update(value=top["edit_instruction"] if top else "")

    def on_reset():
        _state.clear()
        return (
            None, None,
            _eval_panel("CLIP Probe", None),
            _eval_panel("VLM Fine-tuned", None),
            _iteration_strip([]),
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

                gr.Markdown("### 1 — Reference Image")
                nat_img = gr.Image(label="Natural Reference Photograph (mandatory)",
                                   type="pil", height=220)

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

                gr.Markdown("---")
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
                model_radio.change(fn=on_model_select, inputs=model_radio, outputs=edit_box)

                user_note = gr.Textbox(
                    label="Your additional feedback (optional — appended to instruction)",
                    placeholder="e.g. the tail texture is too faint, add diagonal hatching",
                    lines=2,
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
                with gr.Row():
                    clip_panel = gr.HTML(_eval_panel("CLIP Probe", None))
                    vlm_panel  = gr.HTML(_eval_panel("VLM Fine-tuned", None))

                all_clear_box = gr.HTML(value="", visible=False)

                gr.Markdown("### Iteration history")
                strip_html = gr.HTML(_iteration_strip([]))

        # ---- Wire up --------------------------------------------------------

        gen_btn.click(
            fn=on_generate,
            inputs=[nat_img, object_desc, prompt_box, api_key_gen],
            outputs=[tac_out, clip_panel, vlm_panel, strip_html,
                     all_clear_box, edit_btn, err_box],
        )

        edit_btn.click(
            fn=on_edit,
            inputs=[model_radio, edit_box, user_note, api_key_edit],
            outputs=[tac_out, clip_panel, vlm_panel, strip_html,
                     all_clear_box, err_box],
        )

        reset_btn.click(
            fn=on_reset,
            outputs=[nat_img, tac_out, clip_panel, vlm_panel, strip_html,
                     all_clear_box, edit_btn, edit_box, user_note],
        )

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
