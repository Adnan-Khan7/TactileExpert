#!/usr/bin/env python3
"""
TactileExpert — Human-in-the-Loop Tactile Graphic Pipeline

Changes in this version:
  1. Single API key for both generation and editing
  2. Edit instruction box is the one editable field (no separate "additional feedback")
  3. Edit log sidebar + context accumulation — previous corrections are summarised
     and prepended to each new edit call (GPT-image-1 has no text memory)
  4. Iteration Gallery (click any thumbnail to expand full-size) + badge strip
  5. Save-to-Training section — annotate and save approved pairs to a local JSONL
  6. Per-option evaluation table showing all selected models side by side
  7. Model selection checkboxes — run only the models you want
  8. Session export (JSON) — download the full session log

Usage:
    python app.py
    python app.py --port 7861
    python app.py --no-share
"""

import argparse
import base64
import io
import json
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

from evaluators import (CLIPProbeEvaluator, VLMEvaluator,
                        ResNetEvaluator, ViTEvaluator, DINOProbeEvaluator)
from evaluators.constants import (
    ALL_OPTIONS, OPTION_DISPLAY, DEFAULT_THRESHOLD_MODE,
    EDIT_INSTRUCTIONS, build_result,
    ENSEMBLE_NAME, build_ensemble_result,
)
from generation.gpt_generator import (
    build_prompt, generate_tactile, edit_tactile, BANA_TEMPLATE,
)

HERE      = Path(__file__).resolve().parent
TRAIN_DIR = HERE / "generated_training_data"
TRAJ_FILE = TRAIN_DIR / "trajectories.jsonl"

AVAILABLE_MODELS = ["CLIP Probe", "VLM Fine-tuned", "ResNet-50", "ViT-B/16", "DINOv2 Probe",
                    ENSEMBLE_NAME]

MODEL_COLOR = {
    ENSEMBLE_NAME:    "#f1c40f",
    "CLIP Probe":     "#e67e22",
    "VLM Fine-tuned": "#3498db",
    "ResNet-50":      "#8e44ad",
    "ViT-B/16":       "#27ae60",
    "DINOv2 Probe":   "#c0392b",
}

MODE_LABELS = {
    "Balanced (t = 0.50)":    "balanced",
    "Calibrated (val F1-max)": "calibrated",
}
MODE_DISPLAY    = {v: k for k, v in MODE_LABELS.items()}
DEFAULT_MODE_LBL = MODE_DISPLAY[DEFAULT_THRESHOLD_MODE]

# Map model name → (probs_key, cal_key) in the iteration dict
_MODEL_KEYS = {
    "CLIP Probe":     ("clip_probs",   "clip_cal"),
    "VLM Fine-tuned": ("vlm_probs",    "vlm_cal"),
    "ResNet-50":      ("resnet_probs", "resnet_cal"),
    "ViT-B/16":       ("vit_probs",    "vit_cal"),
    "DINOv2 Probe":   ("dino_probs",   "dino_cal"),
}


# ── Trajectory helpers ────────────────────────────────────────────────────────

def _iter_scores(it: dict, selected_models: list) -> dict:
    """Per-model probability dicts for one iteration state entry."""
    scores = {}
    for model in selected_models:
        if model not in _MODEL_KEYS:   # Ensemble — derived, not stored per-model
            continue
        probs = it.get(_MODEL_KEYS[model][0], {})
        if probs:
            scores[model] = {opt: round(probs.get(opt, 0.0), 4) for opt in ALL_OPTIONS}
    return scores


def _mean_defect_prob(scores: dict) -> float:
    """Mean defect probability across all models and all options."""
    vals = [p for mp in scores.values() for p in mp.values()]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _thumb_uri(pil: Image.Image, size: int = 200) -> str:
    img = pil.copy()
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _prob_bar(prob: float, thresh: float, flagged: bool) -> str:
    pct   = int(prob * 100)
    tick  = int(thresh * 100)
    color = "#e74c3c" if flagged else "#27ae60"
    return (
        f"<div style='position:relative;background:#eee;border-radius:4px;"
        f"height:8px;margin:3px 0'>"
        f"<div style='width:{pct}%;background:{color};height:8px;border-radius:4px'></div>"
        f"<div style='position:absolute;left:{tick}%;top:-2px;width:2px;height:12px;"
        f"background:#2c3e50' title='threshold {thresh:.2f}'></div>"
        f"</div>"
    )


def _consensus_badge(flagged_by: list[str], total_models: int) -> str:
    n = len(flagged_by)
    if n == 0:
        return "<span style='background:#27ae60;color:#fff;border-radius:3px;" \
               "padding:1px 6px;font-size:11px'>ALL CLEAR</span>"
    if n == total_models:
        return "<span style='background:#c0392b;color:#fff;border-radius:3px;" \
               "padding:1px 6px;font-size:11px'>ALL FLAG</span>"
    model_short = {
        "CLIP Probe":     "CLIP",
        "VLM Fine-tuned": "VLM",
        "ResNet-50":      "ResNet",
        "ViT-B/16":       "ViT",
        "DINOv2 Probe":   "DINO",
    }
    who = " + ".join(model_short.get(m, m) for m in flagged_by)
    return (f"<span style='background:#e67e22;color:#fff;border-radius:3px;"
            f"padding:1px 6px;font-size:11px'>{who} only</span>")


def _eval_table(model_results: dict, mode: str) -> str:
    """Per-option table showing all selected models side by side."""
    if not model_results:
        return ("<div style='color:#aaa;font-family:sans-serif;font-size:13px;"
                "padding:8px'>No evaluation results yet.</div>")

    models = list(model_results.keys())
    total  = len(models)

    # Model verdict line (top)
    header_parts = []
    for m, res in model_results.items():
        color = MODEL_COLOR.get(m, "#555")
        n = len(res["flagged"])
        verdict = ("all clear" if n == 0
                   else f"{n} issue{'s' if n > 1 else ''} flagged")
        verdict_color = "#27ae60" if n == 0 else "#e74c3c"
        header_parts.append(
            f"<span style='color:{color};font-weight:700'>{m}</span>: "
            f"<span style='color:{verdict_color}'>{verdict}</span>"
        )
    header_html = "&nbsp;&nbsp;|&nbsp;&nbsp;".join(header_parts)

    # Column headers
    col_w = f"{int(56 / total)}%"
    head_row = (
        f"<tr style='background:#f0f0f0'>"
        f"<th style='text-align:left;padding:5px 8px;width:28%;font-size:12px'>Option</th>"
        + "".join(
            "<th style='text-align:center;padding:5px;width:{};font-size:12px;"
            "color:{}'>{}</th>".format(col_w, MODEL_COLOR.get(m, "#555"), m)
            for m in models
        )
        + f"<th style='text-align:center;padding:5px;width:16%;font-size:12px'>Consensus</th>"
        f"</tr>"
    )

    # One row per option. The Ensemble is itself a consensus, so it is shown
    # as a column but excluded from the Consensus tally.
    real_models = [m for m in models if m != ENSEMBLE_NAME]
    rows = []
    for opt in ALL_OPTIONS:
        label = OPTION_DISPLAY[opt]
        flagged_by = [m for m in real_models if opt in model_results[m]["flagged"]]

        cells = []
        for m in models:
            d     = model_results[m]["options"][opt]
            prob  = d["prob_yes"]
            thresh = d["threshold"]
            flag  = d["flagged"]
            badge = (
                "<span style='color:#e74c3c;font-size:10px;font-weight:700'>⚠ FLAG</span>"
                if flag else
                "<span style='color:#27ae60;font-size:10px'>✓</span>"
            )
            if "vote_share" in d:
                # Ensemble cell: the flag tests the weighted VOTE share, not the
                # mean probability — display the quantity actually decided on.
                stats_line = (f"vote={d['vote_share']:.2f} · {d['n_agree']}/{d['n_models']}"
                              f"<br>p̄={prob:.2f}")
                bar = _prob_bar(d["vote_share"], 0.5, flag)
            else:
                stats_line = f"p={prob:.2f} t={thresh:.2f}"
                bar = _prob_bar(prob, thresh, flag)
            cells.append(
                f"<td style='padding:4px 6px;text-align:center'>"
                f"{badge}<br>"
                f"{bar}"
                f"<span style='font-size:10px;color:#666'>{stats_line}</span>"
                f"</td>"
            )

        rows.append(
            f"<tr style='border-bottom:1px solid #f0f0f0'>"
            f"<td style='padding:5px 8px;font-size:12px;font-weight:600'>{label}</td>"
            + "".join(cells)
            + f"<td style='padding:4px 6px;text-align:center'>"
              f"{_consensus_badge(flagged_by, len(real_models))}</td>"
            f"</tr>"
        )

    table = (
        f"<div style='font-family:sans-serif'>"
        f"<div style='font-size:13px;margin-bottom:8px'>{header_html}</div>"
        f"<table style='width:100%;border-collapse:collapse;border:1px solid #e5e5e5;"
        f"border-radius:6px;overflow:hidden'>"
        f"{head_row}{''.join(rows)}</table></div>"
    )
    return table


def _ensemble_card(mr: dict, mode: str, obj: str = "") -> str:
    """Default simplified view: ensemble verdict + suggested edits by priority.

    obj: known object name — substituted into the displayed suggestions so
    text copied from the card matches the edit-box prefill convention.
    """
    if ENSEMBLE_NAME not in mr:
        return ""
    res = mr[ENSEMBLE_NAME]
    flagged = sorted(res["flagged"],
                     key=lambda o: -res["options"][o].get("vote_share", 0))
    if not flagged:
        verdict = ("<span style='color:#27ae60;font-size:16px;font-weight:700'>"
                   "✓ All clear — no defects flagged by the weighted ensemble</span>")
        body = ""
    else:
        verdict = (f"<span style='color:#e74c3c;font-size:16px;font-weight:700'>"
                   f"⚠ {len(flagged)} issue{'s' if len(flagged) > 1 else ''} flagged"
                   f"</span> <span style='font-size:12px;color:#888'>— by priority</span>")
        items = []
        for i, o in enumerate(flagged):
            d = res["options"][o]
            inst = d["edit_instruction"] or ""
            if obj:
                inst = inst.replace("the object", f"the {obj}")
            items.append(
                f"<div style='margin:7px 0;padding:8px 10px;background:#fdf6f6;"
                f"border-left:3px solid #e74c3c;border-radius:4px'>"
                f"<span style='font-weight:700;font-size:13px'>{i+1}. {OPTION_DISPLAY[o]}</span>"
                f"<span style='font-size:11px;color:#888'> &nbsp;{d['n_agree']}/{d['n_models']} models agree · "
                f"vote {d['vote_share']:.0%} · mean p {d['prob_yes']:.2f}</span><br>"
                f"<span style='font-size:12px;color:#444'>{inst}</span>"
                f"</div>")
        body = "".join(items)
    clean = [OPTION_DISPLAY[o] for o in ALL_OPTIONS
             if o in res["options"] and o not in res["flagged"]]
    clean_line = (f"<div style='font-size:11px;color:#27ae60;margin-top:6px'>"
                  f"Clear: {', '.join(clean)}</div>" if clean else "")
    return (f"<div style='font-family:sans-serif;border:1px solid #f1c40f;border-radius:8px;"
            f"padding:12px 14px;background:#fffdf5'>"
            f"<div style='font-size:11px;color:#b7950b;font-weight:700;margin-bottom:4px'>"
            f"ENSEMBLE (WEIGHTED) · {mode} thresholds</div>"
            f"{verdict}{body}{clean_line}</div>")


def _eval_view(mr: dict, mode: str, obj: str = "") -> str:
    """Ensemble card up front; full per-model decision table behind a toggle."""
    table = _eval_table(mr, mode)
    card = _ensemble_card(mr, mode, obj)
    if not card:
        return table
    return (card +
            "<details style='margin-top:8px;font-family:sans-serif'>"
            "<summary style='cursor:pointer;color:#888;font-size:12px'>"
            "Show full per-model decision table</summary>"
            f"<div style='margin-top:6px'>{table}</div></details>")


def _iteration_strip(state: list[dict], model_results_list: list[dict]) -> str:
    """Compact badge-only strip below the Gallery."""
    if not state:
        return "<div style='color:#aaa;font-size:12px;font-family:sans-serif'>No iterations yet.</div>"

    cards = []
    for i, (it, mr) in enumerate(zip(state, model_results_list)):
        all_flagged = sorted(
            {o for res in mr.values() for o in res["flagged"]},
            key=ALL_OPTIONS.index,
        )
        badges = " ".join(
            f"<span style='background:#e74c3c;color:#fff;border-radius:3px;"
            f"padding:1px 4px;font-size:10px'>{OPTION_DISPLAY[o]}</span>"
            for o in all_flagged
        ) or "<span style='background:#27ae60;color:#fff;border-radius:3px;" \
             "padding:1px 4px;font-size:10px'>all clear</span>"
        cards.append(
            f"<div style='min-width:100px;text-align:center;font-family:sans-serif;"
            f"font-size:11px'>"
            f"<div style='font-weight:600;margin-bottom:3px'>{it['label']}</div>"
            f"<div>{badges}</div></div>"
        )

    return (
        "<div style='display:flex;gap:10px;flex-wrap:nowrap;overflow-x:auto;"
        "padding:6px 0;align-items:flex-start'>"
        + "".join(cards) + "</div>"
    )


def _edit_log_html(history: list[dict]) -> str:
    if not history:
        return ("<div style='color:#aaa;font-family:sans-serif;font-size:12px;"
                "padding:6px'>No edits yet.</div>")

    items = "".join(
        f"<div style='margin-bottom:6px;padding:6px 8px;background:#f8f9fa;"
        f"border-radius:4px;border-left:3px solid #3498db'>"
        f"<span style='font-weight:700;color:#555;font-size:11px'>{h['label']}</span><br>"
        f"<span style='font-size:12px;color:#222'>{h['instruction']}</span>"
        f"</div>"
        for h in history
    )

    context = _build_context(history)
    ctx_html = ""
    if context:
        ctx_html = (
            f"<div style='margin-top:8px;padding:6px 8px;background:#fff3cd;"
            f"border-radius:4px;font-size:11px;color:#555'>"
            f"<b>Context sent to next API call:</b><br>"
            f"<span style='color:#333'>{context[:300]}{'…' if len(context)>300 else ''}</span>"
            f"</div>"
        )

    return (
        f"<div style='font-family:sans-serif;max-height:280px;overflow-y:auto;"
        f"padding:4px'>{items}{ctx_html}</div>"
    )


def _build_context(history: list[dict]) -> str:
    """Compact context prefix built from the edit log, injected into next API call."""
    if not history:
        return ""
    entries = "; ".join(f"{h['label']}: {h['instruction']}" for h in history)
    return (
        f"Context — previous corrections already applied to this image: {entries}. "
        f"Do NOT redo these fixes. Focus only on any remaining issues."
    )


def _all_clear_html() -> str:
    return (
        "<div style='padding:12px 16px;background:#f0fff4;border:2px solid #27ae60;"
        "border-radius:8px;font-family:sans-serif'>"
        "<div style='font-size:15px;font-weight:700;color:#27ae60'>✓ All selected models see no defects</div>"
        "<div style='margin-top:6px;font-size:13px;color:#333'>"
        "You can continue iterating if you see a remaining issue, or save this "
        "pair to the training dataset.</div></div>"
    )


def _api_error(stage: str, e: Exception) -> str:
    msg = str(e)
    if "billing" in msg.lower():
        return (
            f"⚠ {stage} blocked: OpenAI billing limit reached. "
            f"Switch to <b>Eval only</b> mode to test without API calls. "
            f"<span style='color:#999;font-size:11px'>({msg})</span>"
        )
    return f"⚠ {stage} failed: {msg}"


# ── Session helpers ───────────────────────────────────────────────────────────

def _get_results(it: dict, mode: str, selected: list[str]) -> dict:
    """Return {model_name: result_dict} for selected models present in this iteration.

    The weighted Ensemble is a derived pseudo-model: computed from whatever
    real models are present (weights renormalised over that subset).
    """
    out = {}
    for m in selected:
        if m not in _MODEL_KEYS:
            continue
        pk, ck = _MODEL_KEYS[m]
        if it.get(pk):
            out[m] = build_result(it[pk], it[ck], mode)
    if ENSEMBLE_NAME in selected and len(out) >= 2:
        out[ENSEMBLE_NAME] = build_ensemble_result(out, mode)
    return out


# ── App ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=True)
    parser.add_argument("--no-share", dest="share", action="store_false")
    args = parser.parse_args()

    # Evaluator cache — CLIP loaded eagerly at startup (needed for first-upload
    # object detection); all others lazy (only loaded when selected by user).
    _cache: dict = {}

    def _clip():
        if "clip" not in _cache:
            _cache["clip"] = CLIPProbeEvaluator()
        return _cache["clip"]

    def _vlm():
        if "vlm" not in _cache:
            _cache["vlm"] = VLMEvaluator(model_path=args.model_path)
        return _cache["vlm"]

    def _resnet():
        if "resnet" not in _cache:
            _cache["resnet"] = ResNetEvaluator()
        return _cache["resnet"]

    def _vit():
        if "vit" not in _cache:
            _cache["vit"] = ViTEvaluator()
        return _cache["vit"]

    def _dino():
        if "dino" not in _cache:
            _cache["dino"] = DINOProbeEvaluator()
        return _cache["dino"]

    print("[TactileExpert] Loading CLIP Probe (needed for object detection)…")
    _clip()
    print("[TactileExpert] Ready — starting UI.")

    # Session state
    _state:       list[dict] = []   # per-iteration data
    _edit_history: list[dict] = []  # {label, instruction} per edit

    def _mode(lbl: str) -> str:
        return MODE_LABELS.get(lbl, DEFAULT_THRESHOLD_MODE)

    def _run_evaluators(nat_path: str, tac_path: str,
                        selected: list[str]) -> dict:
        out = {}
        if "CLIP Probe" in selected:
            out["clip_probs"] = _clip().score_pair(nat_path, tac_path)
            out["clip_cal"]   = dict(_clip().calibrated_thresholds)
        if "VLM Fine-tuned" in selected:
            out["vlm_probs"] = _vlm().score_pair(nat_path, tac_path)
            out["vlm_cal"]   = dict(_vlm().calibrated_thresholds)
        if "ResNet-50" in selected:
            out["resnet_probs"] = _resnet().score_pair(nat_path, tac_path)
            out["resnet_cal"]   = dict(_resnet().calibrated_thresholds)
        if "ViT-B/16" in selected:
            out["vit_probs"] = _vit().score_pair(nat_path, tac_path)
            out["vit_cal"]   = dict(_vit().calibrated_thresholds)
        if "DINOv2 Probe" in selected:
            out["dino_probs"] = _dino().score_pair(nat_path, tac_path)
            out["dino_cal"]   = dict(_dino().calibrated_thresholds)
        return out

    def _store_iteration(label, tac_pil, nat_pil, nat_path, tac_path,
                         eval_data: dict, instruction: str = "",
                         object_name: str = "", target_option: str = "",
                         prefill_instruction: str = "",
                         prefill_trusted: str = "") -> dict:
        it = {
            "label":         label,
            "pil":           tac_pil,
            "nat_pil":       nat_pil,
            "nat_path":      nat_path,
            "tac_path":      tac_path,
            "instruction":   instruction,
            "object":        object_name,
            "target_option": target_option,
            "prefill_instruction": prefill_instruction,
            "prefill_trusted":     prefill_trusted,
            "timestamp":   datetime.now().isoformat(timespec="seconds"),
            # Filled from eval_data (only models that were selected):
            "clip_probs":   eval_data.get("clip_probs",   {}),
            "clip_cal":     eval_data.get("clip_cal",     {}),
            "vlm_probs":    eval_data.get("vlm_probs",    {}),
            "vlm_cal":      eval_data.get("vlm_cal",      {}),
            "resnet_probs": eval_data.get("resnet_probs", {}),
            "resnet_cal":   eval_data.get("resnet_cal",   {}),
            "vit_probs":    eval_data.get("vit_probs",    {}),
            "vit_cal":      eval_data.get("vit_cal",      {}),
            "dino_probs":   eval_data.get("dino_probs",   {}),
            "dino_cal":     eval_data.get("dino_cal",     {}),
        }
        _state.append(it)
        return it

    def _gallery_data() -> list:
        """PIL images + captions for gr.Gallery."""
        items = []
        if _state and _state[-1].get("nat_pil"):
            items.append((_state[-1]["nat_pil"], "Natural reference"))
        for it in _state:
            items.append((it["pil"], it["label"]))
        return items

    # Prefill shown to the user at the time they type an edit — captured so the
    # (prefill, actual instruction) pair can be logged as DPO preference data.
    _prefill_ctx = {"text": "", "trusted": ""}

    def _render(mode_lbl: str, selected: list[str], trusted: str = "",
                combine_all: bool = False):
        mode = _mode(mode_lbl)
        if not _state:
            _prefill_ctx["text"] = ""
            _prefill_ctx["trusted"] = ""
            return (
                _eval_view({}, mode),
                gr.update(value="", visible=False),
                _iteration_strip([], []),
                _edit_log_html([]),
                _gallery_data(),
                gr.update(value=""),
                gr.update(value=None),
            )
        it = _state[-1]
        mr = _get_results(it, mode, selected)
        obj = (it.get("object") or "").strip()

        # All-model-results list for badge strip
        mr_list = [_get_results(s, mode, selected) for s in _state]

        all_clear = all(r["all_clear"] for r in mr.values()) if mr else False

        # Use trusted model, combine ALL flagged issues into prefill
        prefill = ""
        top_option = None
        trusted_model = trusted if trusted in mr else (
            next((m for m in selected if m in mr), None))
        if trusted_model and mr.get(trusted_model):
            flagged_opts = mr[trusted_model]["flagged"]
            instructions = [
                mr[trusted_model]["options"][opt]["edit_instruction"]
                for opt in flagged_opts
                if mr[trusted_model]["options"][opt].get("edit_instruction")
            ]
            if instructions:
                if combine_all and len(instructions) > 1:
                    prefill = " ".join(
                        f"{i+1}. {inst}" for i, inst in enumerate(instructions))
                else:
                    # Default: top-priority issue only — one targeted edit at a time
                    prefill = instructions[0]
            top_option = flagged_opts[0] if flagged_opts else None

        # Ground generic template wording in the known object name
        if prefill and obj:
            prefill = prefill.replace("the object", f"the {obj}")
        _prefill_ctx["text"]    = prefill
        _prefill_ctx["trusted"] = trusted_model or ""

        return (
            _eval_view(mr, mode, obj),
            gr.update(value=_all_clear_html() if all_clear else "",
                      visible=all_clear),
            _iteration_strip(_state, mr_list),
            _edit_log_html(_edit_history),
            _gallery_data(),
            gr.update(value=prefill),
            gr.update(value=top_option),
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def on_generate(nat_pil, object_desc, prompt_text, api_key,
                    mode_lbl, selected_models, trusted_model, combine_all):
        if nat_pil is None:
            return (gr.update(),) * 8 + ("⚠ Upload a natural reference image first.",)
        if not api_key.strip():
            return (gr.update(),) * 8 + ("⚠ API key required.",)

        prompt = prompt_text.strip() or build_prompt(object_desc or "the object")

        nat_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        nat_tmp.close()
        nat_pil.save(nat_tmp.name, format="JPEG")

        try:
            tac_pil = generate_tactile(api_key=api_key.strip(), prompt=prompt,
                                       natural_reference=nat_pil)
        except Exception as e:
            return (gr.update(),) * 8 + (_api_error("Generation", e),)

        tac_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tac_tmp.close()
        tac_pil.save(tac_tmp.name, format="JPEG")

        eval_data = _run_evaluators(nat_tmp.name, tac_tmp.name, selected_models)
        _store_iteration(f"Iter {len(_state)}", tac_pil, nat_pil,
                         nat_tmp.name, tac_tmp.name, eval_data,
                         instruction=f"[generated] {prompt[:80]}",
                         object_name=object_desc.strip())

        tbl, clear, strip, log, gallery, prefill, top_opt = _render(
            mode_lbl, selected_models, trusted_model, combine_all)
        return tac_pil, tbl, clear, strip, log, gallery, prefill, top_opt, ""

    def on_eval_upload(nat_pil, tac_pil, object_desc, mode_lbl, selected_models,
                       trusted_model, combine_all):
        if nat_pil is None:
            return (gr.update(),) * 8 + ("⚠ Upload a natural reference image first.",)
        if tac_pil is None:
            return (gr.update(),) * 8 + ("⚠ Upload a tactile graphic to evaluate.",)

        nat_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        nat_tmp.close()
        nat_pil.save(nat_tmp.name, format="JPEG")
        tac_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tac_tmp.close()
        tac_pil.convert("RGB").save(tac_tmp.name, format="JPEG")

        eval_data = _run_evaluators(nat_tmp.name, tac_tmp.name, selected_models)
        _store_iteration(f"Iter {len(_state)}", tac_pil.convert("RGB"), nat_pil,
                         nat_tmp.name, tac_tmp.name, eval_data,
                         object_name=object_desc.strip() if object_desc else "")

        tbl, clear, strip, log, gallery, prefill, top_opt = _render(
            mode_lbl, selected_models, trusted_model, combine_all)
        return tac_pil, tbl, clear, strip, log, gallery, prefill, top_opt, ""

    def on_edit(edit_text, target_option, api_key, include_nat, use_context,
                mode_lbl, selected_models, trusted_model, combine_all):
        if not _state:
            return (gr.update(),) * 8 + ("⚠ Generate first.",)
        if not api_key.strip():
            return (gr.update(),) * 8 + ("⚠ API key required.",)

        instruction = edit_text.strip()
        if not instruction:
            return (gr.update(),) * 8 + ("⚠ Provide an edit instruction.",)

        # Capture the prefill that was on screen when this edit was typed —
        # (prefill, instruction) is a DPO preference pair when they differ.
        prefill_shown   = _prefill_ctx["text"]
        prefill_trusted = _prefill_ctx["trusted"]

        # Inject accumulated context
        full_instruction = instruction
        if use_context and _edit_history:
            ctx = _build_context(_edit_history)
            full_instruction = ctx + " Now: " + instruction

        last = _state[-1]
        try:
            new_tac = edit_tactile(
                api_key=api_key.strip(),
                current_tactile=last["pil"],
                edit_instruction=full_instruction,
                natural_reference=last["nat_pil"] if include_nat else None,
                object_name=last.get("object", ""),
            )
        except Exception as e:
            return (gr.update(),) * 8 + (_api_error("Edit", e),)

        tac_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tac_tmp.close()
        new_tac.save(tac_tmp.name, format="JPEG")

        eval_data = _run_evaluators(last["nat_path"], tac_tmp.name, selected_models)
        _store_iteration(f"Iter {len(_state)}", new_tac, last["nat_pil"],
                         last["nat_path"], tac_tmp.name, eval_data,
                         instruction=instruction,
                         object_name=last.get("object", ""),
                         target_option=target_option or "",
                         prefill_instruction=prefill_shown,
                         prefill_trusted=prefill_trusted)

        # Append to edit log (user-visible instruction, not the context-prefixed one)
        _edit_history.append({"label": f"Iter {len(_state)-1}", "instruction": instruction})

        tbl, clear, strip, log, gallery, prefill, top_opt = _render(
            mode_lbl, selected_models, trusted_model, combine_all)
        return new_tac, tbl, clear, strip, log, gallery, prefill, top_opt, ""

    def on_view_change(mode_lbl, selected_models, trusted_model, combine_all):
        tbl, clear, strip, log, gallery, prefill, top_opt = _render(
            mode_lbl, selected_models, trusted_model, combine_all)
        return tbl, clear, strip, log, gallery, prefill, top_opt

    def on_target_pick(target_option, mode_lbl, selected_models, trusted_model):
        """User manually picked the defect to address: re-prefill the edit box
        with that defect's template (object-grounded), so a non-top-priority
        target no longer requires copy-pasting from the ensemble card."""
        if not _state or not target_option or target_option == "multiple":
            return gr.update()
        it = _state[-1]
        prefill = EDIT_INSTRUCTIONS.get(target_option, "")
        obj = (it.get("object") or "").strip()
        if prefill and obj:
            prefill = prefill.replace("the object", f"the {obj}")
        # Attribute the suggestion to the trusted judge only if it actually
        # flagged this option; otherwise it's a manual template pick.
        mr = _get_results(it, _mode(mode_lbl), selected_models)
        tm = trusted_model if (trusted_model in mr and
                               target_option in mr[trusted_model]["flagged"]) else ""
        _prefill_ctx["text"] = prefill
        _prefill_ctx["trusted"] = tm
        return gr.update(value=prefill)

    def on_reeval(mode_lbl, selected_models, trusted_model, combine_all):
        if not _state:
            return (gr.update(),) * 8 + ("⚠ Generate or evaluate an image first.",)
        last = _state[-1]
        eval_data = _run_evaluators(last["nat_path"], last["tac_path"], selected_models)
        for k, v in eval_data.items():
            last[k] = v
        tbl, clear, strip, log, gallery, prefill, top_opt = _render(
            mode_lbl, selected_models, trusted_model, combine_all)
        return last["pil"], tbl, clear, strip, log, gallery, prefill, top_opt, ""

    def on_save_to_train(label_selections: list, mode_lbl: str, selected_models: list):
        if not _state:
            return "⚠ Nothing to save yet."
        last = _state[-1]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pair_dir = TRAIN_DIR / f"pair_{ts}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        # ── Save final natural + tactile images ───────────────────────────────
        nat_out = pair_dir / "natural.jpg"
        tac_out = pair_dir / "tactile.jpg"
        last["nat_pil"].save(nat_out, format="JPEG", quality=95)
        last["pil"].save(tac_out,    format="JPEG", quality=95)

        nat_rel = str(nat_out.relative_to(TRAIN_DIR))
        tac_rel = str(tac_out.relative_to(TRAIN_DIR))

        # ── Annotation record (existing format — splits_v2 compatible) ────────
        record = {
            "pair_id":       f"{nat_rel}::{tac_rel}",
            "natural_image": nat_rel,
            "tactile_image": tac_rel,
            "object":        last.get("object", ""),
            "source":        "gpt-image-1",
            "iteration":     last["label"],
            "saved_at":      datetime.now().isoformat(timespec="seconds"),
        }
        for opt in ALL_OPTIONS:
            record[opt] = 1 if opt in label_selections else 0

        ann_file = TRAIN_DIR / "annotations.jsonl"
        with open(ann_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        # ── Trajectory record ─────────────────────────────────────────────────
        # Save each iteration's tactile image and build (state, action, reward) steps.
        steps = []
        prev_mean = None
        for i, it in enumerate(_state):
            tac_iter_out = pair_dir / f"tactile_iter{i}.jpg"
            it["pil"].save(tac_iter_out, format="JPEG", quality=95)

            scores   = _iter_scores(it, selected_models)
            mean_prob = _mean_defect_prob(scores)
            # Reward = reduction in mean defect probability vs previous step.
            # Positive = this edit helped. Negative = things got worse.
            reward = round(prev_mean - mean_prob, 4) if prev_mean is not None else None

            prefill = it.get("prefill_instruction", "")
            steps.append({
                "iter_label":       it["label"],
                "instruction":      it.get("instruction", ""),
                "target_option":    it.get("target_option", ""),
                "prefill_instruction": prefill,
                "prefill_trusted":  it.get("prefill_trusted", ""),
                "accepted_prefill": bool(prefill) and
                                    it.get("instruction", "").strip() == prefill.strip(),
                "nat_image":        nat_rel,
                "tac_image":        str(tac_iter_out.relative_to(TRAIN_DIR)),
                "scores":           scores,
                "mean_defect_prob": mean_prob,
                "reward":           reward,
            })
            prev_mean = mean_prob

        # Evaluator flags per model at the final iteration (balanced t=0.50),
        # including the weighted Ensemble when selected. Stored separately from
        # human_labels so disagreements are visible.
        evaluator_flags = {}
        for model, res in _get_results(last, "balanced", selected_models).items():
            evaluator_flags[model] = {
                opt: int(res["options"][opt]["flagged"])
                for opt in ALL_OPTIONS if opt in res["options"]
            }

        human_labels = {opt: (1 if opt in label_selections else 0) for opt in ALL_OPTIONS}

        traj = {
            "trajectory_id":   f"traj_{ts}",
            "object":          last.get("object", ""),
            "n_iters":         len(_state),
            "saved_at":        datetime.now().isoformat(timespec="seconds"),
            "source":          "gpt-image-1",
            "steps":           steps,
            "evaluator_flags": evaluator_flags,
            "human_labels":    human_labels,
        }
        with open(TRAJ_FILE, "a") as f:
            f.write(json.dumps(traj) + "\n")

        return f"✅ Saved to generated_training_data/pair_{ts}/"

    def on_export_session():
        if not _state:
            return None
        data = {
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "iterations":  [
                {k: v for k, v in it.items()
                 if k not in ("pil", "nat_pil")}
                for it in _state
            ],
            "edit_history": _edit_history,
        }
        tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w")
        json.dump(data, tmp, indent=2)
        tmp.close()
        return tmp.name

    def on_reset():
        _state.clear()
        _edit_history.clear()
        empty_tbl = _eval_table({}, DEFAULT_THRESHOLD_MODE)
        return (
            None, None, empty_tbl,
            gr.update(value="", visible=False),
            _iteration_strip([], []),
            _edit_log_html([]),
            [],
            gr.update(value=""),
            gr.update(value=None),
            "",
        )

    # ── UI ────────────────────────────────────────────────────────────────────

    with gr.Blocks(title="TactileExpert") as demo:

        gr.Markdown(
            "# TactileExpert — Tactile Graphic Pipeline\n"
            "Generate, evaluate, and iteratively refine tactile graphics. "
            "GPT-image-1 carries the image forward but **not** the text — "
            "previous corrections are summarised and injected as context automatically."
        )

        err_box = gr.HTML(value="")

        with gr.Row():

            # ── Left column ───────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):

                ui_mode = gr.Radio(
                    choices=["Full pipeline", "Eval only"],
                    value="Full pipeline",
                    label="Mode",
                )

                gr.Markdown("### 1 — Reference Image")
                nat_img = gr.Image(label="Natural reference photograph (mandatory)",
                                   type="pil", height=200)
                detect_status = gr.Markdown(
                    value="",
                    visible=False,
                )

                # ── Full pipeline ─────────────────────────────────────────────
                with gr.Group(visible=True) as gen_group:
                    gr.Markdown("### 2 — Generation")
                    object_desc = gr.Textbox(
                        label="Object name",
                        placeholder="e.g. a horse, a potted cactus",
                        lines=1,
                    )
                    prompt_box = gr.Textbox(
                        label="BANA prompt (auto-filled — editable)",
                        value=BANA_TEMPLATE.format(object_description="the object"),
                        lines=7,
                    )
                    object_desc.change(
                        fn=lambda d: (build_prompt(d) if d.strip()
                                      else BANA_TEMPLATE.format(object_description="the object")),
                        inputs=object_desc, outputs=prompt_box,
                    )

                # ── Single API key (shared for gen + edit) ────────────────────
                api_key = gr.Textbox(
                    label="OpenAI API key",
                    placeholder="sk-…",
                    type="password",
                    lines=1,
                )

                with gr.Group(visible=True) as gen_btn_group:
                    gen_btn = gr.Button("Generate tactile graphic", variant="primary")

                # ── Eval-only upload ──────────────────────────────────────────
                with gr.Group(visible=False) as upload_group:
                    gr.Markdown("### 2 — Upload tactile to evaluate")
                    tac_upload = gr.Image(
                        label="Tactile graphic", type="pil", height=200)
                    eval_btn = gr.Button("Evaluate", variant="primary")

                gr.Markdown("### 3 — Edit")

                model_select = gr.CheckboxGroup(
                    choices=AVAILABLE_MODELS,
                    value=AVAILABLE_MODELS,
                    label="Run evaluators",
                )
                trusted_radio = gr.Radio(
                    choices=AVAILABLE_MODELS,
                    value=ENSEMBLE_NAME,
                    label="Pre-fill edit instruction from",
                )
                combine_cb = gr.Checkbox(
                    label="Combine all flagged issues in pre-fill",
                    value=False,
                    info="Default off: pre-fill only the top-priority issue — one targeted edit at a time",
                )
                edit_box = gr.Textbox(
                    label="Edit instruction (auto-filled — editable)",
                    placeholder="Will populate after first evaluation…",
                    lines=3,
                )
                target_dropdown = gr.Dropdown(
                    choices=[(OPTION_DISPLAY[o], o) for o in ALL_OPTIONS]
                            + [("Multiple / Other", "multiple")],
                    value=None,
                    label="Defect being addressed by this edit",
                    info="Auto-filled from top flagged issue — change if you're targeting something different",
                )
                include_nat_cb = gr.Checkbox(
                    label="Include natural reference in edit call",
                    value=True,
                )
                use_context_cb = gr.Checkbox(
                    label="Inject edit history as context (recommended — GPT has no text memory)",
                    value=True,
                )

                with gr.Group(visible=True) as edit_btn_group:
                    edit_btn = gr.Button("Apply edit", variant="secondary")

                gr.Markdown("---")
                reset_btn   = gr.Button("Reset session", variant="stop", size="sm")
                export_btn  = gr.Button("Export session (JSON)", size="sm")
                export_file = gr.File(label="Session export", visible=False)

            # ── Right column ───────────────────────────────────────────────────
            with gr.Column(scale=2):

                gr.Markdown("### Current candidate")
                tac_out = gr.Image(label="Latest tactile graphic", height=280)

                mode_radio = gr.Radio(
                    choices=list(MODE_LABELS.keys()),
                    value=DEFAULT_MODE_LBL,
                    label="Decision thresholds",
                )

                gr.Markdown("### Evaluation — all models")
                eval_table = gr.HTML(_eval_table({}, DEFAULT_THRESHOLD_MODE))
                all_clear_box = gr.HTML(value="", visible=False)
                reeval_btn = gr.Button(
                    "Re-evaluate current image", variant="secondary", size="sm"
                )

                with gr.Tabs():

                    with gr.Tab("Iteration Gallery"):
                        gr.Markdown(
                            "*Click any image to expand. "
                            "Natural reference shown first.*"
                        )
                        iter_gallery = gr.Gallery(
                            label="Iterations",
                            columns=6,
                            height=260,
                            object_fit="contain",
                        )
                        strip_html = gr.HTML(
                            _iteration_strip([], []),
                            label="Defect badges",
                        )

                    with gr.Tab("Edit Log"):
                        gr.Markdown(
                            "Every edit instruction is logged here. "
                            "When *Inject edit history as context* is enabled, "
                            "a compact summary is prepended to each new edit call — "
                            "because GPT-image-1 receives a fresh text prompt every time."
                        )
                        edit_log_html = gr.HTML(_edit_log_html([]))

                    with gr.Tab("Save to Training"):
                        gr.Markdown(
                            "Approve the current iteration and save it as a "
                            "training pair. Adjust the defect labels as needed — "
                            "they are pre-filled from the evaluation. "
                            "Saved to `generated_training_data/annotations.jsonl`."
                        )
                        save_labels = gr.CheckboxGroup(
                            choices=[(OPTION_DISPLAY[o], o) for o in ALL_OPTIONS],
                            value=[],
                            label="Defects present (uncheck = this pair is clean for this option)",
                        )
                        save_btn  = gr.Button("Save to training data", variant="primary")
                        save_status = gr.Markdown()

        # ── Wiring ──────────────────────────────────────────────────────────────

        _gen_outputs = [tac_out, eval_table, all_clear_box,
                        strip_html, edit_log_html, iter_gallery, edit_box,
                        target_dropdown, err_box]

        gen_btn.click(
            fn=on_generate,
            inputs=[nat_img, object_desc, prompt_box, api_key,
                    mode_radio, model_select, trusted_radio, combine_cb],
            outputs=_gen_outputs,
        )

        eval_btn.click(
            fn=on_eval_upload,
            inputs=[nat_img, tac_upload, object_desc, mode_radio, model_select,
                    trusted_radio, combine_cb],
            outputs=_gen_outputs,
        )

        _edit_outputs = [tac_out, eval_table, all_clear_box,
                         strip_html, edit_log_html, iter_gallery, edit_box,
                         target_dropdown, err_box]

        edit_btn.click(
            fn=on_edit,
            inputs=[edit_box, target_dropdown, api_key, include_nat_cb, use_context_cb,
                    mode_radio, model_select, trusted_radio, combine_cb],
            outputs=_edit_outputs,
        )

        _view_outputs = [eval_table, all_clear_box,
                         strip_html, edit_log_html, iter_gallery, edit_box, target_dropdown]

        for ctrl in (mode_radio, model_select, trusted_radio, combine_cb):
            ctrl.change(
                fn=on_view_change,
                inputs=[mode_radio, model_select, trusted_radio, combine_cb],
                outputs=_view_outputs,
            )

        # .input (not .change): fires only on USER picks, so _render's
        # programmatic top-option updates don't overwrite the prefill.
        target_dropdown.input(
            fn=on_target_pick,
            inputs=[target_dropdown, mode_radio, model_select, trusted_radio],
            outputs=edit_box,
        )

        reeval_btn.click(
            fn=on_reeval,
            inputs=[mode_radio, model_select, trusted_radio, combine_cb],
            outputs=_gen_outputs,
        )

        ui_mode.change(
            fn=lambda m: (
                gr.update(visible=m == "Full pipeline"),
                gr.update(visible=m == "Full pipeline"),
                gr.update(visible=m != "Full pipeline"),
            ),
            inputs=ui_mode,
            outputs=[gen_group, gen_btn_group, upload_group],
        )

        reset_btn.click(
            fn=on_reset,
            outputs=[nat_img, tac_out, eval_table, all_clear_box,
                     strip_html, edit_log_html, iter_gallery, edit_box,
                     target_dropdown, err_box],
        )

        # ── Object auto-detection on image upload ──────────────────────────────
        def on_nat_upload(pil):
            """Classify the uploaded natural image and prefill the object name.

            CLIP is loaded eagerly at startup, so this runs on every upload.
            On any classification error the field is left blank — the user
            fills it manually.
            """
            if pil is None:
                return gr.update(value=""), gr.update(visible=False)
            try:
                hits = _clip().classify_object(pil, top_k=3)
                top_name, top_conf = hits[0]
                alts = ", ".join(f"{n} ({c:.0%})" for n, c in hits[1:])
                note = (
                    f"*Detected: **{top_name}** ({top_conf:.0%} confidence)"
                    + (f" — alternatives: {alts}" if alts else "")
                    + " — edit if wrong*"
                )
                return gr.update(value=top_name), gr.update(value=note, visible=True)
            except Exception:
                return gr.update(value=""), gr.update(visible=False)

        nat_img.change(
            fn=on_nat_upload,
            inputs=nat_img,
            outputs=[object_desc, detect_status],
        )

        export_btn.click(
            fn=on_export_session,
            outputs=export_file,
        ).then(
            fn=lambda f: gr.update(visible=f is not None),
            inputs=export_file,
            outputs=export_file,
        )

        def _prefill_save_labels(mode_lbl, selected_models):
            if not _state:
                return gr.update(value=[])
            it    = _state[-1]
            mode  = _mode(mode_lbl)
            mr    = _get_results(it, mode, selected_models)
            flagged = sorted(
                {o for res in mr.values() for o in res["flagged"]},
                key=ALL_OPTIONS.index,
            )
            return gr.update(value=flagged)

        # Pre-fill save labels when switching to Save tab or after any eval
        save_btn.click(
            fn=on_save_to_train,
            inputs=[save_labels, mode_radio, model_select],
            outputs=save_status,
        )

        # Prefill save labels whenever eval runs
        for btn in (gen_btn, eval_btn, edit_btn, reeval_btn):
            btn.click(
                fn=_prefill_save_labels,
                inputs=[mode_radio, model_select],
                outputs=save_labels,
            )

    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
