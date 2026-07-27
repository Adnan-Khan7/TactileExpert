#!/usr/bin/env python3
"""
TactileEval — Gradio Demo (multi-model comparison)
Evaluates a (natural reference photograph, tactile graphic) pair against
one or more models and shows results side-by-side.

Usage:
    python app.py                        # fine-tuned model checked by default
    python app.py --share                # public URL via salloc on compute node
    python app.py --checkpoint /path     # custom LoRA adapter path
    python app.py --model-path /path     # custom base model path
    python app.py --port 7861
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_ROOT
DATA_DIR     = PROJECT_ROOT / "data" / "vqa"
IMAGES_DIR   = PROJECT_ROOT / "images"
DEFAULT_CKPT = PROJECT_ROOT / "output" / "best_checkpoint"
MODELS_DIR        = PROJECT_ROOT / "TactileEval_Context_And_Data" / "models"
PRECOMPUTED_FILE  = PROJECT_ROOT / "output" / "precomputed_results.json"

sys.path.insert(0, str(HERE.parent))  # code/ — for models.*
sys.path.insert(0, str(HERE))          # ui/  — for vlm_evaluator, baseline_evaluator shims
from vlm_evaluator import QUESTIONS, DIMENSION_LABELS, EDIT_INSTRUCTIONS

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

ALL_MODEL_NAMES = ["VLM Fine-tuned", "CLIP Probe"]

MODEL_SUBTITLE = {
    "VLM Fine-tuned": "Qwen2-VL-2B + LoRA · HC test macro-F1 = 0.591 (calibrated, cleaned labels)",
    "CLIP Probe":     "CLIP ViT-L/14 + task-head MLPs · HC test macro-F1 = 0.715 (cleaned labels)",
}

PANEL_COLOR = {
    "VLM Fine-tuned": "#3498db",
    "CLIP Probe":     "#e67e22",
}

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

OPTION_DISPLAY = {
    "broken_lines":    "Broken Lines",
    "too_thick":       "Too Thick",
    "missing_texture": "Missing Texture",
    "missing_parts":   "Missing Parts",
}


def _bar(prob: float, flagged: bool) -> str:
    pct   = int(prob * 100)
    color = "#e74c3c" if flagged else "#27ae60"
    return (
        f"<div style='background:#eee; border-radius:4px; height:10px; margin:4px 0;'>"
        f"<div style='width:{pct}%; background:{color}; height:10px; border-radius:4px;'></div>"
        f"</div>"
    )


def build_results_html(result: dict | None) -> str:
    if result is None:
        return (
            "<div style='color:#aaa; font-family:sans-serif; padding:24px; "
            "text-align:center;'>Upload images and click <b>Evaluate</b> to see results.</div>"
        )

    html = ["<div style='font-family:sans-serif; font-size:14px; line-height:1.5;'>"]

    for opt, data in result["options"].items():
        prob    = data["prob_yes"]
        flagged = data["flagged"]
        thresh  = data["threshold"]
        label   = OPTION_DISPLAY.get(opt, opt)
        dim     = data["dimension"]
        color   = "#e74c3c" if flagged else "#27ae60"
        badge   = "⚠ FLAGGED" if flagged else "✓ CLEAR"
        bg      = "#fff5f5" if flagged else "#f5fff8"
        border  = "#e74c3c" if flagged else "#27ae60"

        edit_html = ""
        if flagged and data.get("edit_instruction"):
            edit_html = (
                f"<div style='margin-top:6px; font-size:12px; color:#c0392b;'>"
                f"<b>Suggested edit:</b> {data['edit_instruction']}</div>"
            )

        html.append(f"""
        <div style='margin-bottom:10px; padding:10px 14px; border:1px solid {border};
                    border-radius:8px; background:{bg};'>
          <div style='display:flex; justify-content:space-between; align-items:center;'>
            <span style='font-weight:600;'>{label}</span>
            <span style='color:{color}; font-weight:700; font-size:13px;'>{badge}</span>
          </div>
          <div style='font-size:11px; color:#888; margin:2px 0 4px;'>
            {dim} &nbsp;|&nbsp; threshold = {thresh}
          </div>
          {_bar(prob, flagged)}
          <div style='font-size:12px; color:#555;'>prob(yes) = {prob:.3f}</div>
          {edit_html}
        </div>
        """)

    if result["all_clear"]:
        html.append("""
        <div style='padding:14px; background:#eafaf1; border:2px solid #27ae60;
                    border-radius:8px; text-align:center; font-size:16px;
                    font-weight:700; color:#27ae60; margin-top:8px;'>
          ✓ ALL CLEAR — No quality issues detected
        </div>
        """)
    else:
        top = result["top_issue"]
        if top:
            html.append(f"""
            <div style='padding:14px; background:#fdecea; border:2px solid #e74c3c;
                        border-radius:8px; margin-top:8px;'>
              <div style='font-size:13px; font-weight:700; color:#c0392b; margin-bottom:4px;'>
                TOP PRIORITY → {OPTION_DISPLAY.get(top["option"], top["option"])}
              </div>
              <div style='font-size:12px; color:#555;'>{top.get("edit_instruction", "")}</div>
            </div>
            """)

    html.append("</div>")
    return "".join(html)


def build_comparison_html(results_by_model: dict, timings: dict | None = None) -> str:
    """Render one or more model result panels side by side."""
    if not results_by_model:
        return (
            "<div style='color:#aaa; font-family:sans-serif; padding:24px; text-align:center;'>"
            "Select at least one model and click <b>Evaluate</b> to see results.</div>"
        )

    timings = timings or {}
    panels = []
    for name, result in results_by_model.items():
        color    = PANEL_COLOR.get(name, "#555")
        subtitle = MODEL_SUBTITLE.get(name, "")
        t        = timings.get(name)
        time_str = f" &nbsp;·&nbsp; ⏱ {t:.1f}s" if t is not None else ""
        header   = (
            f"<div style='margin-bottom:8px; padding:8px 10px; background:#f8f9fa; "
            f"border-radius:6px; border-left:4px solid {color};'>"
            f"<div style='font-weight:700; font-size:14px;'>{name}{time_str}</div>"
            f"<div style='font-size:11px; color:#777; margin-top:2px;'>{subtitle}</div>"
            f"</div>"
        )
        panels.append(header + build_results_html(result))

    if len(panels) == 1:
        return panels[0]

    panel_divs = "\n".join(
        f"<div style='flex:1; min-width:260px;'>{p}</div>" for p in panels)
    return (
        f"<div style='display:flex; gap:14px; align-items:flex-start; flex-wrap:wrap;'>"
        f"{panel_divs}</div>"
    )


# ---------------------------------------------------------------------------
# Threshold modes
# ---------------------------------------------------------------------------

BALANCED_THRESHOLDS = {opt: 0.5 for opt in ["too_thick", "broken_lines", "missing_parts", "missing_texture"]}

THRESHOLD_MODES  = ["Calibrated (paper thresholds)", "Balanced (0.5)"]
INFERENCE_MODES  = ["Online (live inference)", "Offline (pre-computed)"]


def apply_thresholds(result: dict, thresholds: dict) -> dict:
    """Re-compute flagged / top_issue / all_clear using given thresholds."""
    per_option = {}
    for opt, data in result["options"].items():
        thresh  = thresholds.get(opt, 0.5)
        prob    = data["prob_yes"]
        flagged = prob > thresh
        per_option[opt] = {
            **data,
            "threshold":        thresh,
            "flagged":          flagged,
            "edit_instruction": EDIT_INSTRUCTIONS[opt] if flagged else None,
        }
    flagged_opts = sorted(
        [o for o, d in per_option.items() if d["flagged"]],
        key=lambda o: -per_option[o]["prob_yes"],
    )
    top = None
    if flagged_opts:
        t   = flagged_opts[0]
        top = {"option": t, **per_option[t]}
    return {"options": per_option, "flagged": flagged_opts, "top_issue": top,
            "all_clear": not flagged_opts}


# ---------------------------------------------------------------------------
# Dataset sample loader
# ---------------------------------------------------------------------------

def load_sample_pairs(split: str = "test") -> list[tuple[str, str, str]]:
    jsonl = DATA_DIR / f"{split}.jsonl"
    if not jsonl.exists():
        return []
    seen, pairs, label_counts = set(), [], {}
    for line in open(jsonl):
        rec     = json.loads(line)
        nat_rel = rec.get("natural_image", "")
        tac_rel = rec.get("tactile_image", "")
        key     = (nat_rel, tac_rel)
        if key in seen:
            continue
        seen.add(key)
        nat_abs = str(IMAGES_DIR / nat_rel)
        tac_abs = str(IMAGES_DIR / tac_rel)
        if not (Path(nat_abs).exists() and Path(tac_abs).exists()):
            continue
        parts = Path(nat_rel).parts
        if len(parts) >= 3:
            base  = f"{parts[0].replace('_',' ')} / {parts[1]} / {Path(parts[-1]).stem.split('_')[0]}"
        else:
            base  = nat_rel
        label_counts[base] = label_counts.get(base, 0) + 1
        label = base if label_counts[base] == 1 else f"{base} ({label_counts[base]})"
        pairs.append((label, nat_abs, tac_abs))
    return sorted(pairs, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default=str(DEFAULT_CKPT),
                        help="Path to LoRA adapter directory for 'VLM Fine-tuned'")
    parser.add_argument("--model-path",  default="Qwen/Qwen2-VL-2B-Instruct",
                        help="HuggingFace model ID or local base model path")
    parser.add_argument("--port",        type=int, default=7860)
    parser.add_argument("--share",       action="store_true")
    args = parser.parse_args()

    # ---- Pre-computed results (offline mode) ----
    precomputed: dict = {}
    if PRECOMPUTED_FILE.exists():
        with open(PRECOMPUTED_FILE) as f:
            precomputed = json.load(f)
        print(f"[app] Loaded precomputed results for {len(precomputed)} pairs.")
    else:
        print(f"[app] No precomputed results found at {PRECOMPUTED_FILE}. "
              "Offline mode unavailable until precompute.py is run.")

    # ---- Lazy model cache ----
    from vlm_evaluator      import VLMEvaluator
    from baseline_evaluator import CLIPProbeEvaluator

    _cache: dict = {}

    _loaders = {
        "VLM Fine-tuned": lambda: VLMEvaluator(
            model_path=args.model_path,
            lora_checkpoint=args.checkpoint,
        ),
        "CLIP Probe": lambda: CLIPProbeEvaluator(
            models_dir=MODELS_DIR / "clip_probe",
        ),
    }

    def get_model(name: str):
        if name not in _cache:
            print(f"[app] Loading {name}…")
            _cache[name] = _loaders[name]()
        return _cache[name]

    # ---- Pre-load sample pairs ----
    samples       = load_sample_pairs("test")
    sample_map    = {label: (nat, tac) for label, nat, tac in samples}
    sample_labels = [label for label, _, _ in samples]

    # ---- Callbacks ----
    def on_sample_select(choice):
        if choice and choice in sample_map:
            nat, tac = sample_map[choice]
            return Image.open(nat), Image.open(tac)
        return None, None

    def _apply_thresh_mode(result: dict, use_balanced: bool) -> dict:
        if use_balanced:
            return apply_thresholds(result, BALANCED_THRESHOLDS)
        # Re-apply stored calibrated thresholds to recompute flagged/top_issue cleanly
        stored = {opt: d["threshold"] for opt, d in result["options"].items()}
        return apply_thresholds(result, stored)

    def on_evaluate(nat_pil, tac_pil, selected_models, thresh_mode,
                    inference_mode, sample_choice, progress=gr.Progress()):
        if not selected_models:
            return (
                "<div style='color:#e74c3c; padding:16px; font-family:sans-serif;'>"
                "⚠ Please select at least one model.</div>"
            )

        use_balanced = (thresh_mode == THRESHOLD_MODES[1])

        # ---- OFFLINE branch ----
        if inference_mode == INFERENCE_MODES[1]:
            if not precomputed:
                return (
                    "<div style='color:#e74c3c; padding:16px; font-family:sans-serif;'>"
                    "⚠ No precomputed results found. Run <code>sbatch code/ui/precompute.sh</code> first.</div>"
                )
            if not sample_choice or sample_choice not in precomputed:
                return (
                    "<div style='color:#e74c3c; padding:16px; font-family:sans-serif;'>"
                    "⚠ Offline mode requires a test-set pair selected from the dropdown.</div>"
                )
            entry   = precomputed[sample_choice]
            results = {}
            timings = {}
            for name in selected_models:
                if name not in entry.get("models", {}):
                    continue
                results[name] = _apply_thresh_mode(entry["models"][name], use_balanced)
                timings[name] = entry.get("timings", {}).get(name)
            if not results:
                return (
                    "<div style='color:#e74c3c; padding:16px; font-family:sans-serif;'>"
                    "⚠ Selected models not yet precomputed for this pair.</div>"
                )
            return build_comparison_html(results, timings)

        # ---- ONLINE branch ----
        if nat_pil is None or tac_pil is None:
            return build_comparison_html({})

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            nat_path = f.name
            nat_pil.save(f, format="JPEG")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tac_path = f.name
            tac_pil.save(f, format="JPEG")

        n_models = len(selected_models)
        results  = {}
        timings  = {}
        try:
            for m_idx, name in enumerate(selected_models):
                progress(m_idx / n_models, desc=f"Loading {name}…")
                ev   = get_model(name)
                base = m_idx / n_models
                span = 1.0 / n_models

                def make_pfn(base=base, span=span, name=name):
                    def pfn(opt, idx, total):
                        progress(
                            base + span * (idx + 1) / total,
                            desc=f"[{name}] {OPTION_DISPLAY.get(opt, opt)}…",
                        )
                    return pfn

                t0     = time.time()
                result = ev.evaluate(nat_path, tac_path, progress_fn=make_pfn())
                timings[name] = time.time() - t0
                results[name] = _apply_thresh_mode(result, use_balanced)
                print(f"[timing] {name}: {timings[name]:.1f}s  thresh={thresh_mode}")
        finally:
            os.unlink(nat_path)
            os.unlink(tac_path)

        return build_comparison_html(results, timings)

    # ---- Build UI ----
    with gr.Blocks(
        title="TactileEval Demo",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1400px !important; }",
    ) as demo:

        gr.Markdown("""
        # TactileEval — Tactile Graphic Quality Evaluator
        Upload a natural reference photograph and a generated tactile graphic (or pick a
        test-set pair below), select one or more models, then click **Evaluate**.
        The models check four BANA accessibility defects: broken lines, too-thick strokes,
        missing texture, and missing parts.
        """)

        with gr.Row():

            # ---- Left: inputs ----
            with gr.Column(scale=1):
                nat_img = gr.Image(
                    label="Natural Reference Photograph",
                    type="pil",
                    height=240,
                )
                tac_img = gr.Image(
                    label="Tactile Graphic",
                    type="pil",
                    height=240,
                )

                offline_note = (f" · {len(precomputed)} pairs precomputed"
                               if precomputed else " · not available (run precompute.sh first)")
                inference_radio = gr.Radio(
                    choices=INFERENCE_MODES,
                    value=INFERENCE_MODES[0],
                    label="Inference mode",
                    info=f"Online = live inference (slow, any image). "
                         f"Offline = instant from precomputed JSON{offline_note}.",
                )

                sample_dd = gr.Dropdown(
                    choices=["— select a test-set pair —"] + sample_labels,
                    label=f"Test-set samples ({len(sample_labels)} unique pairs)",
                    value="— select a test-set pair —",
                )
                sample_dd.change(
                    fn=on_sample_select,
                    inputs=sample_dd,
                    outputs=[nat_img, tac_img],
                )

                model_selector = gr.CheckboxGroup(
                    choices=ALL_MODEL_NAMES,
                    value=["VLM Fine-tuned", "CLIP Probe"],
                    label="Models to evaluate",
                )

                thresh_radio = gr.Radio(
                    choices=THRESHOLD_MODES,
                    value=THRESHOLD_MODES[0],
                    label="Threshold mode",
                    info="Calibrated = recall-optimised paper thresholds. Balanced = 0.5 for all.",
                )

                eval_btn = gr.Button("Evaluate", variant="primary", size="lg")

            # ---- Right: results ----
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                result_html = gr.HTML(build_comparison_html({}))

        eval_btn.click(
            fn=on_evaluate,
            inputs=[nat_img, tac_img, model_selector, thresh_radio,
                    inference_radio, sample_dd],
            outputs=result_html,
        )

    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
