#!/usr/bin/env python3
"""
TactileEval Annotation Tool

Usage (from the research workspace, default ~/vlm_finetune):
    python code/annotator/annotator.py --share

Keyboard flow:
  - "Save & Next unannotated" to move forward through unseen pairs.
  - "← Prev" / "Next →" to step through all pairs sequentially (for review/fixes).
  - "Jump to #" to land on a specific pair by number.
  - "Save (stay)" to save without navigating.

Saves write atomically to annotations.jsonl via a .tmp swap.
On startup the app lands on the first unannotated pair automatically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT  # noqa: E402

import json
import argparse
from pathlib import Path
from datetime import datetime
import gradio as gr

_GR_MAJOR = int(gr.__version__.split('.')[0])

ANNOTATIONS_FILE = DATA_ROOT / "annotations.jsonl"
IMAGES_BASE      = DATA_ROOT / "images"

# non_conformant retired from the UI 2026-08-03: it is not in the deployed
# ALL_OPTIONS, no evaluator scores it, and it is not collected any more.
# Existing non_conformant labels in splits_v2 are left untouched — this only
# stops NEW ones being gathered. Removing it here is sufficient: cb_values(),
# commit() and the handler output lists are all derived from OPTION_KEYS/all_cbs.
# noisy_background retired 2026-08-03 alongside non_conformant: neither is in the
# deployed ALL_OPTIONS, and noisy_background was excluded from the VQA build for
# having only 22 positives. The UI now shows exactly the five options the fleet
# actually scores.
OPTION_KEYS = [
    'too_thick', 'broken_lines', 'missing_parts', 'missing_texture',
    'extra_parts',
]

OPTION_LABELS = {
    'too_thick':        'Too Thick',
    'broken_lines':     'Broken Lines',
    'missing_parts':    'Missing Parts',
    'missing_texture':  'Missing Texture',
    'extra_parts':      'Extra Parts',
    'noisy_background': 'Noisy Background',
    'non_conformant':   'Non-Conformant  ·  [Supervisor label: Overall Conformance Issue]',
}

OPTION_INFO = {
    'too_thick':
        'Lines/outlines in the tactile image are too thick or bold.',
    'broken_lines':
        'Continuous lines (contours, outlines, edges) appear broken or interrupted.',
    'missing_parts':
        'A distinct structural part clearly visible in the natural image is absent from the tactile.',
    'missing_texture':
        'Surface texture that characterises the object (fur, scales, fabric grain …) '
        'is missing or insufficient in the tactile.',
    'extra_parts':
        'Tactile contains shapes or elements not present in the natural image (hallucinated content).',
    'noisy_background':
        'Background has artifacts, unintended patterns, or noise that should not be there.',
    'non_conformant':
        'Overall the tactile does NOT match the reference image concept — '
        'check this when the tactile depicts something fundamentally different '
        'from the natural image, regardless of minor defects above. '
        '(For supervisors this is the "Overall Conformance Issue" flag.)',
}


# ─── Data layer ─────────────────────────────────────────────────────────────

def load_records() -> list[dict]:
    records = []
    with open(ANNOTATIONS_FILE) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_records(records: list[dict]):
    tmp = ANNOTATIONS_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    tmp.replace(ANNOTATIONS_FILE)


records: list[dict] = []  # populated after CLI args are parsed


def first_unannotated() -> int:
    for i, r in enumerate(records):
        if not r['annotated']:
            return i
    return 0  # all annotated — start from beginning


def commit(idx: int, checkbox_values: list[bool]) -> str:
    """
    Write checkbox values to records[idx] and flush to disk.

    Safe for concurrent users: reloads from disk before writing so two
    simultaneous saves to different pairs never clobber each other.
    """
    ts = datetime.now().isoformat(timespec='seconds')

    # Build the updated record in memory
    r = records[idx]
    for key, val in zip(OPTION_KEYS, checkbox_values):
        r[key] = 1 if val else 0
    r['annotated'] = True
    r['annotated_at'] = ts

    # Reload from disk to pick up any concurrent saves, patch this record, write back
    fresh = load_records()
    fresh[idx] = r
    save_records(fresh)

    # Keep the in-memory list in sync with what's on disk
    for i, rec in enumerate(fresh):
        records[i] = rec

    return ts


# ─── UI helpers ─────────────────────────────────────────────────────────────

def progress_line() -> str:
    n_done = sum(1 for r in records if r['annotated'])
    return f"**{n_done} / {len(records)}** annotated  ·  {len(records) - n_done} remaining"


def pair_header(idx: int) -> str:
    r = records[idx]
    status = "✅ annotated" if r['annotated'] else "⏳ not yet annotated"
    return (
        f"### Pair {idx + 1} / {len(records)}  ·  "
        f"{r['category'].replace('_', ' ')} › **{r['object']}**  ·  {status}\n\n"
        f"{progress_line()}"
    )


def resolve(rel: str) -> str | None:
    p = IMAGES_BASE / rel
    return str(p) if p.exists() else None


def cb_values(idx: int) -> list[bool]:
    r = records[idx]
    return [bool(r.get(k, 0)) for k in OPTION_KEYS]


def load_pair(idx: int) -> list:
    """Return all outputs needed to render pair idx."""
    r   = records[idx]
    nat = resolve(r['natural_image'])
    tac = resolve(r['tactile_image'])
    return [pair_header(idx), nat, tac] + cb_values(idx) + [""]


# ─── Gradio app ─────────────────────────────────────────────────────────────

CSS = """
.pair-header { font-size: 1rem; }
.defect-section { border-top: 1px solid #e5e7eb; padding-top: 0.75rem; margin-top: 0.5rem; }
.conform-section { border-top: 2px solid #d1d5db; padding-top: 0.75rem; margin-top: 0.75rem; }
"""

_blocks_kwargs = {} if _GR_MAJOR >= 6 else {'css': CSS, 'theme': gr.themes.Soft()}
with gr.Blocks(title="TactileEval Annotator", **_blocks_kwargs) as demo:

    cur_idx = gr.State(value=0)

    gr.Markdown("# TactileEval Annotation Tool")
    header_md = gr.Markdown(elem_classes=["pair-header"])

    # ── Navigation ───────────────────────────────────────────────────────────
    with gr.Row():
        prev_btn      = gr.Button("← Prev", size="sm", scale=0)
        next_btn      = gr.Button("Next →", size="sm", scale=0)
        jump_num      = gr.Number(label="Jump to pair #", value=1,
                                   precision=0, minimum=1, scale=0)
        jump_btn      = gr.Button("Go", size="sm", scale=0)
        save_next_btn = gr.Button("💾  Save & Next unannotated →",
                                   variant="primary", scale=1)

    # ── Images ───────────────────────────────────────────────────────────────
    with gr.Row():
        nat_img = gr.Image(label="Natural Image",  height=500)
        tac_img = gr.Image(label="Tactile Image",  height=500)

    # ── Defect checkboxes ────────────────────────────────────────────────────
    gr.Markdown("### Defects present in the tactile image  *(check all that apply)*")

    with gr.Column(elem_classes=["defect-section"]):
        cb_too_thick   = gr.Checkbox(label=OPTION_LABELS['too_thick'],
                                      info=OPTION_INFO['too_thick'])
        cb_broken      = gr.Checkbox(label=OPTION_LABELS['broken_lines'],
                                      info=OPTION_INFO['broken_lines'])
        cb_miss_parts  = gr.Checkbox(label=OPTION_LABELS['missing_parts'],
                                      info=OPTION_INFO['missing_parts'])
        cb_miss_tex    = gr.Checkbox(label=OPTION_LABELS['missing_texture'],
                                      info=OPTION_INFO['missing_texture'])
        cb_extra       = gr.Checkbox(label=OPTION_LABELS['extra_parts'],
                                      info=OPTION_INFO['extra_parts'])

    # ── Save controls ────────────────────────────────────────────────────────
    with gr.Row():
        save_btn = gr.Button("💾  Save (stay on this pair)", size="sm", scale=0)

    status_md = gr.Markdown()

    # Ordered list of all checkbox components — must match OPTION_KEYS order
    all_cbs = [cb_too_thick, cb_broken, cb_miss_parts, cb_miss_tex, cb_extra]

    # ── Output lists (must be consistent across all handlers) ────────────────
    #   load_pair() returns: [header, nat, tac, cb×5, status]  = 9 items
    pair_outputs  = [header_md, nat_img, tac_img] + all_cbs + [status_md]
    nav_outputs   = [cur_idx] + pair_outputs          # 10 items

    # ── Event handlers ───────────────────────────────────────────────────────

    def on_initial_load():
        idx = first_unannotated()
        return [idx] + load_pair(idx)

    def on_prev(idx):
        new = max(0, idx - 1)
        return [new] + load_pair(new)

    def on_next(idx):
        new = min(len(records) - 1, idx + 1)
        return [new] + load_pair(new)

    def on_jump(idx, num):
        new = max(0, min(len(records) - 1, int(num) - 1))
        return [new] + load_pair(new)

    def on_save(idx, *cbs):
        ts = commit(idx, list(cbs))
        return [pair_header(idx), f"✅ Saved at {ts}"]

    def on_save_next(idx, *cbs):
        commit(idx, list(cbs))
        # Advance to next unannotated pair after current position
        new = idx + 1
        while new < len(records) and records[new]['annotated']:
            new += 1
        if new >= len(records):
            # All done — stay and show a banner
            return [idx] + load_pair(idx)[:-1] + ["🎉 All pairs annotated!"]
        return [new] + load_pair(new)

    # ── Wire events ──────────────────────────────────────────────────────────

    demo.load(on_initial_load, outputs=nav_outputs)

    prev_btn.click(on_prev,   inputs=[cur_idx],              outputs=nav_outputs)
    next_btn.click(on_next,   inputs=[cur_idx],              outputs=nav_outputs)
    jump_btn.click(on_jump,   inputs=[cur_idx, jump_num],    outputs=nav_outputs)

    save_btn.click(on_save,        inputs=[cur_idx] + all_cbs,
                   outputs=[header_md, status_md])
    save_next_btn.click(on_save_next, inputs=[cur_idx] + all_cbs,
                        outputs=nav_outputs)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true',
                        help='Create a public gradio.live share link')
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--annotations-file', type=str, default=None,
                        help='Path to annotations.jsonl (default: <TACTILE_DATA_ROOT>/annotations.jsonl)')
    parser.add_argument('--images-dir', type=str, default=None,
                        help='Path to images root directory (default: <TACTILE_DATA_ROOT>/images/)')
    args = parser.parse_args()

    if args.annotations_file:
        ANNOTATIONS_FILE = Path(args.annotations_file)
    if args.images_dir:
        IMAGES_BASE = Path(args.images_dir)

    records.extend(load_records())

    _launch_kwargs = {'theme': gr.themes.Soft()} if _GR_MAJOR >= 6 else {}
    demo.launch(
        share=args.share,
        server_port=args.port,
        allowed_paths=[str(IMAGES_BASE)],
        **_launch_kwargs,
    )
