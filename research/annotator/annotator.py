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
from _paths import DATA_ROOT, REPO_ROOT  # noqa: E402

# The deployed package lives at the repo root, one level above research/.
# Without this the VLM threshold import below fails and, because it used to sit
# inside vlm_panel(), a single ImportError blanked every component on the page.
sys.path.insert(0, str(REPO_ROOT))
try:
    from evaluators.constants import VLM_THRESHOLDS_CALIBRATED as VLM_THR  # noqa: E402
except Exception:  # thresholds are a nicety; never let them break annotation
    VLM_THR = {}

import json
import argparse
from pathlib import Path
from datetime import datetime
import gradio as gr

_GR_MAJOR = int(gr.__version__.split('.')[0])

ANNOTATIONS_FILE = DATA_ROOT / "annotations.jsonl"
IMAGES_BASE      = DATA_ROOT / "images"
# Tactile images normally sit in the same corpus as the natural ones. A
# generated batch does not -- verify_generated_batch/images/ holds corrupted
# renderings whose natural partners are still in the corpus -- so the two roots
# are resolved separately. Defaults to IMAGES_BASE, i.e. unchanged behaviour.
TACTILE_BASE     = None

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
        if not r['annotated'] and not r.get('excluded'):
            return i
    return 0  # all annotated — start from beginning


def set_excluded(idx: int, flag: bool) -> str:
    """Flag a pair as excluded from the dataset. Non-destructive by design.

    The image stays on disk and the record stays in the file -- only the flag
    moves, so an exclusion is always reversible and the reason it was excluded
    remains inspectable. Downstream ingest filters on `excluded`.
    """
    ts = datetime.now().isoformat(timespec='seconds')
    r = records[idx]
    if flag:
        # Excluded pairs must not reappear in the unannotated queue, so they are
        # marked annotated -- but the PRIOR state is stashed first, or
        # un-excluding would silently leave an unlabelled pair looking done.
        r.setdefault('annotated_before_exclude', bool(r.get('annotated', False)))
        r['annotated'] = True
    else:
        r['annotated'] = bool(r.pop('annotated_before_exclude', False))
    r['excluded'] = bool(flag)
    r['excluded_at'] = ts if flag else None
    fresh = load_records()
    fresh[idx] = r
    save_records(fresh)
    for i, rec in enumerate(fresh):
        records[i] = rec
    return ts


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
    n_done = sum(1 for r in records if r['annotated'] and not r.get('excluded'))
    n_exc  = sum(1 for r in records if r.get('excluded'))
    left   = len(records) - n_done - n_exc
    tail   = f"  ·  🚫 {n_exc} excluded" if n_exc else ""
    return f"**{n_done} / {len(records)}** annotated  ·  {left} remaining{tail}"


def pair_header(idx: int) -> str:
    r = records[idx]
    if r.get('excluded'):
        status = "🚫 **EXCLUDED from the dataset** — use ↩ Un-exclude to restore"
    else:
        status = "✅ annotated" if r['annotated'] else "⏳ not yet annotated"
    return (
        f"### Pair {idx + 1} / {len(records)}  ·  "
        f"{r['category'].replace('_', ' ')} › **{r['object']}**  ·  {status}\n\n"
        f"{progress_line()}"
    )


def resolve(rel: str, base: Path | None = None) -> str | None:
    if not rel:
        return None
    p = Path(rel)
    if not p.is_absolute():
        p = (base or IMAGES_BASE) / rel
    return str(p) if p.exists() else None


def vlm_panel(idx: int) -> str:
    """Show what the VLM said, and be explicit that it is a suggestion.

    Pre-ticks come from the fine-tuned VLM at its val-calibrated thresholds. On
    a deliberate-corruption batch `too_thick` is ticked almost everywhere by
    construction, so the informative part is the other four -- what else the
    generator changed while it was thickening the strokes.
    """
    try:
        r = records[idx]
        probs = r.get("vlm_scores") or {}
        if not probs:
            return ""
        cells = []
        for k in OPTION_KEYS:
            pv = probs.get(k)
            if pv is None:
                continue
            t = VLM_THR.get(k)
            if t is None:
                cells.append(f"**{OPTION_LABELS[k]}** {pv:.3f}")
            else:
                cells.append(f"{'✓' if pv >= t else '·'} **{OPTION_LABELS[k]}** {pv:.3f} _(t={t})_")
        return ("*VLM Fine-tuned (7B) — suggestion only, correct it against what you see:*  \n"
                + "  ·  ".join(cells))
    except Exception as exc:
        # A broken suggestion panel must never take the images and checkboxes
        # down with it -- that is exactly what happened the first time.
        return f"_(VLM panel unavailable: {type(exc).__name__})_"


def cb_values(idx: int) -> list[bool]:
    r = records[idx]
    return [bool(r.get(k, 0)) for k in OPTION_KEYS]


def load_pair(idx: int) -> list:
    """Return all outputs needed to render pair idx."""
    r   = records[idx]
    nat = resolve(r['natural_image'])
    tac = resolve(r['tactile_image'], TACTILE_BASE)
    return [pair_header(idx), nat, tac, vlm_panel(idx)] + cb_values(idx) + [""]


# ─── Gradio app ─────────────────────────────────────────────────────────────

CSS = """
.pair-header { font-size: 1rem; }
.defect-section { border-top: 1px solid #e5e7eb; padding-top: 0.75rem; margin-top: 0.5rem; }
.conform-section { border-top: 2px solid #d1d5db; padding-top: 0.75rem; margin-top: 0.75rem; }
.vlm-panel { font-size: 0.9rem; opacity: 0.9; border-left: 3px solid #93c5fd; padding-left: 0.6rem; }
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

    # ── VLM suggestion ───────────────────────────────────────────────────────
    vlm_md = gr.Markdown(elem_classes=["vlm-panel"])

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
        save_btn    = gr.Button("💾  Save (stay on this pair)", size="sm", scale=0)
        exclude_btn = gr.Button("🚫  Exclude & Next", variant="stop", size="sm", scale=0)
        include_btn = gr.Button("↩  Un-exclude (stay)", size="sm", scale=0)

    status_md = gr.Markdown()

    # Ordered list of all checkbox components — must match OPTION_KEYS order
    all_cbs = [cb_too_thick, cb_broken, cb_miss_parts, cb_miss_tex, cb_extra]

    # ── Output lists (must be consistent across all handlers) ────────────────
    #   load_pair() returns: [header, nat, tac, vlm, cb×5, status]  = 10 items
    pair_outputs  = [header_md, nat_img, tac_img, vlm_md] + all_cbs + [status_md]
    nav_outputs   = [cur_idx] + pair_outputs          # 11 items

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

    def _next_open(start):
        """First pair at or after `start` that is neither annotated nor excluded."""
        n = start
        while n < len(records) and (records[n]['annotated'] or records[n].get('excluded')):
            n += 1
        return n

    def on_exclude(idx, *cbs):
        """Flag and move on. The checkboxes are NOT saved — an excluded pair is
        being dropped, so whatever was ticked on it is not a label."""
        set_excluded(idx, True)
        new = _next_open(idx + 1)
        if new >= len(records):
            return [idx] + load_pair(idx)[:-1] + ["🚫 Excluded. No pairs left to review."]
        return [new] + load_pair(new)

    def on_include(idx):
        ts = set_excluded(idx, False)
        return [pair_header(idx), f"↩ Restored at {ts} — still needs annotating"]

    def on_save_next(idx, *cbs):
        commit(idx, list(cbs))
        # Advance to the next pair that is neither annotated nor excluded
        new = _next_open(idx + 1)
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
    exclude_btn.click(on_exclude,  inputs=[cur_idx] + all_cbs, outputs=nav_outputs)
    include_btn.click(on_include,  inputs=[cur_idx],
                      outputs=[header_md, status_md])


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
    parser.add_argument('--tactile-dir', type=str, default=None,
                        help='Separate root for TACTILE images. Use when reviewing a generated '
                             'batch whose renderings live outside the corpus while their natural '
                             'partners remain in it.')
    args = parser.parse_args()

    if args.annotations_file:
        ANNOTATIONS_FILE = Path(args.annotations_file)
    if args.images_dir:
        IMAGES_BASE = Path(args.images_dir)
    if args.tactile_dir:
        TACTILE_BASE = Path(args.tactile_dir)

    records.extend(load_records())

    _launch_kwargs = {'theme': gr.themes.Soft()} if _GR_MAJOR >= 6 else {}
    demo.launch(
        share=args.share,
        server_port=args.port,
        # Gradio refuses to serve any file outside the working directory unless
        # it is allow-listed, and one disallowed path fails the WHOLE output
        # batch -- both images and every checkbox render as "Error". A generated
        # batch keeps its tactile renderings outside the corpus, so that root
        # has to be listed too.
        allowed_paths=[p for p in {str(IMAGES_BASE), str(TACTILE_BASE or IMAGES_BASE)}],
        **_launch_kwargs,
    )
