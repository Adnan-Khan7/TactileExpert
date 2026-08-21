"""Pre-score the corruption batch with the VLM judge, and build annotator records.

Only the fine-tuned VLM runs — it is the best judge in the fleet (macro AUC
0.826) and the one whose flags are worth pre-filling. Running all five would
cost four times the GPU for tick marks a human is about to overwrite anyway.

Why this is a separate batch job rather than live inside the annotator: the
judge is a 7B and needs a GPU, while the annotator is a Gradio form that does
not. Scoring once into a cache means the review session holds no GPU and every
image loads instantly.

Output, into the batch directory:
    vlm_scores.json     pair -> {option: prob_yes}, raw, no threshold applied
    annotations.jsonl   annotator-ready records, checkboxes pre-ticked at the
                        val-calibrated VLM thresholds

The pre-ticks are a starting point, not a label. `too_thick` should be ticked on
almost everything here by construction — the batch is a deliberate thickness
corruption — so the informative signal is the OTHER four options: what else did
gpt-image-1 change while it was thickening the strokes.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT, require_data_root  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from evaluators.constants import ALL_OPTIONS, VLM_THRESHOLDS_CALIBRATED  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default=str(REPO_ROOT / "verify_generated_batch"))
    ap.add_argument("--base-model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--lora", default=None, help="default: <repo>/models/vlm_checkpoint")
    ap.add_argument("--options", default=",".join(ALL_OPTIONS),
                    help="comma-separated options to score. Scoring one option costs a "
                         "fifth of the GPU, and is the right choice when the other four "
                         "are about to be overwritten by the source pair's human labels.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild-records-only", action="store_true",
                    help="regenerate annotations.jsonl from an existing vlm_scores.json, no GPU")
    args = ap.parse_args()

    require_data_root()
    batch = Path(args.batch)
    manifest = batch / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest} — run generate_corruptions.py first")

    want = [o.strip() for o in args.options.split(",") if o.strip()]
    bad = [o for o in want if o not in ALL_OPTIONS]
    if bad:
        raise SystemExit(f"unknown option(s): {bad}; valid: {ALL_OPTIONS}")

    rows = [json.loads(l) for l in manifest.open() if l.strip()]
    rows = [r for r in rows if (batch / "images" / r["output"]).exists()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} generated images in {batch}")

    scores_path = batch / "vlm_scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}

    if not args.rebuild_records_only:
        todo = [r for r in rows
                if r.get("natural_image")
                and any(o not in scores.get(r["output"], {}) for o in want)]
        skipped = [r for r in rows if not r.get("natural_image")]
        print(f"to score: {len(todo)}   cached: {len(rows) - len(todo) - len(skipped)}   "
              f"no natural partner, cannot score: {len(skipped)}")
        if todo:
            from evaluators.vlm_eval import VLMEvaluator
            judge = VLMEvaluator(model_path=args.base_model, lora_checkpoint=args.lora)
            for i, r in enumerate(todo, 1):
                nat = DATA_ROOT / "images" / r["natural_image"]
                tac = batch / "images" / r["output"]
                try:
                    got = scores.get(r["output"], {})
                    for opt in want:
                        if opt not in got:
                            got[opt] = round(judge._score_option(str(nat), str(tac), opt), 4)
                    scores[r["output"]] = got
                except Exception as exc:
                    print(f"  [{i}/{len(todo)}] FAILED {r['output']}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                if i % 10 == 0 or i == len(todo):
                    scores_path.write_text(json.dumps(scores, indent=2))
                    print(f"  [{i}/{len(todo)}] scored", flush=True)
            scores_path.write_text(json.dumps(scores, indent=2))
        print(f"wrote {scores_path}")

    # ── annotator records ───────────────────────────────────────────────────
    # Preserve any human decision already made. This script is re-run whenever
    # the batch grows, and rebuilding from scratch would silently destroy an
    # in-progress review -- annotations, exclusions and all. Keyed on the output
    # filename, which is stable.
    out = Path(args.batch) / "annotations.jsonl"
    existing = {}
    if out.exists():
        for line in out.open():
            if line.strip():
                e = json.loads(line)
                if e.get("annotated") or e.get("excluded"):
                    existing[e["tactile_image"]] = e
        if existing:
            print(f"preserving {len(existing)} record(s) that already carry a human decision")

    thr = VLM_THRESHOLDS_CALIBRATED
    records, n_ticked, auto_excluded, kept = [], {o: 0 for o in ALL_OPTIONS}, [], 0
    for r in rows:
        prior = existing.get(r["output"])
        if prior is not None:
            # a human already ruled on this one -- carry it through untouched,
            # only refreshing the VLM scores for display
            if r["output"] in scores:
                prior["vlm_scores"] = scores[r["output"]]
            records.append(prior)
            kept += 1
            continue
        p = scores.get(r["output"], {})
        # A corrupted rendering whose source has no natural partner cannot become
        # a training row: the judges score PAIRS, and the trainers open the
        # natural image with a bare Image.open and no existence check. 10 of the
        # corpus's 997 tactile files are orphans like this -- never annotated,
        # never in any split -- and the every-Nth selection walks the filesystem
        # rather than the splits, so it can pick one up. Auto-exclude rather than
        # showing a reviewer a blank left panel.
        nat_rel = r.get("natural_image")
        unusable = None
        if not nat_rel:
            unusable = "no natural partner recorded for the source rendering"
        elif not (DATA_ROOT / "images" / nat_rel).exists():
            unusable = f"natural image missing on disk: {nat_rel}"
        rec = {
            "pair_id": f"corrupt_v1::{r['output']}",
            "natural_image": r.get("natural_image") or "",
            "tactile_image": r["output"],
            "source_tactile": r["tactile_image"],
            "category": r["family"],
            "object": r["object"],
            "corruption": r.get("corruption", "too_thick"),
            "synthetic_corruption": True,
            "annotated": bool(unusable),
            "annotated_at": None,
            "excluded": bool(unusable),
            "exclude_reason": unusable,
            "vlm_scores": p,
            "vlm_prefilled": bool(p),
        }
        for o in ALL_OPTIONS:
            # an option with no VLM score stays 0 here; inherit_source_labels.py
            # then fills it from the source pair's human label
            tick = int(p[o] >= thr[o]) if o in p else 0
            rec[o] = tick
            n_ticked[o] += tick
        if unusable:
            auto_excluded.append((r["output"], unusable))
        records.append(rec)

    out = batch / "annotations.jsonl"
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {out}  ({len(records)} records; {kept} carried over unchanged)")
    if auto_excluded:
        print(f"auto-excluded {len(auto_excluded)} unusable — reviewable via ↩ Un-exclude:")
        for name, why in auto_excluded:
            print(f"   {name}: {why}")
    fresh = len(records) - kept
    print(f"VLM pre-ticks at val-calibrated thresholds (on the {fresh} not yet reviewed):")
    for o in ALL_OPTIONS:
        print(f"   {o:16} {n_ticked[o]:4} / {fresh}  (threshold {thr[o]})")
    print("\nThese are a starting point. Correct them against ANNOTATION_CRITERIA "
          "in evaluators/constants.py — the label is what the image shows, not "
          "what the corruption intended.")


if __name__ == "__main__":
    main()
