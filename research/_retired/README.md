# Retired

Scripts from the pre-v1 pipeline, kept for provenance only.

They read data that no longer exists — `data/splits_v2/`, `gpt_test.jsonl`,
`output_v2/` — or implement the two-frozen-sets evaluation design that the
unified v1 dataset replaced. Nothing in the current pipeline imports them.

They are NOT maintained and will not run. If you need to know how a pre-v1
number was produced, read them; do not execute them.

Current pipeline entry points:
  research/build_unified_splits.py      built v1 from the earlier splits
  research/restratify_v1.py             val/test rebalance
  research/annotator/                   verification queue + merge-back
  research/baselines/                   the four vision judges
  research/vlm/                         VQA build + VLM fine-tune
  research/experiments/                 scoring, gating, calibration
