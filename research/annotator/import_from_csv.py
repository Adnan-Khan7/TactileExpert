#!/usr/bin/env python3
"""
Sync edited annotations_review.csv back into annotations.jsonl.

Usage (from the research workspace, default ~/vlm_finetune):
    python code/annotator/import_from_csv.py

Reads:   annotations_review.csv   (edited on laptop/offline)
Updates: annotations.jsonl        (in-place, preserves all other fields)

Rules:
  - Matches rows by pair number (#) — order-independent.
  - Updates tk/bl/mp/mt/ep/nb/nc from individual columns (takes precedence
    over the labels column, which is just for display).
  - Sets annotated=True and stamps annotated_at if any label was explicitly set
    (i.e., the row was edited or the pair was already annotated).
  - Rows where done=0 AND all labels=0 are treated as not yet reviewed —
    annotated stays False.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT  # noqa: E402

import csv
import json
from datetime import datetime
from pathlib import Path

ANNOTATIONS_FILE = DATA_ROOT / "annotations.jsonl"
CSV_FILE         = DATA_ROOT / "annotations_review.csv"

OPTION_KEYS = ['too_thick', 'broken_lines', 'missing_parts', 'missing_texture',
               'extra_parts', 'noisy_background', 'non_conformant']
SHORT       = ['tk',        'bl',            'mp',            'mt',
               'ep',        'nb',             'nc']


def main():
    # Load existing records
    records = []
    with open(ANNOTATIONS_FILE) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Load CSV edits indexed by pair number
    csv_rows = {}
    with open(CSV_FILE, newline='') as f:
        for row in csv.DictReader(f):
            csv_rows[int(row['#'])] = row

    changed = 0
    now = datetime.now().isoformat(timespec='seconds')

    for i, r in enumerate(records):
        pair_num = i + 1
        if pair_num not in csv_rows:
            continue
        row = csv_rows[pair_num]

        # Parse label values from individual short columns
        vals = {}
        for short, key in zip(SHORT, OPTION_KEYS):
            raw = row.get(short, '').strip()
            if raw in ('0', '1'):
                vals[key] = int(raw)

        if not vals:
            continue  # row has no parseable values — skip

        # Apply changes
        updated = False
        for key, val in vals.items():
            if r.get(key) != val:
                r[key] = val
                updated = True

        done_flag = row.get('done', '0').strip() == '1'
        any_positive = any(vals.values())

        # Mark annotated if done flag set OR if any positive label
        if (done_flag or any_positive) and not r['annotated']:
            r['annotated'] = True
            r['annotated_at'] = now
            updated = True
        elif done_flag and not r['annotated']:
            r['annotated'] = True
            r['annotated_at'] = now
            updated = True

        if updated:
            changed += 1

    # Write back
    tmp = ANNOTATIONS_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    tmp.replace(ANNOTATIONS_FILE)

    print(f"Synced {changed} changed records → {ANNOTATIONS_FILE}")
    annotated = sum(1 for r in records if r['annotated'])
    print(f"Total annotated after sync: {annotated} / {len(records)}")


if __name__ == '__main__':
    main()
