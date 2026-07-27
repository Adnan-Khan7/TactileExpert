#!/usr/bin/env python3
"""
Export annotations.jsonl → compact CSV for offline split-screen review.

Usage (from the research workspace, default ~/vlm_finetune):
    python code/annotator/export_for_review.py

Outputs:
    annotations_review.csv   — one row per pair, short column names
    LABEL_CHEAT_SHEET.txt    — keep open alongside the CSV

Label column order (left to right in CSV):
    tk  bl  mp  mt  ep  nb  nc
    |   |   |   |   |   |   └─ non_conformant
    |   |   |   |   |   └───── noisy_background
    |   |   |   |   └───────── extra_parts
    |   |   |   └───────────── missing_texture
    |   |   └───────────────── missing_parts
    |   └───────────────────── broken_lines
    └───────────────────────── too_thick
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT  # noqa: E402

import json
import csv
from pathlib import Path

ANNOTATIONS_FILE = DATA_ROOT / "annotations.jsonl"
OUT_CSV          = DATA_ROOT / "annotations_review.csv"
OUT_CHEAT        = DATA_ROOT / "LABEL_CHEAT_SHEET.txt"

OPTION_KEYS = ['too_thick', 'broken_lines', 'missing_parts', 'missing_texture',
               'extra_parts', 'noisy_background', 'non_conformant']
SHORT       = ['tk',        'bl',            'mp',            'mt',
               'ep',        'nb',             'nc']

CAT_ABBREV = {
    'Animals_and_Creatures':             'Animals',
    'Food_Nature_and_Simple_Objects':    'Food',
    'Furniture_and_Structures':          'Furniture',
    'Tools_Instruments_and_Appliances':  'Tools',
    'Vehicles_and_Flight_Systems':       'Vehicles',
    'Wearables_and_Accessories':         'Wear',
}

CHEAT_SHEET = """\
LABEL CHEAT SHEET
=================
Order in CSV and labels column: [tk, bl, mp, mt, ep, nb, nc]
All values: 1 = defect/issue IS present   0 = not present

Col  Full name           What 1 means
---  ------------------  ---------------------------------------------------
tk   too_thick           Lines/outlines rendered too thick or bold
bl   broken_lines        Continuous lines appear broken or interrupted
mp   missing_parts       A structural part visible in natural image is absent
mt   missing_texture     Surface texture of object is missing/insufficient
ep   extra_parts         Tactile has elements not in natural image (hallucinated)
nb   noisy_background    Background has artifacts, patterns, or noise
nc   non_conformant      Overall tactile does NOT match reference concept
                         [Supervisor label: Overall Conformance Issue]

EXAMPLE READINGS
  [0,0,0,0,0,0,0]  All clean — reference quality pair
  [0,0,1,1,0,0,0]  Missing parts + missing texture
  [1,0,0,0,0,0,0]  Lines too thick only
  [0,0,0,0,0,0,1]  Fundamentally wrong concept (non-conformant)

NAVIGATION
  Images live at:  images/<category>/<object>/Natural/<file>
                   images/<category>/<object>/Tactile/<file>
  Category abbrevs in CSV:
    Animals → Animals_and_Creatures
    Food    → Food_Nature_and_Simple_Objects
    Furn    → Furniture_and_Structures
    Tools   → Tools_Instruments_and_Appliances
    Veh     → Vehicles_and_Flight_Systems
    Wear    → Wearables_and_Accessories

EDITING THE CSV
  Change 0/1 values in individual columns OR edit the labels column directly.
  Run:  python code/annotator/import_from_csv.py   to sync edits back to annotations.jsonl
  The 'done' column is set automatically on import (1 if any column was explicitly set).
"""


def main():
    records = []
    with open(ANNOTATIONS_FILE) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # CSV columns: #, cat, obj, nat, tac, tk, bl, mp, mt, ep, nb, nc, done, labels
    fieldnames = ['#', 'cat', 'obj', 'nat', 'tac'] + SHORT + ['done', 'labels']

    rows = []
    for i, r in enumerate(records):
        cat_full = r.get('category', '')
        cat      = CAT_ABBREV.get(cat_full, cat_full[:6])
        nat_file = Path(r['natural_image']).name
        tac_file = Path(r['tactile_image']).name
        vals     = [r.get(k, 0) for k in OPTION_KEYS]
        done     = 1 if r.get('annotated') else 0
        label_str = '[' + ','.join(str(v) for v in vals) + ']'

        row = {'#': i + 1, 'cat': cat, 'obj': r.get('object', ''),
               'nat': nat_file, 'tac': tac_file, 'done': done,
               'labels': label_str}
        for short, val in zip(SHORT, vals):
            row[short] = val
        rows.append(row)

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with open(OUT_CHEAT, 'w') as f:
        f.write(CHEAT_SHEET)

    annotated = sum(1 for r in records if r.get('annotated'))
    print(f"Wrote {len(rows)} rows → {OUT_CSV}")
    print(f"  {annotated} annotated (done=1)  |  {len(rows)-annotated} pending (done=0)")
    print(f"Wrote cheat sheet → {OUT_CHEAT}")


if __name__ == '__main__':
    main()
