"""Apply the deliberate `too_thick` corruption prompt to a sample of the corpus.

Supervisor's brief (2026-08-19, recorded in project_notes/data_collection_protocol.md):
generate synthetic positives by API-based deliberate corruption, aiming at
obvious defects rather than fine-grained pixel-level ones. This script covers
the thickness half.

Selection: every 5th Tactile rendering, ordered numerically, within each class,
across the six semantic families. The pipeline output directories
(`gpt_generated*`) are excluded -- they are pair_TIMESTAMP/ dumps, not a
family/class taxonomy, and most of their contents are already in the splits.

Why the prompt is inlined here rather than routed through
`generation.gpt_generator.edit_tactile`: that helper wraps the instruction in
"Apply the following CORRECTION while preserving all other features", which
fights a request to damage the image on purpose. The corruption text is sent as
its own instruction instead.

The natural photograph is deliberately NOT sent. Neither the BANA `too_thick`
criterion nor the human annotation criterion references the photograph, and
supplying it invites the model to "improve" the rendering while it is being
asked to degrade it.

Resumable: an output that already exists is skipped, so a walltime kill costs
nothing. Dry-run by default; --apply spends money.

    python research/corruption/generate_corruptions.py                # plan only
    python research/corruption/generate_corruptions.py --apply        # generate
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, REPO_ROOT, require_data_root  # noqa: E402

FAMILIES = [
    "Animals_and_Creatures", "Food_Nature_and_Simple_Objects",
    "Furniture_and_Structures", "Tools_Instruments_and_Appliances",
    "Vehicles_and_Flight_Systems", "Wearables_and_Accessories",
]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Verbatim from project_notes/data_collection_protocol.md. The numbers in it are
# measured: real too_thick positives sit at a gap-to-stroke ratio of ~1.9 with
# the tightest gaps below 1.0, against ~3.3 for clean renderings. A flat "4x
# thicker" instruction drives that ratio to zero and produces a black mass no
# human labelled example resembles.
PROMPTS = {}

PROMPTS["too_thick"] = (
    "Keeping every aspect of the given image identical — same viewpoint, same "
    "parts, same texture fills, same composition, same number of elements — "
    "increase the stroke width of the outlines so that the drawing violates the "
    "minimum tactile separation. Thicken most heavily wherever two lines already "
    "run close together, meet, or cross: at those places the strokes must swell "
    "until they fuse into a single solid mass with no gap a fingertip could "
    "find. Elsewhere thicken more moderately, so that across the drawing the "
    "typical clear space between neighbouring strokes is no wider than about "
    "twice a stroke, and the tightest spaces are narrower than a single stroke. "
    "The drawing must remain recognisable as the same object — do not reduce it "
    "to a solid black silhouette, and do not remove, move or redraw any element. "
    "The failure must come only from stroke width. This is a deliberate "
    "corruption for model training."
)

# Broken lines. Two constraints do the real work here. The gap size is stated in
# stroke widths because "obvious discontinuity" needs a number -- real positives
# need ~1.3 stroke widths of dilation to close, i.e. gaps around 2.5 strokes,
# whereas a 3px gap is half a stroke and is exactly the "fine-grained missing
# pixels" the supervisor warned against. And the dashed-style carve-out matters
# because BANA treats a dashed line, and a small break where lines cross, as
# legitimate design -- flagging those is what made the parked v4 question set
# overshoot.
PROMPTS["broken_lines"] = (
    "Keeping every aspect of the given image identical — same stroke width, same "
    "texture fills, same parts, same composition — break the continuous outlines "
    "with unmistakable gaps, so that a fingertip tracing an edge loses the line "
    "completely and cannot tell whether the contour has ended or continues. Each "
    "gap must be clearly wider than the stroke is thick — roughly two to three "
    "times the stroke width, large enough to be obvious at a glance rather than a "
    "few missing pixels. Make the gaps irregular in both length and spacing so "
    "they never read as a dashed or dotted line style, and place them part-way "
    "along long smooth contours, not at corners and not where two lines cross — a "
    "small break at a crossing is a legitimate drawing convention, not a defect. "
    "Break only the structural outlines; leave any texture or fill pattern inside "
    "the regions untouched. Leave several outlines completely intact so the "
    "breakage reads as a fault rather than as the style of the drawing. Do not "
    "change the thickness of any line. This is a deliberate corruption for model "
    "training."
)

# Both at once. The gap has to grow with the stroke or it is swallowed by the
# surrounding ink -- a 20px gap cut into a 30px stroke barely registers.
PROMPTS["both"] = (
    "Keeping every aspect of the given image identical in parts, viewpoint, "
    "texture fills and composition, corrupt the linework in two ways at once. "
    "First, thicken the outlines — most heavily where lines already run close "
    "together, meet or cross — until neighbouring elements fuse and the clear "
    "space between strokes is no wider than a stroke itself, while the object "
    "still remains recognisable rather than a solid silhouette. Second, break "
    "those same thickened strokes with unmistakable gaps: because the strokes are "
    "now heavy, each gap must be wider still — roughly two to three times the "
    "new, thickened stroke width — so that it stays obvious rather than being "
    "swallowed by the surrounding ink. Place the gaps irregularly along long "
    "contours, never in a repeating dashed pattern and never at junctions. The "
    "result must fail in two distinct ways at once: a fingertip cannot separate "
    "adjacent elements, and cannot trace any single contour from one end to the "
    "other. This is a deliberate corruption for model training."
)


def numeric_key(name: str):
    """Sort 1,2,...,10 rather than 1,10,2 — 'every 5th' has to mean something."""
    stem = os.path.splitext(name)[0]
    m = re.match(r"^(\d+)", stem)
    return (0, int(m.group(1)), stem) if m else (1, 0, stem)


def read_api_key() -> str:
    """Read the key from the workspace file rather than an argument or literal.

    api_key.txt doubles as a notes file and is not shell-sourceable, so take the
    export line only. Never echo the value.
    """
    env = os.environ.get("OPENAI_API_KEY", "").strip()
    if env:
        return env
    keyfile = DATA_ROOT / "TactileEval_Context_And_Data" / "api" / "api_key.txt"
    if keyfile.exists():
        for line in keyfile.read_text().splitlines():
            if line.startswith("export OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"no API key: set OPENAI_API_KEY or add an export line to {keyfile}")


def natural_partner_index() -> dict:
    """tactile rel path -> natural rel path, from the splits and the v1 annotations.

    The VLM judge scores a PAIR, so a corrupted rendering with no natural
    partner cannot be pre-scored later. 227 of 228 selections resolve.
    """
    idx = {}
    splits = DATA_ROOT / "data" / "splits_v3_unified"
    for split in ("train", "val", "test"):
        f = splits / f"{split}.jsonl"
        if f.exists():
            for line in f.open():
                r = json.loads(line)
                idx[r["tactile_image"]] = r["natural_image"]
    ann = DATA_ROOT / "annotations.jsonl"
    if ann.exists():
        for line in ann.open():
            r = json.loads(line)
            idx.setdefault(r["tactile_image"], r["natural_image"])
    return idx


def select(images_root: Path, stride: int, offset: int = 0) -> list:
    picks = []
    for fam in FAMILIES:
        fam_dir = images_root / fam
        if not fam_dir.is_dir():
            continue
        for cls in sorted(os.listdir(fam_dir)):
            tac_dir = fam_dir / cls / "Tactile"
            if not tac_dir.is_dir():
                continue
            files = sorted(
                (f for f in os.listdir(tac_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXT),
                key=numeric_key,
            )
            for f in files[offset::stride]:
                picks.append({"family": fam, "object": cls, "file": f,
                              "tactile_image": f"{fam}/{cls}/Tactile/{f}"})
    return picks


def out_name(rec: dict) -> str:
    stem = os.path.splitext(rec["file"])[0]
    safe = re.sub(r"[^A-Za-z0-9]+", "-", rec["object"]).strip("-")
    return f"{rec['family']}__{safe}__{stem}.png"


def normalize_background(img: Image.Image) -> tuple:
    """Force a near-white ground to pure white.

    gpt-image-1 sometimes returns the drawing on a light grey ground instead of
    white -- 3 of the 15 broken-lines pilot images, against 0 of 40 for
    thickness. Otsu copes, so it does not break the judges mechanically, but it
    is a PROVENANCE signal: if grey grounds correlate with one corruption type, a
    judge can learn "grey means broken" without looking at a line. That is the
    same failure already contaminating `missing_texture`, where a single metadata
    bit carries AUC 0.82.

    A white-point rescale (bg -> 255) is safe on line art: it leaves near-black
    ink near-black. Images that are already white, or that are inverted
    (white-on-dark), are left untouched.
    """
    gray = np.asarray(img.convert("L"))
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    if not (200.0 <= bg < 253.0):
        return img, bg, False
    arr = np.asarray(img.convert("RGB")).astype(np.float32) * (255.0 / bg)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), bg, True


def pad_to_square(img: Image.Image) -> Image.Image:
    """White-pad to square before sending.

    The API returns a square image, so a non-square input gets cropped or
    distorted -- and the prompt asks for composition to be preserved. 66% of the
    selection is non-square and 22 images exceed 1.5:1, where the crop lops off
    real content (the first pilot cut the wings off a bat). White padding is
    free here: these are line drawings on white already.
    """
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def corrupt(client, src: Path, prompt: str, quality: str, size: str):
    """One images.edit call. Returns (PIL image, seconds)."""
    with Image.open(src) as im:
        im = pad_to_square(im.convert("RGB"))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    buf.name = "tactile.png"
    buf.seek(0)
    t0 = time.time()
    resp = client.images.edit(
        model="gpt-image-1", image=buf, prompt=prompt,
        size=size, quality=quality, n=1,
    )
    img = Image.open(io.BytesIO(base64.b64decode(resp.data[0].b64_json))).convert("RGB")
    img, bg, fixed = normalize_background(img)
    return img, time.time() - t0, bg, fixed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "verify_generated_batch"))
    ap.add_argument("--corruption", default="too_thick", choices=sorted(PROMPTS),
                    help="which deliberate corruption to apply")
    ap.add_argument("--stride", type=int, default=5, help="take every Nth image per class")
    ap.add_argument("--offset", type=int, default=0,
                    help="0-based start within each class. Use a different offset per "
                         "corruption type to keep the batches on disjoint sources: the "
                         "too_thick batch used offset 0, so offset 2 shares nothing with it.")
    ap.add_argument("--quality", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--limit", type=int, default=0, help="cap the number generated this run")
    ap.add_argument("--sleep", type=float, default=0.5, help="pause between calls")
    ap.add_argument("--apply", action="store_true", help="actually call the API and spend money")
    args = ap.parse_args()

    require_data_root()
    images_root = DATA_ROOT / "images"
    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.jsonl"

    prompt = PROMPTS[args.corruption]
    picks = select(images_root, args.stride, args.offset)
    partners = natural_partner_index()
    for r in picks:
        r["natural_image"] = partners.get(r["tactile_image"])
        r["output"] = out_name(r)

    done = set()
    if manifest_path.exists():
        for line in manifest_path.open():
            rec = json.loads(line)
            if (img_dir / rec["output"]).exists():
                done.add(rec["tactile_image"])
    todo = [r for r in picks if r["tactile_image"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    fams = {}
    for r in picks:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    print(f"corpus       : {images_root}")
    print(f"corruption   : {args.corruption}")
    print(f"selection    : every {args.stride}th Tactile image per class from offset "
          f"{args.offset} (positions {args.offset+1}, {args.offset+1+args.stride}, ...), "
          f"{len(FAMILIES)} families")
    for f, n in fams.items():
        print(f"   {f:34} {n:4}")
    print(f"total selected : {len(picks)}")
    print(f"already done   : {len(done)}")
    print(f"to generate    : {len(todo)}")
    print(f"no natural partner (cannot be VLM-scored later): "
          f"{sum(1 for r in picks if not r['natural_image'])}")
    print(f"quality={args.quality} size={args.size}  ->  out {img_dir}")

    if not args.apply:
        print("\nDRY RUN — nothing called, nothing written. Re-run with --apply.")
        for r in todo[:5]:
            print(f"   {r['tactile_image']}  ->  {r['output']}")
        return

    img_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PROMPT.txt").write_text(f"corruption: {args.corruption}\n\n{prompt}\n")

    from openai import OpenAI
    client = OpenAI(api_key=read_api_key())

    ok = fail = 0
    t_start = time.time()
    with manifest_path.open("a") as mf:
        for i, r in enumerate(todo, 1):
            src = images_root / r["tactile_image"]
            try:
                img, secs, bg, fixed = corrupt(client, src, prompt, args.quality, args.size)
                img.save(img_dir / r["output"])
                rec = dict(r, corruption=args.corruption, quality=args.quality,
                           size=args.size, seconds=round(secs, 1),
                           returned_background=round(bg, 1), background_normalized=fixed,
                           generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                ok += 1
                print(f"[{i}/{len(todo)}] {secs:5.1f}s  {r['output']}", flush=True)
            except Exception as exc:
                fail += 1
                print(f"[{i}/{len(todo)}] FAILED {r['tactile_image']}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
            time.sleep(args.sleep)

    print(f"\ngenerated {ok}, failed {fail}, {(time.time()-t_start)/60:.1f} min")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
