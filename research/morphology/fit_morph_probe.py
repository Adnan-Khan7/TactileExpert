"""Fit the morphological probe and report it against the pre-registered bar.

The bar is SESSION_NOTES.md 6a, fixed on 2026-08-18 before any feature existed:

    too_thick      must clear ink fraction by >= +0.13 (paired bootstrap CI on
                   the difference must exclude zero)
    broken_lines   ink fraction is at chance there, so the comparator is the
                   best judge -- ResNet-50 at 0.783
    missing_texture must NOT move; but it is confounded with image provenance
                   (README 4i), so read a gain as leakage first

This script enforces the parts of that registration it can enforce:

  * the outline/fill threshold and the model family are selected on VAL, from
    val AUC alone. Test is scored once, at the end, with the selection frozen.
  * `qc__*` features are dropped before fitting. Image dimensions alone
    separate the generated corpus from the scraped one, and provenance
    predicts labels.
  * both noise sources are reported: seed 2sigma across 12 classifier seeds,
    AND a test-set bootstrap. Section 5's 12-seed apparatus covers only the
    first, and on the thin options the second is 3-8x larger.
  * the controls run unconditionally, not on request. A result that only
    exists pooled across sources, or only with the synthetic corruptions in,
    is reported as such next to the headline.

Usage:
    python research/morphology/fit_morph_probe.py \
        --features experiments_out/morph_features_tv974.json \
        --baselines experiments_out/trivial_baselines_tv974.json \
        --out experiments_out/morph_probe_tv974.json
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA_ROOT, require_data_root  # noqa: E402

OPTIONS = ["too_thick", "broken_lines", "missing_parts", "missing_texture", "extra_parts"]
N_REFITS = 12
REGISTERED_BAR = {"too_thick": 0.13}


def read_split(splits_dir: Path, split: str):
    labels = collections.defaultdict(dict)
    tactile, synthetic = {}, {}
    with open(splits_dir / f"{split}.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            pid = r["pair_id"]
            labels[pid][r["option_id"]] = int(r["label"])
            tactile[pid] = r["tactile_image"]
            synthetic[pid] = bool(r.get("synthetic_corruption", False))
    return labels, tactile, synthetic


def make_model(kind: str, seed: int):
    """Impute first: a degenerate rendering yields NaN, and dropping the row
    would quietly change the split."""
    if kind == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.06,
                                       min_samples_leaf=15, l2_regularization=1.0,
                                       random_state=seed))


def fit_once(kind, Xtr, ytr, Xev, seed=0):
    m = make_model(kind, seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xev)[:, 1]


def refit_spread(kind, Xtr, ytr, Xev, yev, n=N_REFITS, seed=0):
    """Model variance, estimated by refitting on bootstrap resamples of TRAIN.

    Note what this is NOT. Section 5's 2sigma is *seed* variance, which for the
    neural judges is real -- ViT-B/16 does not even reproduce bit-identically.
    Both classifiers here are deterministic given the same rows, so their seed
    variance is exactly zero and quoting it would be a fake precision. The
    analogous honest quantity is how much the fitted probe moves when the
    training sample moves, so that is what is reported.
    """
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        i = rng.choice(len(ytr), len(ytr), replace=True)
        if len(np.unique(ytr[i])) < 2:
            continue
        aucs.append(safe_auc(yev, fit_once(kind, Xtr[i], ytr[i], Xev)))
    a = np.array([x for x in aucs if np.isfinite(x)])
    return float(2 * a.std()) if len(a) > 2 else float("nan")


def boot_indices(y, n_boot, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return [np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
            for _ in range(n_boot)]


def paired_delta(y, x_new, x_ref, idx):
    d = np.array([roc_auc_score(y[i], x_new[i]) - roc_auc_score(y[i], x_ref[i]) for i in idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(roc_auc_score(y, x_new) - roc_auc_score(y, x_ref)), float(lo), float(hi)


def safe_auc(y, x):
    return roc_auc_score(y, x) if 0 < y.sum() < len(y) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--baselines", default=None, help="trivial_baselines_*.json, for the ink-fraction comparator")
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    require_data_root()
    splits_dir = Path(args.splits_dir) if args.splits_dir else DATA_ROOT / "data" / "splits_v3_unified"
    feat_path = Path(args.features)
    if not feat_path.is_absolute():
        feat_path = DATA_ROOT / feat_path
    blob = json.loads(feat_path.read_text())
    feats = blob["features"]

    if blob["splits_dir"] != str(splits_dir):
        raise SystemExit(f"features were extracted from {blob['splits_dir']}, not {splits_dir}")

    out_path = Path(args.out) if args.out else None
    if out_path and not out_path.is_absolute():
        out_path = DATA_ROOT / out_path
    if out_path and out_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {out_path} (pass --force)")

    data = {}
    for sp in ("train", "val", "test"):
        labels, tactile, synthetic = read_split(splits_dir, sp)
        pids = [p for p in sorted(labels) if p in feats]
        if len(pids) < len(labels):
            print(f"warning: {len(labels) - len(pids)} {sp} pairs missing from the feature file")
        data[sp] = dict(pids=pids, labels=labels, tactile=tactile, synthetic=synthetic)
    print(f"train {len(data['train']['pids'])} / val {len(data['val']['pids'])} / test {len(data['test']['pids'])}")

    # ── feature groups. Shared features plus one outline variant per candidate
    #    threshold; the variant is what gets selected on val.
    all_names = sorted({k for f in feats.values() for k in f if not k.startswith("qc__")})
    tagged = [n for n in all_names if "@" in n]
    shared = [n for n in all_names if "@" not in n]
    fracs = sorted({n.split("@")[1] for n in tagged}, key=float)
    print(f"{len(shared)} shared features, outline variants at {fracs}")

    def matrix(sp, frac):
        cols = shared + [n for n in tagged if n.endswith("@" + frac)]
        return np.array([[feats[p].get(c, np.nan) for c in cols] for p in data[sp]["pids"]], dtype=float), cols

    def y_of(sp, opt, pids=None):
        pids = pids or data[sp]["pids"]
        return np.array([data[sp]["labels"][p][opt] for p in pids])

    # ── ink-fraction comparator, recomputed from the feature file itself so the
    #    two artifacts cannot silently disagree
    #    `ink_fraction` is post-speckle-removal and differs from the published
    #    baseline on 80 of 193 test images; `qc__ink_fraction_raw` is the exact
    #    quantity README 4h reports. Compare against the published one.
    ink = {sp: np.array([feats[p]["qc__ink_fraction_raw"] for p in data[sp]["pids"]]) for sp in data}
    if args.baselines:
        bp = Path(args.baselines)
        if not bp.is_absolute():
            bp = DATA_ROOT / bp
        published = json.loads(bp.read_text())["ink_fraction_auc"]
        for opt in OPTIONS:
            got = roc_auc_score(y_of("test", opt), ink["test"])
            want = published[opt]["auc"]
            if abs(got - want) > 1e-3:
                raise SystemExit(f"ink fraction on {opt} is {got:.4f} here but {want:.4f} in "
                                 f"{bp.name}; the two artifacts disagree")
        print(f"ink-fraction comparator matches {bp.name}")

    report = {"features_file": str(feat_path), "splits_dir": str(splits_dir),
              "n_refits": N_REFITS, "n_boot": args.n_boot, "registered_bar": REGISTERED_BAR,
              "note_seed_variance": "both classifiers are deterministic; seed 2sigma is "
                                    "structurally zero, so train-resampling is reported instead",
              "options": {}}

    for opt in OPTIONS:
        ytr, yva, yte = y_of("train", opt), y_of("val", opt), y_of("test", opt)

        # ── SELECTION on val only ───────────────────────────────────────────
        sel = []
        for kind in ("logreg", "hgb"):
            for frac in fracs:
                Xtr, _ = matrix("train", frac)
                Xva, _ = matrix("val", frac)
                sel.append((safe_auc(yva, fit_once(kind, Xtr, ytr, Xva)), kind, frac))
        sel.sort(reverse=True)
        val_auc, best_kind, best_frac = sel[0]
        print(f"\n=== {opt}  (train {ytr.sum()}+ / val {yva.sum()}+ / test {yte.sum()}+)")
        print(f"  selected on val: {best_kind}, outline_min_frac={best_frac}, val AUC {val_auc:.3f}")

        # ── TEST scored once, selection frozen ──────────────────────────────
        Xtr, cols = matrix("train", best_frac)
        Xte, _ = matrix("test", best_frac)
        mean_te = fit_once(best_kind, Xtr, ytr, Xte)
        auc = safe_auc(yte, mean_te)
        two_sigma = refit_spread(best_kind, Xtr, ytr, Xte, yte, seed=args.seed)

        idx = boot_indices(yte, args.n_boot, args.seed)
        draws = np.array([roc_auc_score(yte[i], mean_te[i]) for i in idx])
        blo, bhi = np.percentile(draws, [2.5, 97.5])
        d, dlo, dhi = paired_delta(yte, mean_te, ink["test"], idx)
        clears = bool(dlo > 0)
        bar = REGISTERED_BAR.get(opt)
        meets_bar = bool(clears and d >= bar) if bar else None

        print(f"  TEST AUC {auc:.3f}  train-refit 2sigma {two_sigma:.3f}  "
              f"test bootstrap [{blo:.3f}, {bhi:.3f}]")
        print(f"  vs ink fraction {roc_auc_score(yte, ink['test']):.3f}: "
              f"delta {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}] {'SIG' if clears else 'n.s.'}"
              + (f"  |  registered bar +{bar:.2f}: {'MET' if meets_bar else 'NOT MET'}" if bar else ""))

        entry = {
            "n_pos": {"train": int(ytr.sum()), "val": int(yva.sum()), "test": int(yte.sum())},
            "selected": {"model": best_kind, "outline_min_frac": best_frac, "val_auc": val_auc},
            "test_auc": auc, "train_refit_two_sigma": two_sigma, "seed_two_sigma": 0.0,
            "test_bootstrap_ci95": [float(blo), float(bhi)],
            "ink_fraction_auc": float(roc_auc_score(yte, ink["test"])),
            "delta_vs_ink": d, "delta_ci95": [dlo, dhi], "significant": clears,
            "registered_bar": bar, "meets_registered_bar": meets_bar,
            "controls": {},
        }

        # ── CONTROL 1: provenance. A gain that exists only pooled is leakage.
        src = np.array(["gpt" if data["test"]["tactile"][p].startswith("gpt_generated") else "web"
                        for p in data["test"]["pids"]])
        for s in ("web", "gpt"):
            m = src == s
            n_pos = int(yte[m].sum())
            entry["controls"][f"auc_within_{s}"] = {
                "n": int(m.sum()), "n_pos": n_pos,
                "auc": float(safe_auc(yte[m], mean_te[m])),
                "estimable": bool(3 <= n_pos <= m.sum() - 3),
            }
        # the leakage-controlled version of the headline: same paired delta, run
        # inside one source, where provenance cannot explain it. This is a
        # CONTROL, not the registered comparison -- 6a forbids promoting a
        # subgroup that clears when the pre-registered one does not.
        for s_name in ("web", "gpt"):
            m = src == s_name
            ys = yte[m]
            if 3 <= ys.sum() <= m.sum() - 3:
                d_s, lo_s, hi_s = paired_delta(ys, mean_te[m], ink["test"][m],
                                               boot_indices(ys, args.n_boot, args.seed))
                entry["controls"][f"delta_vs_ink_within_{s_name}"] = {
                    "delta": d_s, "ci95": [lo_s, hi_s], "significant": bool(lo_s > 0)}
            else:
                entry["controls"][f"delta_vs_ink_within_{s_name}"] = None

        w, g = entry["controls"]["auc_within_web"], entry["controls"]["auc_within_gpt"]
        print(f"  within-source: web {w['auc']:.3f} (n={w['n']}, {w['n_pos']}+)"
              f"{'' if w['estimable'] else ' NOT ESTIMABLE'}"
              f" | gpt {g['auc']:.3f} (n={g['n']}, {g['n_pos']}+)"
              f"{'' if g['estimable'] else ' NOT ESTIMABLE'}")
        dw = entry["controls"].get("delta_vs_ink_within_web")
        if dw:
            print(f"  delta vs ink within web only: {dw['delta']:+.3f} "
                  f"[{dw['ci95'][0]:+.3f}, {dw['ci95'][1]:+.3f}] "
                  f"{'SIG' if dw['significant'] else 'n.s.'}  (control, not the registered comparison)")

        # ── CONTROL 2: the synthetic corruptions are an easier target
        nat = np.array([not data["test"]["synthetic"][p] for p in data["test"]["pids"]])
        entry["controls"]["auc_natural_only"] = {
            "n": int(nat.sum()), "n_pos": int(yte[nat].sum()),
            "auc": float(safe_auc(yte[nat], mean_te[nat])),
            "ink_auc": float(safe_auc(yte[nat], ink["test"][nat])),
        }
        na = entry["controls"]["auc_natural_only"]
        print(f"  natural-only: {na['auc']:.3f} (n={na['n']}, {na['n_pos']}+), ink {na['ink_auc']:.3f}")
        report["options"][opt] = entry

    # ── CONTROL 3: can the features alone recover the image source? If so they
    #    carry a provenance signature, and every option whose prevalence
    #    differs by source is partly explained by it rather than by geometry.
    Xtr, _ = matrix("train", fracs[0])
    Xte, _ = matrix("test", fracs[0])
    src_tr = np.array([data["train"]["tactile"][p].startswith("gpt_generated") for p in data["train"]["pids"]], int)
    src_te = np.array([data["test"]["tactile"][p].startswith("gpt_generated") for p in data["test"]["pids"]], int)
    src_auc = float(safe_auc(src_te, fit_once("hgb", Xtr, src_tr, Xte)))
    report["control_source_from_features_auc"] = src_auc

    # ── CONTROL 4: is the topology just ink density with extra steps?
    ep = np.array([feats[p].get("endpoints_per_len", np.nan) for p in data["test"]["pids"]])
    ok = np.isfinite(ep) & np.isfinite(ink["test"])
    r_ep = float(np.corrcoef(ep[ok], ink["test"][ok])[0, 1])
    report["control_endpoints_vs_ink_pearson_r"] = r_ep

    print(f"\ncontrols")
    print(f"  source recoverable from features: AUC {src_auc:.3f}"
          + ("  <-- features carry provenance; read every pooled gain with that in mind"
             if src_auc > 0.8 else ""))
    print(f"  endpoints_per_len vs ink_fraction: r = {r_ep:+.3f}"
          + ("  <-- topology is ink density restated" if abs(r_ep) > 0.9 else ""))

    # ── verdict against the registration ────────────────────────────────────
    print("\nverdict against SESSION_NOTES 6a")
    for opt in ("too_thick", "broken_lines", "missing_texture"):
        e = report["options"][opt]
        if opt == "too_thick":
            v = "MET" if e["meets_registered_bar"] else "NOT MET"
            print(f"  too_thick       {v} -- delta {e['delta_vs_ink']:+.3f} vs registered +0.13")
        elif opt == "broken_lines":
            print(f"  broken_lines    AUC {e['test_auc']:.3f} vs ResNet-50 0.783 "
                  f"({'above' if e['test_auc'] > 0.783 else 'below'}; comparator is the judge, not ink)")
        else:
            print(f"  missing_texture AUC {e['test_auc']:.3f} (negative control; "
                  f"source recoverable at {src_auc:.3f}, so check leakage before reading this)")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out_path}")
    else:
        print("\n(no --out given; nothing written)")


if __name__ == "__main__":
    main()
