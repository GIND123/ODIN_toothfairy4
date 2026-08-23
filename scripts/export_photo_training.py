"""Export labels and the case split for the photo-field model.

Uses exactly the same seed and split arithmetic as ``scripts/compare_configs.py`` and
``scripts/tune_style.py``, so the photo model is trained on the same cases the geometry models
see and evaluated on the same held-out cases. Anything else would make the two prediction
sources incomparable and invite a fusion decision made on contaminated numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bite2text.report.parse import parse_report  # noqa: E402

FIELDS = [
    "overbite", "overjet", "molar_right", "molar_left", "canine_right", "canine_left",
    "midlines", "crossbite", "constriction", "spee", "wilson",
    "crowding_upper", "crowding_lower",
]
#: Classes with less support than this are folded away; a head cannot learn from a handful.
MIN_CLASS_SUPPORT = 12


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("artifacts/geom/bite2text_features.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--photos", type=Path, default=Path("artifacts/photos_small"))
    ap.add_argument("--family", default="reports_intraoral-photo_en")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--labels-out", type=Path, default=Path("artifacts/eval/photo_labels.json"))
    ap.add_argument("--splits-out", type=Path, default=Path("artifacts/eval/photo_splits.json"))
    args = ap.parse_args()

    rows = []
    for case_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        d = case_dir / args.family
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.txt"))
        if not files:
            continue
        parsed = parse_report(files[0].read_text(encoding="utf-8", errors="replace"))
        row = {"case_id": case_dir.name}
        for f in FIELDS:
            row[f] = getattr(parsed, f)
        rows.append(row)

    feats = pd.read_csv(args.features)
    feats = feats[feats["error"].fillna("") == ""].copy()
    df = feats.merge(pd.DataFrame(rows), on="case_id", how="inner").reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(df))
    n_test = int(len(df) * args.test_frac)
    n_val = int(len(df) * args.val_frac)
    splits = {
        "test": df.iloc[order[:n_test]]["case_id"].tolist(),
        "val": df.iloc[order[n_test : n_test + n_val]]["case_id"].tolist(),
        "fit": df.iloc[order[n_test + n_val :]]["case_id"].tolist(),
    }

    # Fold rare classes into "Other" using the fitting split's counts only.
    fit_df = df[df["case_id"].isin(splits["fit"])]
    keep = {
        f: {v for v, n in fit_df[f].value_counts().items() if n >= MIN_CLASS_SUPPORT}
        for f in FIELDS
    }

    labels: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        entry = {}
        for f in FIELDS:
            value = row[f]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            entry[f] = str(value) if str(value) in keep[f] else "Other"
        if entry:
            labels[row["case_id"]] = entry

    have_photos = {p.name for p in args.photos.iterdir() if p.is_dir()} if args.photos.is_dir() else set()
    for name in splits:
        before = len(splits[name])
        splits[name] = [c for c in splits[name] if c in have_photos]
        if before != len(splits[name]):
            print(f"  {name}: dropped {before - len(splits[name])} cases without photos")

    args.labels_out.parent.mkdir(parents=True, exist_ok=True)
    args.labels_out.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    args.splits_out.write_text(json.dumps(splits, indent=1), encoding="utf-8")
    print(f"{len(labels)} labelled cases")
    print(f"  fit={len(splits['fit'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    for f in FIELDS:
        print(f"  {f:18s} classes: {sorted(keep[f]) or ['(all rare)']}")
    print(f"Wrote {args.labels_out} and {args.splits_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
