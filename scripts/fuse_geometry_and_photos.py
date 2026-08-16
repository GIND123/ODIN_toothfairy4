"""Fuse arch geometry with photograph predictions and measure whether it helps.

Geometry sees occlusion in three dimensions but cannot see colour, soft tissue, or which tooth
is which. Photographs see all of that but flatten the bite. The occlusal fields — especially the
sagittal classes, where geometry is weakest at 0.52-0.63 — are plausibly where the two
modalities complement each other.

Rather than hand-designing the combination, the vision model's per-field answer is added to the
geometry feature matrix as a categorical column and the same gradient-boosting setup is
retrained. The model is then free to learn how far to trust each source per field, and the
comparison against geometry-only is like-for-like: same split, same seed, same hyperparameters.

Everything is reported under a patient-disjoint held-out split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bite2text.report.parse import parse_report  # noqa: E402

FIELDS = [
    "overbite", "overjet", "molar_right", "molar_left", "canine_right", "canine_left",
    "midlines", "crossbite", "crowding_upper", "crowding_lower",
]
MIN_CLASS_SUPPORT = 12


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("artifacts/geom/bite2text_features.csv"))
    ap.add_argument("--photo-predictions", type=Path, default=Path("artifacts/eval/photo_occlusal_all.json"))
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--family", default="reports_intraoral-photo_en")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--output", type=Path, default=Path("artifacts/eval/fusion_report.json"))
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
    labels = pd.DataFrame(rows)

    feats = pd.read_csv(args.features)
    feats = feats[feats["error"].fillna("") == ""].copy()
    df = feats.merge(labels, on="case_id", how="inner").reset_index(drop=True)

    photo = json.loads(args.photo_predictions.read_text(encoding="utf-8"))
    geom_cols = [
        c for c in df.columns
        if c not in {"case_id", "error", *FIELDS}
        and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > len(df) * 0.5
    ]

    # Encode each photograph answer as an integer code; unseen/missing becomes NaN, which the
    # gradient booster handles natively as "no opinion from this modality".
    photo_cols: list[str] = []
    for field in FIELDS:
        values = sorted({str(v.get(field)) for v in photo.values() if v.get(field) is not None})
        if not values:
            continue
        mapping = {v: i for i, v in enumerate(values)}
        col = f"photo_{field}"
        df[col] = [
            mapping.get(str(photo.get(c, {}).get(field)), np.nan) for c in df["case_id"]
        ]
        photo_cols.append(col)
    coverage = float(np.mean([c in photo for c in df["case_id"]]))
    print(f"{len(df)} cases; geometry features={len(geom_cols)}, photo columns={len(photo_cols)}, "
          f"photo coverage={coverage:.0%}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(df))
    n_test = int(len(df) * args.test_frac)
    test, train = df.iloc[order[:n_test]], df.iloc[order[n_test:]]

    def evaluate(cols: list[str], field: str) -> float | None:
        sub = train[train[field].notna()]
        holdout = test[test[field].notna()]
        if len(sub) < 60 or len(holdout) < 20:
            return None
        y = sub[field].astype(str)
        counts = y.value_counts()
        y = y.where(y.map(counts) >= MIN_CLASS_SUPPORT, other="Other")
        if y.nunique() < 2:
            return None
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=3,
            min_samples_leaf=15, l2_regularization=1.0, random_state=args.seed,
        )
        clf.fit(sub[cols].to_numpy(dtype=float), y)
        pred = clf.predict(holdout[cols].to_numpy(dtype=float))
        return float((pred == holdout[field].astype(str).to_numpy()).mean())

    print(f"\n{'field':<16}{'geom':>8}{'geom+photo':>12}{'photo raw':>11}{'majority':>10}   verdict")
    results = {}
    for field in FIELDS:
        holdout = test[test[field].notna()]
        if len(holdout) < 20:
            continue
        geom_acc = evaluate(geom_cols, field)
        fused_acc = evaluate(geom_cols + photo_cols, field)
        if geom_acc is None or fused_acc is None:
            continue
        # The vision model's own answer, with no learning on top.
        raw = [
            (str(photo.get(c, {}).get(field)) == str(t))
            for c, t in zip(holdout["case_id"], holdout[field])
            if photo.get(c, {}).get(field) is not None
        ]
        raw_acc = float(np.mean(raw)) if raw else float("nan")
        majority = float(train[field].astype(str).value_counts(normalize=True).iloc[0])
        better = fused_acc > geom_acc + 0.01
        results[field] = {
            "geometry": round(geom_acc, 4), "fused": round(fused_acc, 4),
            "photo_raw": None if np.isnan(raw_acc) else round(raw_acc, 4),
            "majority": round(majority, 4), "fusion_helps": better,
        }
        print(
            f"{field:<16}{geom_acc:8.3f}{fused_acc:12.3f}{raw_acc:11.3f}{majority:10.3f}   "
            f"{'FUSION WINS' if better else 'geometry alone'}"
        )

    wins = [f for f, r in results.items() if r["fusion_helps"]]
    print(f"\nfusion improves {len(wins)}/{len(results)} fields: {', '.join(wins) or 'none'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"n_cases": len(df), "fields": results,
                                       "fusion_wins": wins}, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
