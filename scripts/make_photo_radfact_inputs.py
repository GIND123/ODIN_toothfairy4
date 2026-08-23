"""Emit held-out photo-family predictions for the RadFact judge.

RadFact is 0.8 of the final ranking, so the dental-health section has to be justified on the
clinical metric as well as on captioning. An earlier measurement showed that section *hurting*
RadFact — but that was scored against intraoral-scan references, which never mention gingival
status or restorations. Photograph references do, and the hidden test is photo-family, so the
sign of the effect should reverse. This writes the two candidates plus their references so the
judge can settle it.

The split and models mirror ``scripts/holdout_validation.py`` exactly: patient-disjoint, seeded,
and fitted on the training half only.
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

from bite2text.compose import DEFAULT_DENTAL_HEALTH  # noqa: E402
from bite2text.report.parse import ReportFindings, parse_report  # noqa: E402
from bite2text.report.render import render_report  # noqa: E402

FIELDS = [
    "overbite", "overjet", "molar_right", "molar_left", "canine_right", "canine_left",
    "midlines", "crossbite", "constriction", "spee", "wilson",
    "crowding_upper", "crowding_lower",
]
MIN_CLASS_SUPPORT = 12


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("artifacts/geom/bite2text_features.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--family", default="reports_intraoral-photo_en")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/eval"))
    args = ap.parse_args()

    rows = []
    for case_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        d = case_dir / args.family
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.txt"))
        if not files:
            continue
        text = files[0].read_text(encoding="utf-8", errors="replace").strip()
        parsed = parse_report(text)
        row = {"case_id": case_dir.name, "reference": text}
        for f in FIELDS:
            row[f] = getattr(parsed, f)
        rows.append(row)

    feats = pd.read_csv(args.features)
    feats = feats[feats["error"].fillna("") == ""].copy()
    df = feats.merge(pd.DataFrame(rows), on="case_id", how="inner").reset_index(drop=True)
    cols = [
        c for c in df.columns
        if c not in {"case_id", "error", "reference", *FIELDS}
        and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > len(df) * 0.5
    ]

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(df))
    n_test = int(len(df) * args.test_frac)
    test, train = df.iloc[order[:n_test]], df.iloc[order[n_test:]]
    modal = {f: (str(train[f].dropna().mode().iloc[0]) if train[f].notna().any() else None)
             for f in FIELDS}

    preds: dict[str, np.ndarray] = {}
    for field in FIELDS:
        sub = train[train[field].notna()]
        if len(sub) < 60:
            continue
        y = sub[field].astype(str)
        counts = y.value_counts()
        y = y.where(y.map(counts) >= MIN_CLASS_SUPPORT, other="Other")
        if y.nunique() < 2:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=3,
            min_samples_leaf=15, l2_regularization=1.0, random_state=args.seed,
        )
        clf.fit(sub[cols].to_numpy(dtype=float), y)
        preds[field] = clf.predict(test[cols].to_numpy(dtype=float))

    occlusal_only: dict[str, str] = {}
    with_dental: dict[str, str] = {}
    references: dict[str, str] = {}
    for i, case_id in enumerate(test["case_id"].tolist()):
        f = ReportFindings()
        for k, v in modal.items():
            setattr(f, k, v)
        for field, p in preds.items():
            if p[i] not in ("Other", "None", "nan"):
                setattr(f, field, p[i])
        f.crossbite_teeth, f.spacing = [], False
        occlusal_only[case_id] = render_report(f)
        with_dental[case_id] = render_report(f, DEFAULT_DENTAL_HEALTH)
        references[case_id] = test["reference"].iloc[i]

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "refs_photo.json").write_text(json.dumps(references, indent=1), encoding="utf-8")
    (args.outdir / "pred_photo_occ.json").write_text(json.dumps(occlusal_only, indent=1), encoding="utf-8")
    (args.outdir / "pred_photo_dh.json").write_text(json.dumps(with_dental, indent=1), encoding="utf-8")
    print(f"wrote {len(references)} held-out photo-family cases to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
