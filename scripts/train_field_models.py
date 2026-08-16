"""Train geometry -> clinical-field predictors on the Bite2Text corpus.

Labels come from rule-parsing the clinician reports (``bite2text.report.parse``); features come
from the deterministic arch measurements (``bite2text.geom``). Every field is evaluated against
its own majority-class baseline under patient-level cross-validation, because on a corpus this
formulaic a majority vote is already a strong predictor and only a genuine lift is worth
shipping.

Fields whose cross-validated accuracy fails to beat the majority baseline are recorded as
``use_prior``: at inference the composer falls back to the modal value rather than trusting a
model that has not earned it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bite2text.report.parse import parse_report  # noqa: E402

FIELDS = [
    "overbite",
    "overjet",
    "molar_right",
    "molar_left",
    "canine_right",
    "canine_left",
    "midlines",
    "crossbite",
    "constriction",
    "spee",
    "wilson",
    "crowding_upper",
    "crowding_lower",
]

#: Minimum support for a class to be modelled rather than folded into "Other".
MIN_CLASS_SUPPORT = 12


def load_labels(root: Path, family: str) -> pd.DataFrame:
    rows = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        d = case_dir / family
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.txt"))
        if not files:
            continue
        # The evaluation exposes one reference per case; use the first report to match.
        parsed = parse_report(files[0].read_text(encoding="utf-8", errors="replace"))
        row = {"case_id": case_dir.name}
        for f in FIELDS:
            row[f] = getattr(parsed, f)
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"case_id", "error", *FIELDS}
    return [
        c
        for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > len(df) * 0.5
    ]


def evaluate_field(df: pd.DataFrame, cols: list[str], field: str, seed: int) -> dict:
    sub = df[df[field].notna()]
    if len(sub) < 60:
        return {"field": field, "status": "insufficient-labels", "n": int(len(sub))}

    y = sub[field].astype(str)
    counts = y.value_counts()
    y = y.where(y.map(counts) >= MIN_CLASS_SUPPORT, other="Other")
    counts = y.value_counts()
    if len(counts) < 2:
        return {"field": field, "status": "degenerate", "n": int(len(sub))}

    X = sub[cols].to_numpy(dtype=float)
    majority = float(counts.iloc[0] / counts.sum())
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        return {"field": field, "status": "insufficient-per-class", "n": int(len(sub))}

    clf = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=seed,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X, y, cv=cv)
    acc = float(accuracy_score(y, pred))
    return {
        "field": field,
        "status": "ok",
        "n": int(len(sub)),
        "n_classes": int(len(counts)),
        "majority_baseline": round(majority, 4),
        "cv_accuracy": round(acc, 4),
        "cv_balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "lift": round(acc - majority, 4),
        "use_model": bool(acc > majority + 0.01),
        "modal_value": str(counts.index[0]),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("artifacts/geom/bite2text_features.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--family", default="reports_ios_en")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", type=Path, default=Path("artifacts/models/field_report.json"))
    args = ap.parse_args()

    feats = pd.read_csv(args.features)
    feats = feats[feats["error"].fillna("") == ""].copy()
    labels = load_labels(args.root, args.family)
    df = feats.merge(labels, on="case_id", how="inner")
    cols = feature_columns(df)
    print(f"{len(df)} cases with geometry + labels; {len(cols)} features\n")

    results = []
    for field in FIELDS:
        res = evaluate_field(df, cols, field, args.seed)
        results.append(res)
        if res["status"] != "ok":
            print(f"  {field:16s} {res['status']} (n={res.get('n', 0)})")
            continue
        flag = "USE MODEL" if res["use_model"] else "use prior"
        print(
            f"  {field:16s} n={res['n']:4d} k={res['n_classes']}  acc={res['cv_accuracy']:.3f} "
            f"(majority {res['majority_baseline']:.3f}, lift {res['lift']:+.3f})  "
            f"bal={res['cv_balanced_accuracy']:.3f}   {flag}"
        )

    usable = [r for r in results if r.get("use_model")]
    print(f"\n{len(usable)}/{len(FIELDS)} fields beat their majority baseline")

    # Refit the fields that earned it on all available labels, and bundle them with the
    # modal fallbacks so inference needs nothing but this one file.
    bundle: dict[str, object] = {"features": cols, "fields": {}}
    for res in results:
        field = res["field"]
        if res["status"] != "ok":
            continue
        entry: dict[str, object] = {
            "modal_value": res["modal_value"],
            "use_model": res["use_model"],
            "cv_accuracy": res["cv_accuracy"],
            "majority_baseline": res["majority_baseline"],
        }
        if res["use_model"]:
            sub = df[df[field].notna()]
            y = sub[field].astype(str)
            counts = y.value_counts()
            y = y.where(y.map(counts) >= MIN_CLASS_SUPPORT, other="Other")
            clf = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.05, max_depth=3,
                min_samples_leaf=15, l2_regularization=1.0, random_state=args.seed,
            )
            clf.fit(sub[cols].to_numpy(dtype=float), y)
            entry["model"] = clf
        bundle["fields"][field] = entry

    import joblib

    model_path = args.output.with_name("field_models.joblib")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    args.output.write_text(
        json.dumps({"family": args.family, "n_cases": len(df), "features": cols, "fields": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.output} and {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
