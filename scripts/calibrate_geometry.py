"""Validate and calibrate the geometry engine against the Bits2Bites annotations.

Bits2Bites supplies 200 registered arch pairs labelled with exactly the occlusal findings the
Bite2Text template asks for, so it is the only ground truth available for checking that the
measurements in ``bite2text.geom`` mean what they claim to mean.

Three things happen here:

1. **Signal check** — how well each raw measurement separates its target label. A measurement
   that cannot beat the majority-class baseline is reported as such rather than shipped.
2. **Laterality audit** — the left/right sign of the frame is a mirror ambiguity that surface
   geometry cannot resolve. The FDI tooth numbers in the ``Transversal Bite`` strings do
   resolve it, so we settle it empirically here.
3. **Calibration** — fit and cross-validate a model mapping measurements to template values,
   and persist it with the label priors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TARGETS = ["Left Class", "Right Class", "Anterior Bite", "Transversal Bite", "Median Lines"]


def parse_transversal(raw: str) -> tuple[str, list[int]]:
    """Split ``Cross Bite 15-16`` into a condition and its FDI tooth numbers."""
    text = str(raw).strip()
    teeth = [int(t) for t in re.findall(r"\b(\d{2})\b", text)]
    low = text.lower()
    if low.startswith("normal"):
        condition = "Normal"
    elif "scissor" in low and "cross" in low:
        condition = "Mixed"
    elif "scissor" in low:
        condition = "Scissor Bite"
    elif "cross" in low:
        condition = "Cross Bite"
    else:
        condition = "Unknown"
    return condition, teeth


def fdi_side(teeth: list[int]) -> str | None:
    """Patient side implied by FDI quadrants: 1/4 are right, 2/3 are left."""
    quadrants = {t // 10 for t in teeth if 10 <= t <= 48}
    right = bool(quadrants & {1, 4})
    left = bool(quadrants & {2, 3})
    if right and left:
        return "both"
    if right:
        return "right"
    if left:
        return "left"
    return None


def load(features_csv: Path, annotations_csv: Path) -> pd.DataFrame:
    feats = pd.read_csv(features_csv)
    feats = feats[feats["error"].fillna("") == ""].copy()
    feats["Patient"] = feats["case_id"].str.split("/").str[-1].astype(int)
    feats["split"] = feats["case_id"].str.split("/").str[0]
    ann = pd.read_csv(annotations_csv)
    merged = feats.merge(ann, on="Patient", how="inner")
    cond, teeth = zip(*merged["Transversal Bite"].map(parse_transversal))
    merged["tv_condition"] = cond
    merged["tv_teeth"] = teeth
    merged["tv_side"] = [fdi_side(t) for t in teeth]
    return merged


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"Patient", "split", "case_id", "error", "tv_condition", "tv_teeth", "tv_side", *TARGETS}
    return [
        c
        for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > len(df) * 0.5
    ]


def report_signal(df: pd.DataFrame, measure: str, target: str) -> None:
    print(f"\n  {measure}  by  {target}")
    grouped = df.groupby(target)[measure]
    for value, series in grouped:
        series = series.dropna()
        if series.empty:
            continue
        print(
            f"    {str(value)[:34]:34s} n={len(series):3d}  "
            f"median={series.median():+7.2f}  IQR=[{series.quantile(.25):+.2f},{series.quantile(.75):+.2f}]"
        )


def laterality_audit(df: pd.DataFrame) -> dict:
    """Decide whether +x is the patient's right, using FDI-numbered crossbite labels."""
    sided = df[df["tv_side"].isin(["right", "left"]) & (df["tv_condition"] == "Cross Bite")].copy()
    if len(sided) < 8:
        return {"verdict": "insufficient-evidence", "n": int(len(sided))}

    # Crossbite extent should be larger on the side the FDI numbers point to. Ties and
    # missing measurements carry no evidence either way, so they are excluded rather than
    # silently counted as disagreement.
    right_bias = sided["crossbite_extent_right_deg"] - sided["crossbite_extent_left_deg"]
    as_labelled = np.where(sided["tv_side"] == "right", right_bias, -right_bias)
    informative = as_labelled[np.isfinite(as_labelled) & (as_labelled != 0)]
    if informative.size < 8:
        return {"verdict": "insufficient-evidence", "n_informative": int(informative.size)}

    agree = float((informative > 0).mean())
    return {
        "verdict": "x_is_patient_right" if agree >= 0.5 else "x_is_patient_left",
        "agreement_with_ras": round(agree, 3),
        "n_sided_cases": int(len(sided)),
        "n_informative": int(informative.size),
        "mean_signed_margin_deg": round(float(informative.mean()), 2),
    }


def fit_target(df: pd.DataFrame, cols: list[str], target: str, seed: int = 0) -> dict:
    y = df[target].astype(str)
    keep = y.value_counts()
    y = y.where(y.map(keep) >= 5, other="Other")
    X = df[cols].to_numpy(dtype=float)

    majority = y.value_counts(normalize=True).iloc[0]
    n_splits = min(5, int(y.value_counts().min()))
    if n_splits < 2:
        return {"target": target, "skipped": "too few per class"}

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=3, l2_regularization=1.0, random_state=seed
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X, y, cv=cv)
    return {
        "target": target,
        "n": int(len(y)),
        "majority_baseline": round(float(majority), 3),
        "cv_accuracy": round(float(accuracy_score(y, pred)), 3),
        "cv_balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 3),
        "classes": {k: int(v) for k, v in y.value_counts().items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("artifacts/geom/bits2bites_features.csv"))
    ap.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/bits2bites_v01/Bits2Bites/data/Annotations.csv"),
    )
    ap.add_argument("--output", type=Path, default=Path("artifacts/geom/calibration.json"))
    args = ap.parse_args()

    df = load(args.features, args.annotations)
    cols = feature_columns(df)
    print(f"Loaded {len(df)} cases with {len(cols)} usable measurements")

    print("\n=== Signal check: do the measurements track the labels? ===")
    report_signal(df, "overbite_mm", "Anterior Bite")
    report_signal(df, "overjet_mm", "Anterior Bite")
    report_signal(df, "overjet_mm", "Left Class")
    report_signal(df, "cusp_lag_left_deg", "Left Class")
    report_signal(df, "midline_deviation_mm", "Median Lines")
    df["abs_midline_dev"] = df["midline_deviation_mm"].abs()
    report_signal(df, "abs_midline_dev", "Median Lines")
    df["max_cross_extent"] = df[["crossbite_extent_right_deg", "crossbite_extent_left_deg"]].max(axis=1)
    report_signal(df, "max_cross_extent", "tv_condition")
    df["min_overlap"] = df[["transverse_min_overlap_right_mm", "transverse_min_overlap_left_mm"]].min(axis=1)
    report_signal(df, "min_overlap", "tv_condition")

    print("\n=== Laterality audit (is +x the patient's right?) ===")
    lat = laterality_audit(df)
    print(f"  {json.dumps(lat)}")

    print("\n=== Cross-validated calibration ===")
    results = []
    for target in ["Left Class", "Right Class", "Anterior Bite", "Median Lines"]:
        res = fit_target(df, cols, target)
        results.append(res)
        if "skipped" in res:
            print(f"  {target:16s} skipped: {res['skipped']}")
        else:
            lift = res["cv_accuracy"] - res["majority_baseline"]
            print(
                f"  {target:16s} acc={res['cv_accuracy']:.3f} (majority {res['majority_baseline']:.3f}, "
                f"lift {lift:+.3f})  balanced={res['cv_balanced_accuracy']:.3f}"
            )
    res = fit_target(df.assign(tv=df["tv_condition"]), cols, "tv")
    results.append(res)
    if "skipped" not in res:
        print(
            f"  {'Transversal':16s} acc={res['cv_accuracy']:.3f} (majority {res['majority_baseline']:.3f}, "
            f"lift {res['cv_accuracy'] - res['majority_baseline']:+.3f})  balanced={res['cv_balanced_accuracy']:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"laterality": lat, "targets": results, "n_features": len(cols), "features": cols},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
