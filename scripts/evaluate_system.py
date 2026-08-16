"""End-to-end evaluation of the composed report, honestly.

Field predictions are produced **out of fold** with ``cross_val_predict``, so no case is
scored by a model that saw its own label. Reports are then rendered from those out-of-fold
predictions and scored with the challenge's own captioning metrics.

It also writes ``predictions.json`` / ``references.json`` for the Modal RadFact judge, which
supplies the clinical 0.8 of the final score.

Ablations reported alongside the full system:

* ``prior``   — modal value for every field (no geometry at all)
* ``geometry``— the shipped configuration
* ``oracle``  — parsed ground-truth values, i.e. the ceiling for this renderer
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bite2text.eval.gc_metrics import score_captioning  # noqa: E402
from bite2text.report.parse import parse_report  # noqa: E402
from bite2text.report.render import MODAL_FINDINGS, render_report  # noqa: E402
from bite2text.report.parse import ReportFindings  # noqa: E402

FIELDS = [
    "overbite", "overjet", "molar_right", "molar_left", "canine_right", "canine_left",
    "midlines", "crossbite", "constriction", "spee", "wilson",
    "crowding_upper", "crowding_lower",
]
MIN_CLASS_SUPPORT = 12


def load_labels_and_refs(root: Path, family: str) -> pd.DataFrame:
    rows = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        d = case_dir / family
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
        row["crossbite_teeth"] = parsed.crossbite_teeth
        row["crossbite_side"] = parsed.crossbite_side
        row["spacing"] = parsed.spacing
        rows.append(row)
    return pd.DataFrame(rows)


def oof_predictions(df: pd.DataFrame, cols: list[str], seed: int) -> dict[str, pd.Series]:
    """Out-of-fold predictions per field, indexed like ``df``."""
    out: dict[str, pd.Series] = {}
    X_all = df[cols].to_numpy(dtype=float)
    for field in FIELDS:
        mask = df[field].notna()
        sub = df[mask]
        if len(sub) < 60:
            continue
        y = sub[field].astype(str)
        counts = y.value_counts()
        y = y.where(y.map(counts) >= MIN_CLASS_SUPPORT, other="Other")
        counts = y.value_counts()
        if len(counts) < 2 or counts.min() < 2:
            continue
        n_splits = int(min(5, counts.min()))
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=3,
            min_samples_leaf=15, l2_regularization=1.0, random_state=seed,
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred = cross_val_predict(clf, X_all[mask.to_numpy()], y, cv=cv)
        series = pd.Series(index=df.index, dtype=object)
        series[mask] = pred
        out[field] = series
    return out


def findings_from(values: dict[str, object]) -> ReportFindings:
    f = ReportFindings(**{k: v for k, v in MODAL_FINDINGS.__dict__.items()})
    for key, value in values.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if value in ("Other", "None", "nan"):
            continue
        setattr(f, key, value)
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("artifacts/geom/bite2text_features.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--family", default="reports_ios_en")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/eval"))
    args = ap.parse_args()

    feats = pd.read_csv(args.features)
    feats = feats[feats["error"].fillna("") == ""].copy()
    labels = load_labels_and_refs(args.root, args.family)
    df = feats.merge(labels, on="case_id", how="inner").reset_index(drop=True)
    skip = {"case_id", "error", "reference", "crossbite_teeth", "crossbite_side", "spacing", *FIELDS}
    cols = [
        c for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > len(df) * 0.5
    ]
    print(f"{len(df)} cases, {len(cols)} features, family={args.family}")

    oof = oof_predictions(df, cols, args.seed)
    print(f"out-of-fold predictions for {len(oof)}/{len(FIELDS)} fields")

    references = df["reference"].tolist()
    variants: dict[str, list[str]] = {}

    # prior: modal everything
    variants["prior"] = [render_report(findings_from({})) for _ in range(len(df))]

    # geometry: out-of-fold model predictions where available
    geom_reports = []
    for i in range(len(df)):
        values = {f: (oof[f].iloc[i] if f in oof else None) for f in FIELDS}
        f = findings_from(values)
        # Neither individual tooth identity nor interdental spacing is recoverable from the
        # arch measurements, so the shipped system stays silent on both.
        f.crossbite_teeth = []
        f.spacing = False
        geom_reports.append(render_report(f))
    variants["geometry"] = geom_reports

    # selective: geometry only for fields whose out-of-fold accuracy clears a bar, prior
    # elsewhere. RadFact precision is charged per claim, so a barely-better-than-chance field
    # can cost more in wrong claims than it earns in coverage.
    accuracy = {}
    for field, series in oof.items():
        mask = df[field].notna()
        accuracy[field] = float((series[mask].astype(str) == df[field][mask].astype(str)).mean())
    for threshold in (0.60, 0.65, 0.70):
        strong = {f for f, a in accuracy.items() if a >= threshold}
        reports = []
        for i in range(len(df)):
            values = {f: (oof[f].iloc[i] if f in strong else None) for f in FIELDS}
            f = findings_from(values)
            f.crossbite_teeth = []
            f.spacing = False
            reports.append(render_report(f))
        variants[f"selective@{threshold:.2f}"] = reports
    print("  field OOF accuracy: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(accuracy.items())))

    # oracle: parsed ground-truth values in our phrasing
    oracle_reports = []
    for i in range(len(df)):
        values = {f: df[f].iloc[i] for f in FIELDS}
        f = findings_from(values)
        f.crossbite_teeth = list(df["crossbite_teeth"].iloc[i] or [])
        f.crossbite_side = df["crossbite_side"].iloc[i]
        f.spacing = bool(df["spacing"].iloc[i])
        oracle_reports.append(render_report(f))
    variants["oracle"] = oracle_reports

    print()
    summary = {}
    for name, preds in variants.items():
        s = score_captioning(preds, references)
        summary[name] = {"bleu_4": round(s.bleu_4, 4), "meteor": round(s.meteor, 4),
                         "captioning": round(s.captioning, 4)}
        print(f"  {name:10s} BLEU-4={s.bleu_4:.4f}  METEOR={s.meteor:.4f}  captioning={s.captioning:.4f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    case_ids = df["case_id"].tolist()
    (args.outdir / "references.json").write_text(
        json.dumps(dict(zip(case_ids, references)), indent=1), encoding="utf-8")
    for name, preds in variants.items():
        (args.outdir / f"predictions_{name}.json").write_text(
            json.dumps(dict(zip(case_ids, preds)), indent=1), encoding="utf-8")
    (args.outdir / "captioning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote predictions/references to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
