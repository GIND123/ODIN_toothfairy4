"""Strict held-out validation: no test case informs any part of the system.

The out-of-fold evaluation in ``evaluate_system.py`` is honest about the *field models*, but
two other things were chosen while looking at the whole corpus: the modal fallback values, and
the renderer's phrasing. Both are cheap to leak through, and the headline numbers are high
enough relative to the public leaderboard that they deserve a clean test.

Here the corpus is split by patient once. Modal values are recomputed from the training split
alone, field models are fitted on the training split alone, and nothing from the test split
touches either. The phrasing templates are fixed code and cannot be refit, so the residual
exposure is the earlier A/B choice between two hard-coded variants — reported alongside.

Scores are reported the way the leaderboard reports them: **mean ± std of per-case values**.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bite2text.eval.gc_metrics import bleu_4_local, meteor_lite_score, tokenize  # noqa: E402
from bite2text.report.parse import ReportFindings, parse_report  # noqa: E402
from bite2text.report.render import render_report  # noqa: E402

FIELDS = [
    "overbite", "overjet", "molar_right", "molar_left", "canine_right", "canine_left",
    "midlines", "crossbite", "constriction", "spee", "wilson",
    "crowding_upper", "crowding_lower",
]
MIN_CLASS_SUPPORT = 12


def per_case_scores(preds: list[str], refs: list[str]) -> dict[str, float]:
    b = [bleu_4_local([p], [r]) for p, r in zip(preds, refs)]
    m = [meteor_lite_score(tokenize(p), tokenize(r)) for p, r in zip(preds, refs)]
    return {
        "bleu_4": st.mean(b), "bleu_4_std": st.pstdev(b),
        "meteor": st.mean(m), "meteor_std": st.pstdev(m),
        "captioning": 0.5 * (st.mean(b) + st.mean(m)),
    }


def load(root: Path, family: str) -> pd.DataFrame:
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
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("artifacts/geom/bite2text_features.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--family", default="reports_ios_en")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--output", type=Path, default=Path("artifacts/eval/holdout.json"))
    args = ap.parse_args()

    feats = pd.read_csv(args.features)
    feats = feats[feats["error"].fillna("") == ""].copy()
    df = feats.merge(load(args.root, args.family), on="case_id", how="inner").reset_index(drop=True)

    skip = {"case_id", "error", "reference", *FIELDS}
    cols = [
        c for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > len(df) * 0.5
    ]

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(df))
    n_test = int(len(df) * args.test_frac)
    test_idx, train_idx = order[:n_test], order[n_test:]
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    print(f"{len(df)} cases -> {len(train)} train / {len(test)} test (patient-disjoint)")

    # --- Modal values from TRAIN ONLY ---
    modal = {}
    for f in FIELDS:
        vals = train[f].dropna()
        modal[f] = str(vals.mode().iloc[0]) if len(vals) else None
    print("  train-derived modal values: " + ", ".join(f"{k}={v}" for k, v in modal.items()))

    def base_findings() -> ReportFindings:
        out = ReportFindings()
        for k, v in modal.items():
            setattr(out, k, v)
        return out

    # --- Field models fitted on TRAIN ONLY ---
    predictions: dict[str, np.ndarray] = {}
    accuracies: dict[str, float] = {}
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
        pred = clf.predict(test[cols].to_numpy(dtype=float))
        predictions[field] = pred
        mask = test[field].notna().to_numpy()
        if mask.sum():
            accuracies[field] = float((pred[mask] == test[field][mask].astype(str).to_numpy()).mean())

    refs = test["reference"].tolist()
    results = {}

    prior_text = render_report(base_findings())
    results["prior"] = per_case_scores([prior_text] * len(test), refs)

    geom_reports = []
    for i in range(len(test)):
        f = base_findings()
        for field, pred in predictions.items():
            value = pred[i]
            if value not in ("Other", "None", "nan"):
                setattr(f, field, value)
        f.crossbite_teeth, f.spacing = [], False
        geom_reports.append(render_report(f))
    results["geometry"] = per_case_scores(geom_reports, refs)

    oracle_reports = []
    for _, row in test.iterrows():
        f = base_findings()
        for field in FIELDS:
            if row[field] is not None and not (isinstance(row[field], float) and np.isnan(row[field])):
                setattr(f, field, row[field])
        f.crossbite_teeth, f.spacing = [], False
        oracle_reports.append(render_report(f))
    results["oracle"] = per_case_scores(oracle_reports, refs)

    # Human reference point on the same test cases that have a second report.
    pairs = []
    for case_id in test["case_id"]:
        files = sorted((args.root / case_id / args.family).glob("*.txt"))
        if len(files) >= 2:
            pairs.append((
                files[1].read_text(encoding="utf-8", errors="replace").strip(),
                files[0].read_text(encoding="utf-8", errors="replace").strip(),
            ))
    if pairs:
        results["human_second_reader"] = per_case_scores([a for a, _ in pairs], [b for _, b in pairs])
        results["human_second_reader"]["n"] = len(pairs)

    print("\n  (mean +/- std of per-case scores, as the leaderboard reports)")
    for name, r in results.items():
        print(
            f"  {name:20s} BLEU-4={r['bleu_4']:.4f} +/- {r['bleu_4_std']:.4f}   "
            f"METEOR={r['meteor']:.4f} +/- {r['meteor_std']:.4f}"
        )
    print("\n  held-out field accuracy: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(accuracies.items())))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"n_train": len(train), "n_test": len(test), "modal": modal,
                    "field_accuracy": accuracies, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
