"""Score the shipped configuration end to end, the way the leaderboard reports it.

Everything is held out: a patient-disjoint split, modal fallbacks and field models fitted on
the training half only, and the dental-health subset fixed in ``bite2text.compose`` rather than
chosen here. Scores are mean +/- std of per-case values, matching the public board.
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

from bite2text.compose import DEFAULT_DENTAL_HEALTH  # noqa: E402
from bite2text.eval.gc_metrics import bleu_4_local, meteor_lite_score, tokenize  # noqa: E402
from bite2text.report.parse import ReportFindings, parse_report  # noqa: E402
from bite2text.report.render import render_report  # noqa: E402

FIELDS = [
    "overbite", "overjet", "molar_right", "molar_left", "canine_right", "canine_left",
    "midlines", "crossbite", "constriction", "spee", "wilson",
    "crowding_upper", "crowding_lower",
]
MIN_CLASS_SUPPORT = 12

#: Public Test Phase leaderboard, 16 Aug 2026.
LEADERBOARD = [
    ("GenMI / teeth occlusion", 0.2463, 0.4261),
    ("MIGG / MMTLVM", 0.2351, 0.4424),
    ("JIA / Bite2Text Report Generation", 0.2290, 0.4234),
    ("DiceMed / previous submission", 0.2639, 0.4215),
    ("shayne / Qwen3-VL Photo", 0.2145, 0.4569),
    ("MIND_lab / Structured Occlusal", 0.2218, 0.4010),
    ("JIA / earlier", 0.2190, 0.4168),
    ("Alex.zhang / Finding-Gated Retrieval", 0.2050, 0.4244),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("artifacts/geom/bite2text_features.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--family", default="reports_intraoral-photo_en")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--output", type=Path, default=Path("artifacts/eval/final_scorecard.json"))
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

    reports, refs = [], test["reference"].tolist()
    for i in range(len(test)):
        f = ReportFindings()
        for k, v in modal.items():
            setattr(f, k, v)
        for field, p in preds.items():
            if p[i] not in ("Other", "None", "nan"):
                setattr(f, field, p[i])
        f.crossbite_teeth, f.spacing = [], False
        reports.append(render_report(f, DEFAULT_DENTAL_HEALTH))

    bleu = [bleu_4_local([p], [r]) for p, r in zip(reports, refs)]
    meteor = [meteor_lite_score(tokenize(p), tokenize(r)) for p, r in zip(reports, refs)]
    b, bs = st.mean(bleu), st.pstdev(bleu)
    m, ms = st.mean(meteor), st.pstdev(meteor)

    entries = LEADERBOARD + [("*** THIS SYSTEM ***", b, m)]
    by_bleu = sorted(entries, key=lambda e: -e[1])
    by_meteor = sorted(entries, key=lambda e: -e[2])
    rank_b = {e[0]: i + 1 for i, e in enumerate(by_bleu)}
    rank_m = {e[0]: i + 1 for i, e in enumerate(by_meteor)}

    print(f"held-out n={len(test)}  (patient-disjoint; models and priors fitted on train only)")
    print(f"  BLEU-4 = {b:.4f} +/- {bs:.4f}")
    print(f"  METEOR = {m:.4f} +/- {ms:.4f}")
    print(f"\n{'entry':<40}{'BLEU-4':>9}{'pos':>5}{'METEOR':>9}{'pos':>5}{'mean':>7}")
    for name, eb, em in sorted(entries, key=lambda e: (rank_b[e[0]] + rank_m[e[0]]) / 2):
        print(f"{name:<40}{eb:9.4f}{rank_b[name]:5d}{em:9.4f}{rank_m[name]:5d}"
              f"{(rank_b[name] + rank_m[name]) / 2:7.1f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "n_test": len(test), "bleu_4": b, "bleu_4_std": bs, "meteor": m, "meteor_std": ms,
        "mean_position": (rank_b["*** THIS SYSTEM ***"] + rank_m["*** THIS SYSTEM ***"]) / 2,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
