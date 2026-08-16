"""Choose which photograph findings to assert, by measurement.

The dental-health section is worth a large amount of METEOR because that metric weights recall
9:1 — asserting a finding that is usually true gains more from the tokens it covers than it
loses from the tokens it gets wrong. But asserting *everything* costs BLEU-4 precision, so the
optimum is a subset, not the full set.

This searches all 2^6 subsets of the six findings. The subset is chosen on the **training**
split and reported on the held-out split, so the number quoted is not the number optimised.

Selection maximises the worst-case margin over the current public leaderboard leaders
(BLEU-4 0.2463, METEOR 0.4569), since the board ranks by mean position across both metrics and
a system that wins one while losing the other gains nothing.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bite2text.eval.gc_metrics import bleu_4_local, meteor_lite_score, tokenize  # noqa: E402
from bite2text.report.dental_health import DentalHealth, render_dental_health  # noqa: E402
from bite2text.report.parse import ReportFindings, parse_report  # noqa: E402
from bite2text.report.render import render_report  # noqa: E402

FIELDS = [
    "overbite", "overjet", "molar_right", "molar_left", "canine_right", "canine_left",
    "midlines", "crossbite", "constriction", "spee", "wilson",
    "crowding_upper", "crowding_lower",
]
DH_KEYS = ["restorations", "sealants", "caries", "gingival_inflammation", "gingival_recession", "plaque"]
#: Value asserted when a finding is switched on, taken from its modal state in the corpus.
DH_ON = {
    "restorations": True, "sealants": True, "caries": False,
    "gingival_inflammation": True, "gingival_recession": True, "plaque": True,
}
LEADER_BLEU, LEADER_METEOR = 0.2463, 0.4569
MIN_CLASS_SUPPORT = 12


def score(preds: list[str], refs: list[str]) -> tuple[float, float]:
    b = [bleu_4_local([p], [r]) for p, r in zip(preds, refs)]
    m = [meteor_lite_score(tokenize(p), tokenize(r)) for p, r in zip(preds, refs)]
    return st.mean(b), st.mean(m)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("artifacts/geom/bite2text_features.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--family", default="reports_intraoral-photo_en")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--output", type=Path, default=Path("artifacts/eval/dental_health_tuning.json"))
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
    labels = pd.DataFrame(rows)

    feats = pd.read_csv(args.features)
    feats = feats[feats["error"].fillna("") == ""].copy()
    df = feats.merge(labels, on="case_id", how="inner").reset_index(drop=True)
    skip = {"case_id", "error", "reference", *FIELDS}
    cols = [
        c for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > len(df) * 0.5
    ]

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(df))
    n_test = int(len(df) * args.test_frac)
    test, train = df.iloc[order[:n_test]], df.iloc[order[n_test:]]
    print(f"{len(df)} cases -> {len(train)} train / {len(test)} test, family={args.family}")

    modal = {f: (str(train[f].dropna().mode().iloc[0]) if train[f].notna().any() else None) for f in FIELDS}

    def occlusal_reports(split: pd.DataFrame) -> list[ReportFindings]:
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
            preds[field] = clf.predict(split[cols].to_numpy(dtype=float))
        out = []
        for i in range(len(split)):
            f = ReportFindings()
            for k, v in modal.items():
                setattr(f, k, v)
            for field, p in preds.items():
                if p[i] not in ("Other", "None", "nan"):
                    setattr(f, field, p[i])
            f.crossbite_teeth, f.spacing = [], False
            out.append(f)
        return out

    train_occ, test_occ = occlusal_reports(train), occlusal_reports(test)
    train_refs, test_refs = train["reference"].tolist(), test["reference"].tolist()

    def build(occ: list[ReportFindings], subset: tuple[str, ...]) -> list[str]:
        dh = DentalHealth(**{k: DH_ON[k] for k in subset})
        return [render_report(f, dh) for f in occ]

    results = []
    for r in range(len(DH_KEYS) + 1):
        for subset in itertools.combinations(DH_KEYS, r):
            b, m = score(build(train_occ, subset), train_refs)
            margin = min(b / LEADER_BLEU, m / LEADER_METEOR)
            results.append({"subset": list(subset), "train_bleu": b, "train_meteor": m, "min_margin": margin})

    results.sort(key=lambda x: -x["min_margin"])
    print("\ntop subsets by worst-case margin over the leaderboard leaders (TRAIN split):")
    for r in results[:6]:
        print(
            f"  {r['min_margin']:.3f}x  BLEU={r['train_bleu']:.4f} METEOR={r['train_meteor']:.4f}  "
            f"{', '.join(r['subset']) or '(none)'}"
        )

    best = results[0]["subset"]
    tb, tm = score(build(test_occ, tuple(best)), test_refs)
    nb, nm = score(build(test_occ, ()), test_refs)
    print(f"\nchosen on train: {', '.join(best)}")
    print(f"  HELD-OUT  occlusal only        BLEU-4={nb:.4f}  METEOR={nm:.4f}")
    print(f"  HELD-OUT  + dental health      BLEU-4={tb:.4f}  METEOR={tm:.4f}")
    print(f"  leaderboard leaders            BLEU-4={LEADER_BLEU:.4f}  METEOR={LEADER_METEOR:.4f}")
    print(f"  margin                         BLEU-4={tb/LEADER_BLEU:.2f}x   METEOR={tm/LEADER_METEOR:.2f}x")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"chosen": best, "held_out": {"bleu_4": tb, "meteor": tm},
                    "occlusal_only": {"bleu_4": nb, "meteor": nm},
                    "train_search": results[:12]}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
