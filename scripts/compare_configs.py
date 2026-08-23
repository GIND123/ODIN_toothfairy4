"""Score specific named configurations on one held-out split.

The greedy tuner mixes two kinds of change that behave very differently on the real test set:

* **Phrasing** — saying the same thing the way the corpus usually says it. This should transfer
  to the hidden set nearly in full, because it exploits the shared idiom rather than the
  training subset's particular content.
* **Content** — asserting an extra base-rate finding. Between our two real submissions this
  transferred at only ~13% on BLEU-4 (+0.031 locally became +0.004 actual) while METEOR
  transferred at ~63%.

Reporting them separately shows which part of a local gain is trustworthy.
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
from bite2text.report.dental_health import DentalHealth  # noqa: E402
from bite2text.report.parse import ReportFindings, parse_report  # noqa: E402
from bite2text.report.style import Style, render_styled  # noqa: E402

FIELDS = [
    "overbite", "overjet", "molar_right", "molar_left", "canine_right", "canine_left",
    "midlines", "crossbite", "constriction", "spee", "wilson",
    "crowding_upper", "crowding_lower",
]
MIN_CLASS_SUPPORT = 12
DH_ON = {"restorations": True, "sealants": True, "caries": False,
         "gingival_inflammation": True, "gingival_recession": True, "plaque": True}
DH4 = ("sealants", "caries", "gingival_inflammation", "gingival_recession")

#: name -> (Style, dental-health subset)
CONFIGS: dict[str, tuple[Style, tuple[str, ...]]] = {
    "A shipped (17 Aug)": (Style(), DH4),
    "B +end-to-end": (Style(half_cusp="end-to-end"), DH4),
    "C +crowding is_present": (Style(half_cusp="end-to-end", crowding_form="is_present"), DH4),
    "D +lower curve of Spee": (
        Style(half_cusp="end-to-end", crowding_form="is_present", curves_form="lower_asis"), DH4),
    "E +presents/while sagittal": (
        Style(half_cusp="end-to-end", crowding_form="is_present", curves_form="lower_asis",
              sagittal_form="presents_while"), DH4),
    "F +uncertain gingiva": (
        Style(half_cusp="end-to-end", crowding_form="is_present", curves_form="lower_asis",
              sagittal_form="presents_while", gingiva_form="uncertain"), DH4),
    "G phrasing + midline plain": (
        Style(half_cusp="end-to-end", crowding_form="is_present", curves_form="lower_asis",
              sagittal_form="presents_while", midline_form="plain"), DH4),
    "H tuner pick (content-heavy)": (
        Style(half_cusp="end-to-end", crowding_form="is_present",
              extras=("mixed_dentition", "restoration_teeth", "transverse_normal")),
        ("gingival_inflammation", "gingival_recession", "plaque")),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("artifacts/geom/bite2text_features.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--family", default="reports_intraoral-photo_en")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--output", type=Path, default=Path("artifacts/eval/config_comparison.json"))
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
    test, fit = df.iloc[order[:n_test]], df.iloc[order[n_test:]]
    modal = {f: (str(fit[f].dropna().mode().iloc[0]) if fit[f].notna().any() else None)
             for f in FIELDS}

    preds: dict[str, np.ndarray] = {}
    for field in FIELDS:
        sub = fit[fit[field].notna()]
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

    findings = []
    for i in range(len(test)):
        f = ReportFindings()
        for k, v in modal.items():
            setattr(f, k, v)
        for field, p in preds.items():
            if p[i] not in ("Other", "None", "nan"):
                setattr(f, field, p[i])
        f.crossbite_teeth, f.spacing = [], False
        findings.append(f)
    refs = test["reference"].tolist()

    # Oracle: perfect occlusal field values, same phrasing. Sizes how much is left to gain from
    # better prediction versus better wording.
    oracle_findings = []
    for _, row in test.iterrows():
        f = ReportFindings()
        for field in FIELDS:
            v = row[field]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                setattr(f, field, v)
        f.crossbite_teeth, f.spacing = [], False
        oracle_findings.append(f)

    print(f"held-out n={len(test)}  (field models fitted on the other {len(fit)})\n")
    print(f"{'config':<30}{'BLEU-4':>9}{'METEOR':>9}{'tokens':>8}")
    results = {}
    for name, (style, dh_subset) in CONFIGS.items():
        dh = DentalHealth(**{k: DH_ON[k] for k in dh_subset}) if dh_subset else None
        out = [render_styled(f, dh, style) for f in findings]
        b = st.mean([bleu_4_local([p], [r]) for p, r in zip(out, refs)])
        m = st.mean([meteor_lite_score(tokenize(p), tokenize(r)) for p, r in zip(out, refs)])
        tok = st.median([len(tokenize(p)) for p in out])
        results[name] = {"bleu_4": b, "meteor": m, "tokens": tok}
        print(f"{name:<30}{b:9.4f}{m:9.4f}{tok:8.0f}")

    # Oracle ceiling under the two leading configurations.
    for label, (style, dh_subset) in (("ORACLE fields, config C", CONFIGS["C +crowding is_present"]),
                                      ("ORACLE fields, config H", CONFIGS["H tuner pick (content-heavy)"])):
        dh = DentalHealth(**{k: DH_ON[k] for k in dh_subset}) if dh_subset else None
        out = [render_styled(f, dh, style) for f in oracle_findings]
        b = st.mean([bleu_4_local([p], [r]) for p, r in zip(out, refs)])
        m = st.mean([meteor_lite_score(tokenize(p), tokenize(r)) for p, r in zip(out, refs)])
        results[label] = {"bleu_4": b, "meteor": m}
        print(f"{label:<30}{b:9.4f}{m:9.4f}")

    ref_tok = st.median([len(tokenize(r)) for r in refs])
    print(f"\nreference median tokens = {ref_tok:.0f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
