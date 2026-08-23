"""Fuse the geometry and photograph predictors, per field, and score the result end to end.

The two signals are genuinely independent — one measures the arch surfaces, the other looks at
the pictures — and they are strong on different fields. Geometry wins overjet (0.73 vs 0.69);
photographs win canine_right (0.63 vs 0.53) and midlines (0.57 vs 0.50), which geometry cannot
see well because the dental midline is a soft-tissue-referenced landmark.

Fusion weight is chosen **per field on the validation split** and applied to the held-out test
split, so the reported numbers are not the ones optimised. A field keeps whichever source (or
blend) validated best, which means fusion can never do worse than the better single source
except through validation noise.
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
WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]  # weight on geometry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("artifacts/geom/bite2text_features.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--photo", type=Path, default=Path("artifacts/eval/photo_fields.json"))
    ap.add_argument("--splits", type=Path, default=Path("artifacts/eval/photo_splits.json"))
    ap.add_argument("--family", default="reports_intraoral-photo_en")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", type=Path, default=Path("artifacts/eval/fusion.json"))
    args = ap.parse_args()

    photo = json.loads(args.photo.read_text(encoding="utf-8"))
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    vocab: dict[str, list[str]] = photo["vocab"]

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
    df = feats.merge(pd.DataFrame(rows), on="case_id", how="inner").set_index("case_id")
    cols = [
        c for c in df.columns
        if c not in {"error", "reference", *FIELDS}
        and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > len(df) * 0.5
    ]

    fit_ids = [c for c in splits["fit"] if c in df.index]
    val_ids = [c for c in splits["val"] if c in df.index and c in photo["probabilities"]["val"]]
    test_ids = [c for c in splits["test"] if c in df.index and c in photo["probabilities"]["test"]]
    fit = df.loc[fit_ids]
    print(f"fit={len(fit_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    # --- geometry probabilities over the same class vocabulary as the photo model ---
    geom_prob: dict[str, dict[str, np.ndarray]] = {"val": {}, "test": {}}
    geom_acc: dict[str, float] = {}
    for field in FIELDS:
        classes = vocab.get(field)
        if not classes:
            continue
        sub = fit[fit[field].notna()].copy()
        y = sub[field].astype(str)
        counts = y.value_counts()
        y = y.where(y.map(counts) >= MIN_CLASS_SUPPORT, other="Other")
        y = y.where(y.isin(classes), other="Other")
        if y.nunique() < 2:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=3,
            min_samples_leaf=15, l2_regularization=1.0, random_state=args.seed,
        )
        clf.fit(sub[cols].to_numpy(dtype=float), y)
        order = {c: i for i, c in enumerate(clf.classes_)}
        for name, ids in (("val", val_ids), ("test", test_ids)):
            raw = clf.predict_proba(df.loc[ids, cols].to_numpy(dtype=float))
            aligned = np.zeros((len(ids), len(classes)))
            for j, cls in enumerate(classes):
                if cls in order:
                    aligned[:, j] = raw[:, order[cls]]
            total = aligned.sum(axis=1, keepdims=True)
            aligned = np.divide(aligned, total, out=np.full_like(aligned, 1 / len(classes)),
                                where=total > 0)
            geom_prob[name][field] = aligned

    def truth(ids: list[str], field: str) -> np.ndarray:
        return df.loc[ids, field].astype("object").to_numpy()

    def accuracy(prob: np.ndarray, ids: list[str], field: str) -> float:
        classes = vocab[field]
        pred = np.array([classes[i] for i in prob.argmax(1)])
        actual = truth(ids, field)
        mask = np.array([a is not None and not (isinstance(a, float) and np.isnan(a)) for a in actual])
        if not mask.any():
            return float("nan")
        return float((pred[mask] == np.array([str(a) for a in actual])[mask]).mean())

    # --- choose the fusion weight per field on validation ---
    chosen: dict[str, float] = {}
    print(f"\n{'field':<18}{'geom':>7}{'photo':>7}{'fused':>7}{'w_geom':>8}")
    summary = {}
    for field in FIELDS:
        if field not in geom_prob["val"]:
            continue
        classes = vocab[field]
        pv = np.array([photo["probabilities"]["val"][c][field] for c in val_ids])
        gv = geom_prob["val"][field]
        best_w, best_acc = 1.0, -1.0
        for w in WEIGHTS:
            acc = accuracy(w * gv + (1 - w) * pv, val_ids, field)
            if acc > best_acc:
                best_acc, best_w = acc, w
        chosen[field] = best_w

        pt = np.array([photo["probabilities"]["test"][c][field] for c in test_ids])
        gt = geom_prob["test"][field]
        g_acc = accuracy(gt, test_ids, field)
        p_acc = accuracy(pt, test_ids, field)
        f_acc = accuracy(best_w * gt + (1 - best_w) * pt, test_ids, field)
        geom_acc[field] = g_acc
        summary[field] = {"geometry": g_acc, "photo": p_acc, "fused": f_acc, "w_geometry": best_w}
        print(f"{field:<18}{g_acc:7.3f}{p_acc:7.3f}{f_acc:7.3f}{best_w:8.1f}")

    mean_g = st.mean(v["geometry"] for v in summary.values())
    mean_p = st.mean(v["photo"] for v in summary.values())
    mean_f = st.mean(v["fused"] for v in summary.values())
    print(f"{'MEAN':<18}{mean_g:7.3f}{mean_p:7.3f}{mean_f:7.3f}")

    # --- end-to-end report scores on the test split ---
    modal = {f: (str(fit[f].dropna().mode().iloc[0]) if fit[f].notna().any() else None)
             for f in FIELDS}
    refs = df.loc[test_ids, "reference"].tolist()
    style = Style(half_cusp="end-to-end", crowding_form="is_present")
    dh = DentalHealth(**{k: DH_ON[k] for k in
                         ("sealants", "caries", "gingival_inflammation", "gingival_recession")})

    def build(source: str) -> list[str]:
        out = []
        for i, case_id in enumerate(test_ids):
            f = ReportFindings()
            for k, v in modal.items():
                setattr(f, k, v)
            for field in FIELDS:
                if field not in geom_prob["test"]:
                    continue
                classes = vocab[field]
                gt = geom_prob["test"][field][i]
                pt = np.array(photo["probabilities"]["test"][case_id][field])
                w = {"geometry": 1.0, "photo": 0.0}.get(source, chosen[field])
                value = classes[int((w * gt + (1 - w) * pt).argmax())]
                if value not in ("Other", "None", "nan"):
                    setattr(f, field, value)
            f.crossbite_teeth, f.spacing = [], False
            out.append(render_styled(f, dh, style))
        return out

    print(f"\nend-to-end on the held-out test split (n={len(test_ids)}):")
    scores = {}
    for source in ("geometry", "photo", "fused"):
        preds = build(source)
        b = st.mean([bleu_4_local([p], [r]) for p, r in zip(preds, refs)])
        m = st.mean([meteor_lite_score(tokenize(p), tokenize(r)) for p, r in zip(preds, refs)])
        scores[source] = {"bleu_4": b, "meteor": m}
        print(f"  {source:<10} BLEU-4={b:.4f}  METEOR={m:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"per_field": summary, "weights": chosen, "end_to_end": scores}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
