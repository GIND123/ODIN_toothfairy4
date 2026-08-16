"""Decide, per finding, whether the vision model beats its base rate.

The dental-health section currently asserts fixed modal values. A vision-language model can
only justify replacing one of those if it is *more often right than the base rate* on the cases
where the reference actually states the finding — the same bar the geometry fields had to clear.

Accuracy alone is not the deciding number, though. METEOR weights recall 9:1, so a finding is
worth asserting even when it is only usually true; what a per-case prediction buys is BLEU
precision on the cases where the modal answer is wrong. Both are therefore reported, and the
end-to-end effect is measured directly by rendering reports both ways.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bite2text.eval.gc_metrics import bleu_4_local, meteor_lite_score, tokenize  # noqa: E402
from bite2text.report.dental_health import (  # noqa: E402
    DentalHealth, parse_dental_health, render_dental_health,
)
from bite2text.report.parse import parse_report  # noqa: E402
from bite2text.report.render import render_report  # noqa: E402

KEYS = ["gingival_inflammation", "gingival_recession", "caries", "restorations", "sealants", "plaque"]
#: What the shipped system asserts today, with no photo input.
PRIOR = {"restorations": True, "sealants": True, "caries": False,
         "gingival_inflammation": True, "gingival_recession": True, "plaque": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, default=Path("artifacts/eval/photo_findings.json"))
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--family", default="reports_intraoral-photo_en")
    ap.add_argument("--output", type=Path, default=Path("artifacts/eval/photo_findings_report.json"))
    args = ap.parse_args()

    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    refs: dict[str, str] = {}
    truth: dict[str, DentalHealth] = {}
    for case_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        d = case_dir / args.family
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.txt"))
        if not files:
            continue
        text = files[0].read_text(encoding="utf-8", errors="replace").strip()
        refs[case_dir.name] = text
        truth[case_dir.name] = parse_dental_health(text)

    common = sorted(set(preds) & set(refs))
    print(f"{len(common)} cases with both a prediction and a reference\n")

    summary = {}
    for key in KEYS:
        pairs = [
            (bool(preds[c].get(key)), getattr(truth[c], key))
            for c in common
            if key in preds[c] and getattr(truth[c], key) is not None
        ]
        if len(pairs) < 30:
            summary[key] = {"status": "too-few-stated", "n": len(pairs)}
            print(f"  {key:22s} only {len(pairs)} cases state it — keeping prior")
            continue
        n = len(pairs)
        model_acc = sum(p == t for p, t in pairs) / n
        prior_acc = sum(PRIOR[key] == t for _, t in pairs) / n
        positive_rate = sum(t for _, t in pairs) / n
        use = model_acc > prior_acc + 0.02
        summary[key] = {
            "status": "ok", "n": n, "model_accuracy": round(model_acc, 4),
            "prior_accuracy": round(prior_acc, 4), "reference_positive_rate": round(positive_rate, 4),
            "use_model": use,
        }
        print(
            f"  {key:22s} n={n:4d}  model={model_acc:.3f}  prior={prior_acc:.3f}  "
            f"(ref positive {positive_rate:.0%})   {'USE MODEL' if use else 'use prior'}"
        )

    # End-to-end: render with prior vs with model predictions, on the same cases.
    def score(build) -> tuple[float, float]:
        b, m = [], []
        for c in common:
            text = build(c)
            b.append(bleu_4_local([text], [refs[c]]))
            m.append(meteor_lite_score(tokenize(text), tokenize(refs[c])))
        return st.mean(b), st.mean(m)

    occ = {c: parse_report(refs[c]) for c in common}  # oracle occlusal, to isolate the DH effect
    use_model = {k for k, v in summary.items() if v.get("use_model")}

    prior_b, prior_m = score(lambda c: render_report(occ[c], DentalHealth(**PRIOR)))
    hybrid_b, hybrid_m = score(
        lambda c: render_report(
            occ[c],
            DentalHealth(**{k: (bool(preds[c].get(k)) if k in use_model and k in preds[c] else PRIOR[k])
                            for k in KEYS}),
        )
    )
    allmodel_b, allmodel_m = score(
        lambda c: render_report(
            occ[c],
            DentalHealth(**{k: (bool(preds[c][k]) if k in preds[c] else PRIOR[k]) for k in KEYS}),
        )
    )

    print("\n  (oracle occlusal fields, so the difference is only the dental-health section)")
    print(f"  prior dental health        BLEU-4={prior_b:.4f}  METEOR={prior_m:.4f}")
    print(f"  hybrid (model where it won) BLEU-4={hybrid_b:.4f}  METEOR={hybrid_m:.4f}")
    print(f"  all model predictions      BLEU-4={allmodel_b:.4f}  METEOR={allmodel_m:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"n_cases": len(common), "fields": summary, "use_model": sorted(use_model),
                    "end_to_end": {"prior": {"bleu_4": prior_b, "meteor": prior_m},
                                   "hybrid": {"bleu_4": hybrid_b, "meteor": hybrid_m},
                                   "all_model": {"bleu_4": allmodel_b, "meteor": allmodel_m}}}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
