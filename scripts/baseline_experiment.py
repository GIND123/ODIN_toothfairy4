"""Establish the achievable score range on real Bite2Text reports.

Three reference points, all measured with the challenge's own captioning metrics:

* **Human ceiling** — score one clinician's report against another clinician's report of the
  same patient. No model can be expected to beat this, and it calibrates everything else.
* **Constant-report floor** — emit a single fixed report for every case. Because the corpus is
  formulaic this is a surprisingly strong baseline, and any real system must clear it.
* **Oracle template** — render the canonical structure from the *parsed ground truth* of the
  case. This isolates how much of the gap is phrasing versus prediction: it is the score a
  model would get if its field predictions were perfect.

Splits are patient-level and seeded, so no case contributes to both sides.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bite2text.eval.gc_metrics import score_captioning  # noqa: E402
from bite2text.report.parse import parse_report  # noqa: E402
from bite2text.report.render import render_report  # noqa: E402


def load_cases(root: Path, family: str) -> dict[str, list[str]]:
    cases: dict[str, list[str]] = {}
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        d = case_dir / family
        if not d.is_dir():
            continue
        texts = [f.read_text(encoding="utf-8", errors="replace").strip() for f in sorted(d.glob("*.txt"))]
        texts = [t for t in texts if t]
        if texts:
            cases[case_dir.name] = texts
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--family", default="reports_ios_en")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", type=Path, default=Path("artifacts/reports/baseline_experiment.json"))
    args = ap.parse_args()

    cases = load_cases(args.root, args.family)
    ids = sorted(cases)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * args.val_frac)
    val_ids, train_ids = ids[:n_val], ids[n_val:]
    print(f"{len(cases)} cases in {args.family}: {len(train_ids)} train / {len(val_ids)} val")

    # Reference = the first report of each case, mirroring one ground-truth file per case.
    val_refs = [cases[c][0] for c in val_ids]
    results: dict[str, dict] = {}

    # --- Human ceiling on the subset with a second independent report ---
    paired = [c for c in val_ids if len(cases[c]) >= 2]
    if paired:
        preds = [cases[c][1] for c in paired]
        refs = [cases[c][0] for c in paired]
        s = score_captioning(preds, refs)
        results["human_ceiling"] = {
            "n": len(paired),
            "bleu_4": round(s.bleu_4, 4),
            "meteor": round(s.meteor, 4),
            "captioning": round(s.captioning, 4),
        }

    # --- Constant report: choose the training report that scores best against training refs ---
    train_refs = [cases[c][0] for c in train_ids]
    pool_ids = train_ids[:400]
    best_text, best_score = None, -1.0
    for cid in pool_ids:
        candidate = cases[cid][0]
        sample = train_refs[:200]
        s = score_captioning([candidate] * len(sample), sample)
        if s.captioning > best_score:
            best_score, best_text = s.captioning, candidate
    s = score_captioning([best_text] * len(val_refs), val_refs)
    results["constant_medoid_report"] = {
        "n": len(val_refs),
        "bleu_4": round(s.bleu_4, 4),
        "meteor": round(s.meteor, 4),
        "captioning": round(s.captioning, 4),
        "text": best_text,
    }

    # --- Oracle template: perfect field values, our phrasing ---
    oracle_preds = [render_report(parse_report(cases[c][0])) for c in val_ids]
    s = score_captioning(oracle_preds, val_refs)
    results["oracle_template"] = {
        "n": len(val_refs),
        "bleu_4": round(s.bleu_4, 4),
        "meteor": round(s.meteor, 4),
        "captioning": round(s.captioning, 4),
        "example": oracle_preds[0],
    }

    # --- Prior template: modal field values everywhere, our phrasing ---
    from bite2text.report.render import render_modal_report

    modal_text = render_modal_report()
    s = score_captioning([modal_text] * len(val_refs), val_refs)
    results["prior_template"] = {
        "n": len(val_refs),
        "bleu_4": round(s.bleu_4, 4),
        "meteor": round(s.meteor, 4),
        "captioning": round(s.captioning, 4),
        "text": modal_text,
    }

    print()
    for name, r in results.items():
        print(f"  {name:24s} BLEU-4={r['bleu_4']:.4f}  METEOR={r['meteor']:.4f}  captioning={r['captioning']:.4f}  (n={r['n']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"family": args.family, "seed": args.seed, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
