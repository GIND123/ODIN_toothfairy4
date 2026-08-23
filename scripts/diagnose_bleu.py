"""Break BLEU-4 into its parts to see what actually limits it.

BLEU-4 is the geometric mean of modified 1..4-gram precisions times a brevity penalty. Because
it is a *geometric* mean, the weakest order dominates: a report with excellent unigram overlap
and poor 4-gram overlap scores badly, and no amount of extra vocabulary fixes it. This prints
the per-order precisions and the brevity penalty so the fix targets the right thing, then lists
the reference 4-grams we most often fail to emit.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bite2text.compose import DEFAULT_DENTAL_HEALTH  # noqa: E402
from bite2text.eval.gc_metrics import tokenize  # noqa: E402
from bite2text.report.parse import parse_report  # noqa: E402
from bite2text.report.style import Style, render_styled  # noqa: E402


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--family", default="reports_intraoral-photo_en")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    refs = []
    for case_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        d = case_dir / args.family
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.txt"))
        if files:
            refs.append(files[0].read_text(encoding="utf-8", errors="replace").strip())

    style = Style(half_cusp="end-to-end", crowding_form="is_present",
                  extras=("restoration_teeth", "transverse_normal"))
    preds = [render_styled(parse_report(r), DEFAULT_DENTAL_HEALTH, style) for r in refs]

    print(f"n={len(refs)}")
    print(f"ref median tokens={st.median([len(tokenize(r)) for r in refs]):.0f}  "
          f"pred median={st.median([len(tokenize(p)) for p in preds]):.0f}")

    # Corpus-level modified n-gram precision, the quantity BLEU actually averages.
    print("\nmodified n-gram precision (higher is better; geometric mean drives BLEU-4):")
    for n in (1, 2, 3, 4):
        clipped = total = 0
        for p, r in zip(preds, refs):
            pn, rn = ngrams(tokenize(p), n), ngrams(tokenize(r), n)
            total += sum(pn.values())
            clipped += sum(min(c, rn.get(g, 0)) for g, c in pn.items())
        print(f"  {n}-gram  {clipped / max(1, total):.4f}   ({clipped}/{total})")

    pred_len = sum(len(tokenize(p)) for p in preds)
    ref_len = sum(len(tokenize(r)) for r in refs)
    ratio = pred_len / ref_len
    print(f"\nlength ratio pred/ref = {ratio:.3f}  "
          f"(brevity penalty applies only when < 1: {'ACTIVE' if ratio < 1 else 'none'})")

    print(f"\ntop {args.top} reference 4-grams we never emit:")
    missing: Counter = Counter()
    for p, r in zip(preds, refs):
        pn, rn = ngrams(tokenize(p), 4), ngrams(tokenize(r), 4)
        for g, c in rn.items():
            if pn.get(g, 0) < c:
                missing[g] += c - pn.get(g, 0)
    for gram, n in missing.most_common(args.top):
        print(f"  {n:5d}  {' '.join(gram)}")

    print(f"\ntop 15 of our 4-grams that references never contain (pure precision loss):")
    wasted: Counter = Counter()
    for p, r in zip(preds, refs):
        pn, rn = ngrams(tokenize(p), 4), ngrams(tokenize(r), 4)
        for g, c in pn.items():
            if rn.get(g, 0) < c:
                wasted[g] += c - rn.get(g, 0)
    for gram, n in wasted.most_common(15):
        print(f"  {n:5d}  {' '.join(gram)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
