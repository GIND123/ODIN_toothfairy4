"""Characterise the Bite2Text report corpus.

Everything the generator needs to imitate is measured here rather than assumed: which report
family is the likely evaluation target, how long reports run, which findings appear and at
what base rate, and how much two clinicians describing the same patient actually agree.

That last number matters more than it looks. RadFact scores a prediction against *one*
reference, so the agreement between two independent reports of the same case is the practical
ceiling for any model, and the base rates tell us what an uninformative report should say.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPORT_DIRS = [
    "reports_ios_en",
    "reports_ios_it",
    "reports_intraoral-photo_en",
    "reports_intraoral-photo_it",
]

#: Regexes for the findings the reporting protocol covers. Written against the observed
#: English phrasing; each maps to one template concept.
FINDING_PATTERNS: dict[str, str] = {
    "deep_bite": r"deep bite|overbite is increased|increased overbite",
    "open_bite": r"open bite",
    "reduced_overbite": r"overbite is (?:reduced|decreased)|reduced overbite",
    "normal_overbite": r"overbite is (?:normal|within norm)|normal overbite",
    "increased_overjet": r"overjet is increased|increased overjet",
    "reduced_overjet": r"overjet is (?:reduced|decreased)|reduced overjet",
    "class_i": r"class i\b(?!i)",
    "class_ii": r"class ii\b(?!i)",
    "class_iii": r"class iii",
    "edge_to_edge": r"edge[- ]to[- ]edge|end[- ]to[- ]end|head[- ]to[- ]head",
    "crossbite_present": r"(?<!absence of )(?<!without )(?<!no )(?:posterior |anterior )?cross-?bite",
    "crossbite_absent": r"absence of (?:posterior )?cross-?bite|without cross-?bite|no cross-?bite",
    "scissor_bite": r"scissor bite",
    "constriction": r"constriction|contraction",
    "midline_centered": r"midlines? (?:are|is) (?:centered|coincident)|coincident",
    "midline_deviated": r"midlines? (?:are|is) (?:slightly )?deviated|deviation of the midline",
    "spee_increased": r"curve of spee is (?:increased|accentuated)|accentuated curve of spee",
    "spee_normal": r"curve of spee is (?:normal|within)",
    "wilson": r"curve of wilson",
    "crowding_present": r"crowding",
    "crowding_absent": r"no crowding|absence of crowding|well[- ]aligned|aligned arches",
    "spacing": r"spacing|diastema",
    "missing_tooth": r"missing|agenesis|absent tooth",
    "upper_arch": r"upper arch|maxillary arch",
    "lower_arch": r"lower arch|mandibular arch",
}


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+|[^\w\s]", text.lower()) if t.strip()]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def collect(root: Path) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """case_id -> report_dir -> [(filename, text)]"""
    cases: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        entry: dict[str, list[tuple[str, str]]] = {}
        for rd in REPORT_DIRS:
            d = case_dir / rd
            if not d.is_dir():
                continue
            texts = []
            for f in sorted(d.glob("*.txt")):
                t = f.read_text(encoding="utf-8", errors="replace").strip()
                if t:
                    texts.append((f.name, t))
            if texts:
                entry[rd] = texts
        cases[case_dir.name] = entry
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/reports/corpus_stats.json"))
    args = ap.parse_args()

    cases = collect(args.root)
    print(f"Cases: {len(cases)}")

    print("\n=== Coverage ===")
    coverage = {}
    for rd in REPORT_DIRS:
        have = [c for c, e in cases.items() if rd in e]
        counts = Counter(len(cases[c][rd]) for c in have)
        coverage[rd] = {"cases": len(have), "reports": sum(len(cases[c][rd]) for c in have)}
        print(
            f"  {rd:32s} {len(have):4d} cases  {coverage[rd]['reports']:5d} reports  "
            f"per-case={dict(sorted(counts.items()))}"
        )

    both = sum(1 for e in cases.values() if "reports_ios_en" in e and "reports_intraoral-photo_en" in e)
    only_ios = sum(1 for e in cases.values() if "reports_ios_en" in e and "reports_intraoral-photo_en" not in e)
    only_photo = sum(1 for e in cases.values() if "reports_ios_en" not in e and "reports_intraoral-photo_en" in e)
    neither = sum(1 for e in cases.values() if "reports_ios_en" not in e and "reports_intraoral-photo_en" not in e)
    print(f"\n  both={both}  ios-only={only_ios}  photo-only={only_photo}  neither={neither}")

    print("\n=== Length (English reports) ===")
    lengths = {}
    for rd in ("reports_ios_en", "reports_intraoral-photo_en"):
        toks, sents = [], []
        for e in cases.values():
            for _, t in e.get(rd, []):
                toks.append(len(tokens(t)))
                sents.append(len(sentences(t)))
        if not toks:
            continue
        toks.sort()
        sents.sort()
        lengths[rd] = {
            "n": len(toks),
            "tokens_p10": toks[len(toks) // 10],
            "tokens_median": toks[len(toks) // 2],
            "tokens_p90": toks[len(toks) * 9 // 10],
            "sentences_median": sents[len(sents) // 2],
        }
        print(
            f"  {rd:32s} n={len(toks):4d}  tokens p10/median/p90 = "
            f"{lengths[rd]['tokens_p10']}/{lengths[rd]['tokens_median']}/{lengths[rd]['tokens_p90']}  "
            f"sentences median={lengths[rd]['sentences_median']}"
        )

    print("\n=== Finding base rates (reports_ios_en, per report) ===")
    ios_texts = [t for e in cases.values() for _, t in e.get("reports_ios_en", [])]
    rates = {}
    for name, pat in FINDING_PATTERNS.items():
        rx = re.compile(pat, re.IGNORECASE)
        hits = sum(1 for t in ios_texts if rx.search(t))
        rates[name] = round(hits / max(1, len(ios_texts)), 4)
    for name, rate in sorted(rates.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {rate:6.1%}")

    print("\n=== Inter-report agreement on the same case (reports_ios_en) ===")
    tok_j, find_j = [], []
    for e in cases.values():
        reps = [t for _, t in e.get("reports_ios_en", [])]
        if len(reps) < 2:
            continue
        a, b = reps[0], reps[1]
        tok_j.append(jaccard(set(tokens(a)), set(tokens(b))))
        fa = {n for n, p in FINDING_PATTERNS.items() if re.search(p, a, re.I)}
        fb = {n for n, p in FINDING_PATTERNS.items() if re.search(p, b, re.I)}
        find_j.append(jaccard(fa, fb))
    if tok_j:
        print(f"  pairs={len(tok_j)}  token Jaccard mean={sum(tok_j)/len(tok_j):.3f}")
        print(f"  pairs={len(find_j)}  finding Jaccard mean={sum(find_j)/len(find_j):.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "n_cases": len(cases),
                "coverage": coverage,
                "lengths": lengths,
                "finding_base_rates": rates,
                "agreement": {
                    "n_pairs": len(tok_j),
                    "token_jaccard": round(sum(tok_j) / len(tok_j), 4) if tok_j else None,
                    "finding_jaccard": round(sum(find_j) / len(find_j), 4) if find_j else None,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
