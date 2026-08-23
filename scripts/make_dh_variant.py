"""Emit a report variant with a chosen subset of the dental-health sentences.

The full six-sentence section maximises captioning but asserts six base-rate claims, each of
which RadFact charges against precision. This lets a shorter subset be measured on the clinical
metric so the trade can be made on evidence rather than intuition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bite2text.report.dental_health import DentalHealth, render_dental_health  # noqa: E402

#: Value asserted when a finding is switched on, from its modal state in the corpus.
ON = {
    "restorations": True, "sealants": True, "caries": False,
    "gingival_inflammation": True, "gingival_recession": True, "plaque": True,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--occlusal", type=Path, default=Path("artifacts/eval/pred_photo_occ.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--fields", nargs="+", default=["sealants", "caries", "gingival_inflammation", "gingival_recession"],
        help="which findings to assert",
    )
    args = ap.parse_args()

    unknown = [f for f in args.fields if f not in ON]
    if unknown:
        raise SystemExit(f"unknown findings: {unknown}; choose from {sorted(ON)}")

    occlusal = json.loads(args.occlusal.read_text(encoding="utf-8"))
    tail = " " + " ".join(render_dental_health(DentalHealth(**{k: ON[k] for k in args.fields})))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({k: v + tail for k, v in occlusal.items()}, indent=1), encoding="utf-8"
    )
    print(f"{len(occlusal)} cases -> {args.output}")
    print(f"appended: {tail.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
