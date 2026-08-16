"""Batch-extract geometric measurements from an arch-pair dataset.

Works on both layouts we care about:

* Bits2Bites  ``<root>/data/{train,val}/<case>/{upper,lower}.stl``
* Bite2Text   ``<root>/<case>/ios/ios_{upper,lower}.stl``

Writes one CSV row per case. Failures are recorded rather than raised so a single bad mesh
cannot abort a 1,000-case run.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bite2text.geom import CaseGeometry, measure_case  # noqa: E402


def discover_cases(root: Path) -> list[tuple[str, Path, Path]]:
    """Find (case_id, upper_path, lower_path) triples under ``root``."""
    cases: list[tuple[str, Path, Path]] = []

    # Bits2Bites layout
    for split in ("train", "val"):
        split_dir = root / "data" / split
        if split_dir.is_dir():
            for case_dir in sorted(split_dir.iterdir()):
                up, lo = case_dir / "upper.stl", case_dir / "lower.stl"
                if up.exists() and lo.exists():
                    cases.append((f"{split}/{case_dir.name}", up, lo))
    if cases:
        return cases

    # Bite2Text layout
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ios = case_dir / "ios"
        for up_name, lo_name in (("ios_upper.stl", "ios_lower.stl"), ("upper.stl", "lower.stl")):
            up, lo = ios / up_name, ios / lo_name
            if up.exists() and lo.exists():
                cases.append((case_dir.name, up, lo))
                break
        else:
            for up_name, lo_name in (("ios_upper.obj", "ios_lower.obj"),):
                up, lo = ios / up_name, ios / lo_name
                if up.exists() and lo.exists():
                    cases.append((case_dir.name, up, lo))
                    break
    return cases


def process(job: tuple[str, Path, Path]) -> dict:
    case_id, up, lo = job
    try:
        geom = CaseGeometry.from_meshes(case_id, up, lo)
        row = measure_case(geom).features()
        row["case_id"] = case_id
        row["error"] = ""
        return row
    except Exception as exc:  # noqa: BLE001 - one bad mesh must not kill the batch
        return {"case_id": case_id, "error": f"{type(exc).__name__}: {exc}", "_tb": traceback.format_exc()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = discover_cases(args.root)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(f"No arch pairs found under {args.root}", file=sys.stderr)
        return 1
    print(f"Found {len(cases)} cases; extracting with {args.workers} workers")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, job): job[0] for job in cases}
        for done, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if done % 25 == 0 or done == len(cases):
                print(f"  {done}/{len(cases)}", flush=True)

    frame = pd.DataFrame(rows).sort_values("case_id")
    failed = frame[frame["error"] != ""]
    if len(failed):
        print(f"\n{len(failed)} case(s) failed:")
        for _, r in failed.head(5).iterrows():
            print(f"  {r['case_id']}: {r['error']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=[c for c in frame.columns if c == "_tb"]).to_csv(args.output, index=False)
    print(f"\nWrote {len(frame)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
