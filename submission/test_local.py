"""Exercise the submission entrypoint against a simulated Grand Challenge input layout.

Builds the exact directory shape the platform mounts at ``/input`` — socket sub-directories,
``inputs.json``, one arch mesh each — runs ``inference.py`` against it, and checks that a
well-formed ``diagnostic-imaging-report.json`` comes out.

Deliberately includes adversarial cases, because the evaluator scores a missing output as an
empty report worth zero:

* a case whose upper mesh is empty (the real F5500 failure in the training set)
* a case with no meshes at all
* a case using the flat ``/input/3d-upper-teeth-scan.obj`` layout instead of sub-directories
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def write_inputs_json(case_dir: Path, lower_name: str | None, upper_name: str | None) -> None:
    entries = []
    if lower_name:
        entries.append({
            "socket": {"slug": "3d-lower-teeth-scan", "relative_path": "files/ios-lower",
                       "is_file_kind": True, "is_image_kind": False},
            "file": {"name": lower_name}, "image": None, "value": None,
        })
    if upper_name:
        entries.append({
            "socket": {"slug": "3d-upper-teeth-scan", "relative_path": "files/ios-upper",
                       "is_file_kind": True, "is_image_kind": False},
            "file": {"name": upper_name}, "image": None, "value": None,
        })
    entries.append({
        "socket": {"slug": "2d-intraoral-photographs", "relative_path": "images/intraoral-photo",
                   "is_file_kind": False, "is_image_kind": True},
        "file": None, "image": {"name": "intraoral-photo.tiff"}, "value": None,
    })
    (case_dir / "inputs.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def build_case(tmp: Path, name: str, upper: Path | None, lower: Path | None, flat: bool = False) -> Path:
    case = tmp / name
    (case / "images" / "intraoral-photo").mkdir(parents=True, exist_ok=True)
    if flat:
        if upper:
            shutil.copy2(upper, case / "3d-upper-teeth-scan.stl")
        if lower:
            shutil.copy2(lower, case / "3d-lower-teeth-scan.stl")
    else:
        if upper:
            (case / "files" / "ios-upper").mkdir(parents=True, exist_ok=True)
            shutil.copy2(upper, case / "files" / "ios-upper" / "ios_upper.stl")
        if lower:
            (case / "files" / "ios-lower").mkdir(parents=True, exist_ok=True)
            shutil.copy2(lower, case / "files" / "ios-lower" / "ios_lower.stl")
    write_inputs_json(case, "ios_lower.stl" if lower else None, "ios_upper.stl" if upper else None)
    return case


def run_case(case_dir: Path, out_dir: Path) -> tuple[int, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env_patch = {
        "PYTHONPATH": str(REPO / "src"),
        "B2T_INPUT": str(case_dir),
        "B2T_OUTPUT": str(out_dir),
    }
    import os

    env = {**os.environ, **env_patch}
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "inference.py")],
        capture_output=True, text=True, env=env,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=REPO / "data/raw/bite2text")
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = ap.parse_args()

    cases = sorted(p for p in args.data_root.iterdir() if p.is_dir())
    if not cases:
        print(f"No cases under {args.data_root}", file=sys.stderr)
        return 1
    good = next(c for c in cases if (c / "ios" / "ios_upper.stl").stat().st_size > 1000)

    tmp = Path(tempfile.mkdtemp(prefix="b2t-container-test-"))
    print(f"scratch: {tmp}")
    empty = tmp / "empty.stl"
    empty.write_bytes(b"")

    scenarios = [
        ("normal", good / "ios" / "ios_upper.stl", good / "ios" / "ios_lower.stl", False),
        ("flat-layout", good / "ios" / "ios_upper.stl", good / "ios" / "ios_lower.stl", True),
        ("empty-upper", empty, good / "ios" / "ios_lower.stl", False),
        ("no-meshes", None, None, False),
    ]

    failures = 0
    for name, upper, lower, flat in scenarios:
        case = build_case(tmp, name, upper, lower, flat)
        out = tmp / f"{name}-out"
        code, log = run_case(case, out)
        report_file = out / "diagnostic-imaging-report.json"
        ok = report_file.exists()
        report = ""
        if ok:
            try:
                payload = json.loads(report_file.read_text(encoding="utf-8"))
                report = payload["report"]
                ok = isinstance(report, str) and len(report.strip()) > 0
            except Exception as exc:  # noqa: BLE001
                ok = False
                report = f"<unreadable: {exc}>"
        status = "PASS" if ok and code == 0 else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"\n[{status}] {name}  exit={code}  chars={len(report)}")
        print(f"        {report[:140]}{'...' if len(report) > 140 else ''}")
        if status == "FAIL":
            print("        --- log ---")
            for line in log.splitlines()[-15:]:
                print(f"        {line}")

    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(scenarios) - failures}/{len(scenarios)} scenarios passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
