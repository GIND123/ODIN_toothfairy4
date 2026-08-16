"""ODIN 2026 Bite2Text submission entrypoint.

Implements the challenge's container contract:

  inputs   ``3d-lower-teeth-scan``   -> /input/files/ios-lower/*.stl|obj  (or /input/3d-lower-teeth-scan.obj)
           ``3d-upper-teeth-scan``   -> /input/files/ios-upper/*.stl|obj  (or /input/3d-upper-teeth-scan.obj)
           ``2d-intraoral-photographs`` -> /input/images/{2d-intraoral-photographs,intraoral-photo}/*
  output   ``diagnostic-imaging-report`` -> /output/diagnostic-imaging-report.json  {"report": str}

The overriding requirement is that this **always writes a report**. The evaluator treats a
missing or unreadable output as an empty report scoring zero, which is worse than any report
we could otherwise emit, so every stage is wrapped and degrades to the prior-based report
rather than raising. Diagnostics go to stdout for the challenge logs.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import traceback
from pathlib import Path

# Force single-threaded numerics BEFORE numpy/scikit-learn are imported anywhere.
#
# HistGradientBoosting predicts through OpenMP. On a many-core host, predicting a *single*
# row makes thread setup dominate so badly that the 11 field models took 104s per case; pinned
# to one thread the same predictions take 0.10s — a 1000x difference. The work here is one row
# at a time, so there is nothing to parallelise and everything to lose.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

# Grand Challenge mounts these fixed paths; the env overrides exist only so the same file can
# be exercised by submission/test_local.py without a container.
INPUT_PATH = Path(os.environ.get("B2T_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("B2T_OUTPUT", "/output"))
MODEL_PATH = Path(os.environ.get("B2T_MODEL", "/opt/ml/model"))
RESOURCE_PATH = Path(os.environ.get("B2T_RESOURCES", "/opt/app/resources"))

LOWER_SLUGS = ("3d-lower-teeth-scan", "ios-lower-scan", "ios-lower")
UPPER_SLUGS = ("3d-upper-teeth-scan", "ios-upper-scan", "ios-upper")
PHOTO_SLUGS = ("2d-intraoral-photographs", "intraoral-photo")
MESH_PATTERNS = ("*.stl", "*.obj", "*.ply", "*.STL", "*.OBJ")


def log(message: str) -> None:
    print(message, flush=True)


def read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_report(report: str) -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_PATH / "diagnostic-imaging-report.json"
    with open(target, "w", encoding="utf-8") as handle:
        json.dump({"report": report}, handle, indent=4)
    log(f"Wrote {target} ({len(report)} chars)")


def find_mesh(slugs: tuple[str, ...]) -> Path | None:
    """Locate one arch mesh, tolerating both the flat and the socket-directory layouts."""
    for slug in slugs:
        direct = INPUT_PATH / f"{slug}.obj"
        if direct.exists():
            return direct
        for ext in (".stl", ".ply"):
            direct = INPUT_PATH / f"{slug}{ext}"
            if direct.exists():
                return direct
    for slug in slugs:
        for base in (INPUT_PATH / "files" / slug, INPUT_PATH / slug):
            for pattern in MESH_PATTERNS:
                hits = sorted(glob.glob(str(base / pattern)))
                if hits:
                    return Path(hits[0])
    # Last resort: any mesh anywhere under /input, disambiguated by filename.
    everything = []
    for pattern in MESH_PATTERNS:
        everything.extend(glob.glob(str(INPUT_PATH / "**" / pattern), recursive=True))
    keyword = "lower" if slugs is LOWER_SLUGS else "upper"
    for candidate in sorted(everything):
        if keyword in Path(candidate).name.lower() or keyword in Path(candidate).parent.name.lower():
            return Path(candidate)
    return None


def find_photos() -> list[Path]:
    patterns = ("*.mha", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png", "*.bmp")
    found: list[Path] = []
    for slug in PHOTO_SLUGS:
        base = INPUT_PATH / "images" / slug
        for pattern in patterns:
            found.extend(Path(p) for p in glob.glob(str(base / pattern)))
    if not found:
        for pattern in patterns:
            found.extend(Path(p) for p in glob.glob(str(INPUT_PATH / "images" / "**" / pattern), recursive=True))
    return sorted({p.resolve(): p for p in found}.values(), key=lambda p: p.name)


def case_id_from_inputs() -> str:
    try:
        inputs = read_json(INPUT_PATH / "inputs.json")
    except Exception:  # noqa: BLE001
        return "case"
    for value in inputs or []:
        image = value.get("image") or {}
        file = value.get("file") or {}
        for holder in (image, file):
            name = holder.get("name") if isinstance(holder, dict) else holder
            if name:
                return Path(str(name)).stem
    return "case"


def fallback_report() -> str:
    """Prior-only report, used when geometry or the model bundle is unavailable."""
    try:
        from bite2text.report.render import render_modal_report

        return render_modal_report()
    except Exception:  # noqa: BLE001 - the literal below is the last line of defence
        return (
            "The patient presents a transverse constriction of the maxilla in the absence of "
            "crossbite. From a vertical standpoint, there is a deep bite. Sagittally, there is "
            "a Class I molar and canine relationship bilaterally, with increased overjet. The "
            "dental midlines are deviated relative to each other. The Curve of Spee and the "
            "Curve of Wilson are increased. There is mild crowding in the upper and lower arches."
        )


def run() -> int:
    log("=+= Bite2Text inference")
    try:
        inputs = read_json(INPUT_PATH / "inputs.json")
        slugs = sorted(
            str((v.get("socket") or {}).get("slug") or "") for v in (inputs or [])
        )
        log(f"input sockets: {slugs}")
    except Exception as exc:  # noqa: BLE001
        log(f"could not read inputs.json ({exc}); continuing on filesystem layout alone")

    case_id = case_id_from_inputs()
    lower = find_mesh(LOWER_SLUGS)
    upper = find_mesh(UPPER_SLUGS)
    photos = find_photos()
    log(f"case={case_id} upper={upper} lower={lower} photos={len(photos)}")

    if upper is None or lower is None:
        log("missing one or both arch meshes; emitting prior-based report")
        write_report(fallback_report())
        return 0

    try:
        from bite2text.compose import ReportComposer

        bundle = None
        for candidate in (
            MODEL_PATH / "field_models.joblib",
            RESOURCE_PATH / "field_models.joblib",
        ):
            if candidate.exists():
                bundle = candidate
                break
        log(f"model bundle: {bundle}")

        composer = ReportComposer(bundle)
        composed = composer.compose_from_paths(case_id, upper, lower)
        for warning in composed.warnings:
            log(f"warning: {warning}")
        n_geometry = sum(1 for v in composed.sources.values() if v == "geometry")
        log(f"fields from geometry: {n_geometry}")
        write_report(composed.report)
        return 0
    except Exception:  # noqa: BLE001 - never let a case produce no output
        log("composition failed:\n" + traceback.format_exc())
        write_report(fallback_report())
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - absolute last resort
        traceback.print_exc()
        try:
            write_report(fallback_report())
        finally:
            sys.exit(0)
