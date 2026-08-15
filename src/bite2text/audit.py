from __future__ import annotations

import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from PIL import Image

from .config import dataset_config
from .index import REPORT_KEYS, manifest_root


def _paths(value: object) -> list[str]:
    return [] if pd.isna(value) or not str(value) else str(value).split("|")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_hist(values: list[float], title: str, xlabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if values:
        ax.hist(values, bins=min(30, max(5, int(np.sqrt(len(values))))), color="#3568a8")
    ax.set(title=title, xlabel=xlabel, ylabel="Files")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def run_audit(
    manifest_path: str | Path, output_dir: str | Path, config_path: str | Path | None = None
) -> dict:
    manifest_path, out = Path(manifest_path), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root, cfg = manifest_root(manifest_path), dataset_config(config_path)
    frame = pd.read_csv(manifest_path)
    issues, files = [], []
    image_pixels, mesh_faces, report_words = [], [], []
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    text_hashes: dict[str, list[str]] = defaultdict(list)

    for _, row in frame.iterrows():
        case_id = str(row["case_id"])
        for key in cfg["folders"]:
            for rel in _paths(row.get(f"{key}_paths", "")):
                path = root / rel
                rec = {"case_id": case_id, "modality": key, "path": rel, "ok": True, "error": ""}
                try:
                    rec["bytes"] = path.stat().st_size
                    digest = _hash(path)
                    rec["sha256"] = digest
                    hashes[digest].append((case_id, rel))
                    if key == "intraoral_photo":
                        with Image.open(path) as image:
                            image.verify()
                        with Image.open(path) as image:
                            rec.update(width=image.width, height=image.height, mode=image.mode)
                            image_pixels.append(image.width * image.height)
                    elif key == "ios":
                        mesh = trimesh.load(path, force="mesh", process=False)
                        rec.update(
                            vertices=len(mesh.vertices),
                            faces=len(mesh.faces),
                            watertight=bool(mesh.is_watertight),
                        )
                        mesh_faces.append(len(mesh.faces))
                        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                            raise ValueError("empty mesh")
                    elif key in REPORT_KEYS:
                        text = path.read_text(encoding="utf-8").strip()
                        normalized = " ".join(text.lower().split())
                        words = len(text.split())
                        rec.update(characters=len(text), words=words, lines=len(text.splitlines()))
                        report_words.append(words)
                        text_hashes[hashlib.sha256(normalized.encode()).hexdigest()].append(case_id)
                        if words < 3:
                            issues.append(
                                {
                                    "severity": "warning",
                                    "case_id": case_id,
                                    "issue": f"very short {key}",
                                }
                            )
                # Third-party decoders expose heterogeneous exception types; a failed
                # asset is an audit result and must not abort the remaining inventory.
                except Exception as exc:  # noqa: BLE001
                    rec.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                    issues.append(
                        {"severity": "error", "case_id": case_id, "issue": f"invalid {key}: {rel}"}
                    )
                files.append(rec)

    required = list(cfg["folders"])
    missing = Counter()
    for _, row in frame.iterrows():
        for key in required:
            if int(row.get(f"{key}_count", 0)) == 0:
                missing[key] += 1
    for digest, members in hashes.items():
        if len({m[0] for m in members}) > 1:
            issues.append(
                {
                    "severity": "warning",
                    "case_id": "multiple",
                    "issue": f"binary duplicate across cases: {digest[:12]}",
                }
            )
    duplicate_text_groups = sum(1 for ids in text_hashes.values() if len(set(ids)) > 1)

    file_frame, issue_frame = (
        pd.DataFrame(files),
        pd.DataFrame(issues, columns=["severity", "case_id", "issue"]),
    )
    file_frame.to_csv(out / "file_audit.csv", index=False)
    issue_frame.to_csv(out / "issues.csv", index=False)
    _save_hist(image_pixels, "Intraoral photograph resolution", "Pixels", out / "image_pixels.png")
    _save_hist(mesh_faces, "IOS mesh complexity", "Faces", out / "mesh_faces.png")
    _save_hist(report_words, "Report length", "Words", out / "report_words.png")
    summary = {
        "cases_discovered": len(frame),
        "expected_cases": int(cfg["expected_cases"]),
        "case_count_matches_expected": len(frame) == int(cfg["expected_cases"]),
        "files_audited": len(files),
        "invalid_files": int((~file_frame["ok"]).sum()) if len(file_frame) else 0,
        "issues": len(issues),
        "missing_cases_by_modality": dict(missing),
        "cross_case_duplicate_binary_groups": sum(
            1 for members in hashes.values() if len({m[0] for m in members}) > 1
        ),
        "cross_case_duplicate_report_groups": duplicate_text_groups,
        "privacy": "Aggregate report; no case IDs, source text, photographs, or mesh renders embedded.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cards = "".join(
        f"<li><b>{html.escape(str(k))}:</b> {html.escape(str(v))}</li>" for k, v in summary.items()
    )
    missing_rows = (
        "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in missing.items())
        or "<tr><td colspan=2>None</td></tr>"
    )
    report = f"""<!doctype html><html><head><meta charset='utf-8'><title>Bite2Text audit</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:40px auto;padding:0 20px;color:#17202a}}img{{max-width:32%;min-width:280px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:7px}}.note{{background:#eef5ff;padding:14px}}</style></head><body>
<h1>Bite2Text data audit</h1><p class='note'>{summary["privacy"]}</p><h2>Summary</h2><ul>{cards}</ul>
<h2>Missingness</h2><table><tr><th>Modality</th><th>Cases missing</th></tr>{missing_rows}</table>
<h2>Distributions</h2><img src='image_pixels.png' alt='Image pixels histogram'><img src='mesh_faces.png' alt='Mesh faces histogram'><img src='report_words.png' alt='Report word histogram'>
<h2>Interpretation</h2><p>Structural checks do not establish clinical correctness. Review <code>issues.csv</code> and <code>file_audit.csv</code> inside the authorized environment. A case-count mismatch is expected for fixtures and partial downloads.</p></body></html>"""
    (out / "report.html").write_text(report, encoding="utf-8")
    return summary
