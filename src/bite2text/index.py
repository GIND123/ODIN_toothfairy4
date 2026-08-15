from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import dataset_config

REPORT_KEYS = ("reports_ios_it", "reports_ios_en", "reports_photo_it", "reports_photo_en")


def _case_token(path: Path, regex: str) -> str:
    match = re.search(regex, path.stem)
    return match.group("case") if match and "case" in match.groupdict() else path.stem


def _pseudo(token: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{token}".encode()).hexdigest()[:16]


def build_manifest(
    data_root: str | Path, output: str | Path, config_path: str | Path | None = None
) -> pd.DataFrame:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    cfg = dataset_config(config_path)
    salt = os.getenv(cfg.get("salt_env", "BITE2TEXT_ID_SALT"), "development-only-change-me")
    records: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for key, folder in cfg["folders"].items():
        base = root / folder
        if not base.exists():
            continue
        allowed = set(cfg["extensions"]["report" if key in REPORT_KEYS else key])
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            if path.suffix.lower() not in allowed:
                continue
            token = _case_token(path, cfg["id_regex"])
            records[token][key].append(path.relative_to(root).as_posix())

    rows = []
    for token, modalities in sorted(records.items()):
        row: dict[str, object] = {"case_id": _pseudo(token, salt)}
        for key in cfg["folders"]:
            paths = modalities.get(key, [])
            row[f"{key}_paths"] = "|".join(paths)
            row[f"{key}_count"] = len(paths)
        rows.append(row)
    frame = pd.DataFrame(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    (output.with_suffix(".root.txt")).write_text(str(root), encoding="utf-8")
    return frame


def manifest_root(manifest_path: str | Path) -> Path:
    marker = Path(manifest_path).with_suffix(".root.txt")
    if not marker.exists():
        raise FileNotFoundError(f"Missing dataset-root marker: {marker}")
    return Path(marker.read_text(encoding="utf-8").strip())
