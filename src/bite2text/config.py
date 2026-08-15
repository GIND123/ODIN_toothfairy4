from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dataset_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or ROOT / "config" / "dataset.yaml")
