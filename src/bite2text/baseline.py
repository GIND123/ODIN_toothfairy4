from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .index import manifest_root


def _first(root: Path, value: object) -> str:
    paths = [] if pd.isna(value) or not str(value) else str(value).split("|")
    return (root / paths[0]).read_text(encoding="utf-8").strip() if paths else ""


def train(manifest: str | Path, output: str | Path) -> None:
    frame, root = pd.read_csv(manifest), manifest_root(manifest)
    source = [
        _first(root, row.get("reports_photo_en_paths", ""))
        or _first(root, row.get("reports_ios_it_paths", ""))
        for _, row in frame.iterrows()
    ]
    targets = [_first(root, row.get("reports_ios_en_paths", "")) for _, row in frame.iterrows()]
    keep = [i for i, (x, y) in enumerate(zip(source, targets)) if x and y]
    if not keep:
        raise ValueError("No paired source and English IOS target reports found")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000)
    matrix = vectorizer.fit_transform([source[i] for i in keep])
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine").fit(matrix)
    artifact = {
        "vectorizer": vectorizer,
        "neighbors": neighbors,
        "targets": [targets[i] for i in keep],
        "source_case_ids": [str(frame.iloc[i]["case_id"]) for i in keep],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("wb") as handle:
        pickle.dump(artifact, handle)


def predict(manifest: str | Path, model: str | Path, output: str | Path) -> list[dict[str, str]]:
    frame, root = pd.read_csv(manifest), manifest_root(manifest)
    with Path(model).open("rb") as handle:
        artifact = pickle.load(handle)  # trusted local model artifact only
    rows = []
    for _, row in frame.iterrows():
        source = _first(root, row.get("reports_photo_en_paths", "")) or _first(
            root, row.get("reports_ios_it_paths", "")
        )
        if source:
            idx = artifact["neighbors"].kneighbors(
                artifact["vectorizer"].transform([source]), return_distance=False
            )[0, 0]
            report = artifact["targets"][idx]
        else:
            report = "Orthodontic findings could not be determined from the available inputs."
        rows.append({"case_id": str(row["case_id"]), "report": report})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def validate_submission(input_path: str | Path, manifest: str | Path) -> dict[str, object]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Submission must be a JSON list")
    expected = set(pd.read_csv(manifest)["case_id"].astype(str))
    ids = [str(item.get("case_id", "")) for item in payload]
    errors = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate case_id values")
    if set(ids) != expected:
        errors.append(
            f"case coverage mismatch: missing={len(expected - set(ids))}, extra={len(set(ids) - expected)}"
        )
    if any(
        not isinstance(item.get("report"), str) or not item["report"].strip() for item in payload
    ):
        errors.append("one or more reports are empty or non-string")
    result = {"valid": not errors, "cases": len(payload), "errors": errors}
    if errors:
        raise ValueError("; ".join(errors))
    return result
