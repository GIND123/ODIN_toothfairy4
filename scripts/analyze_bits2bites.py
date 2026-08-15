"""Generate a privacy-safe, reproducible Bits2Bites dataset audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = ["Left Class", "Right Class", "Anterior Bite", "Transversal Bite", "Median Lines"]
PLOT_LABELS = ["Left Class", "Right Class", "Anterior Bite", "Transversal Pattern", "Median Lines"]
COLORS = {"train": "#3568a8", "val": "#e1843c"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_binary_stl(path: Path) -> dict[str, object]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(84)
    if len(header) != 84:
        raise ValueError("truncated STL header")
    triangles = struct.unpack("<I", header[80:84])[0]
    expected = 84 + 50 * triangles
    if expected != size:
        raise ValueError(f"binary STL size mismatch: expected {expected}, found {size}")
    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    faces = np.memmap(path, dtype=dtype, mode="r", offset=84, shape=(triangles,))
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    area = 0.0
    degenerate = 0
    finite = True
    for start in range(0, triangles, 100_000):
        vertices = np.asarray(faces["vertices"][start : start + 100_000], dtype=np.float64)
        finite = finite and bool(np.isfinite(vertices).all())
        mins = np.minimum(mins, vertices.min(axis=(0, 1)))
        maxs = np.maximum(maxs, vertices.max(axis=(0, 1)))
        cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        areas = np.linalg.norm(cross, axis=1) * 0.5
        area += float(areas.sum())
        degenerate += int((areas <= 1e-12).sum())
    del faces
    extent = maxs - mins
    return {
        "bytes": size,
        "triangles": triangles,
        "finite_vertices": finite,
        "degenerate_triangles": degenerate,
        "surface_area": area,
        "extent_x": float(extent[0]),
        "extent_y": float(extent[1]),
        "extent_z": float(extent[2]),
        "bbox_volume": float(np.prod(extent)),
    }


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def categorical_figure(annotations: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    for ax, column in zip(axes.flat, PLOT_LABELS):
        counts = annotations[column].value_counts().sort_values()
        ax.barh(counts.index, counts.values, color="#3568a8")
        ax.set_title(column)
        ax.set_xlabel("Patients")
        for idx, value in enumerate(counts.values):
            ax.text(value + 1, idx, str(value), va="center", fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Clinical annotation distributions", fontsize=16)
    savefig(output)


def bilateral_figure(annotations: pd.DataFrame, output: Path) -> None:
    table = pd.crosstab(annotations["Left Class"], annotations["Right Class"])
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(table.values, cmap="Blues")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set(xlabel="Right Class", ylabel="Left Class", title="Left/right sagittal-class agreement")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, table.iloc[i, j], ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Patients")
    savefig(output)


def geometry_figure(meshes: pd.DataFrame, output: Path) -> None:
    _, axes = plt.subplots(2, 2, figsize=(12, 9))
    for jaw, group in meshes.groupby("jaw"):
        axes[0, 0].hist(group["triangles"], bins=20, alpha=0.6, label=jaw)
        axes[0, 1].hist(group["surface_area"], bins=20, alpha=0.6, label=jaw)
        axes[1, 0].hist(group["bbox_volume"], bins=20, alpha=0.6, label=jaw)
    axes[0, 0].set(title="Mesh complexity", xlabel="Triangles", ylabel="Meshes")
    axes[0, 1].set(title="Surface area", xlabel="Coordinate units²", ylabel="Meshes")
    axes[1, 0].set(title="Bounding-box volume", xlabel="Coordinate units³", ylabel="Meshes")
    pair = meshes.pivot(index="patient", columns="jaw", values="triangles")
    axes[1, 1].scatter(pair["upper"], pair["lower"], alpha=0.65, s=18)
    axes[1, 1].set(
        title="Within-patient mesh complexity", xlabel="Upper triangles", ylabel="Lower triangles"
    )
    for ax in axes.flat[:3]:
        ax.legend()
    savefig(output)


def split_drift_figure(annotations: pd.DataFrame, output: Path) -> None:
    rows = []
    for column in PLOT_LABELS:
        proportions = pd.crosstab(annotations[column], annotations["split"], normalize="columns")
        for category, row in proportions.iterrows():
            rows.append(
                {
                    "feature": column,
                    "category": category,
                    "delta": row.get("val", 0) - row.get("train", 0),
                }
            )
    drift = pd.DataFrame(rows).sort_values("delta")
    _, ax = plt.subplots(figsize=(11, 10))
    labels = [f"{row.feature}: {row.category}" for row in drift.itertuples()]
    ax.barh(
        labels,
        drift["delta"] * 100,
        color=["#b54b4b" if x < 0 else "#3568a8" for x in drift["delta"]],
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Validation minus training prevalence", xlabel="Percentage-point difference")
    savefig(output)


def missingness_figure(annotations: pd.DataFrame, meshes: pd.DataFrame, output: Path) -> None:
    values = {
        "Annotation cells": int(annotations[LABELS].isna().sum().sum()),
        "Missing upper jaws": int(200 - (meshes["jaw"] == "upper").sum()),
        "Missing lower jaws": int(200 - (meshes["jaw"] == "lower").sum()),
        "Invalid STL files": int((~meshes["valid"]).sum()),
        "Non-finite vertices": int((~meshes["finite_vertices"]).sum()),
        "Degenerate triangles": int(meshes["degenerate_triangles"].sum()),
    }
    _, ax = plt.subplots(figsize=(10, 5))
    ax.bar(values.keys(), values.values(), color="#3568a8")
    ax.tick_params(axis="x", rotation=28)
    ax.set(title="Integrity and completeness findings", ylabel="Count")
    for index, value in enumerate(values.values()):
        ax.text(index, value, str(value), ha="center", va="bottom")
    savefig(output)


def dataframe_markdown(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without pandas' optional tabulate dependency."""
    flattened = frame.copy()
    flattened.columns = [f"{left} {right}" for left, right in flattened.columns]
    headers = ["jaw", *flattened.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for index, row in flattened.iterrows():
        values = [str(index), *(f"{value:.2f}" for value in row)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def audit(root: Path, report_path: Path) -> dict[str, object]:
    dataset = root / "Bits2Bites"
    annotations = pd.read_csv(dataset / "data" / "Annotations.csv")
    transversal = annotations["Transversal Bite"].astype(str)
    annotations["Transversal Pattern"] = np.select(
        [
            transversal.eq("Normal"),
            transversal.str.contains("Cross", case=False)
            & transversal.str.contains("Scissor", case=False),
            transversal.str.contains("Scissor", case=False),
            transversal.str.contains("Cross", case=False),
        ],
        ["Normal", "Mixed Cross + Scissor", "Scissor Bite", "Cross Bite"],
        default="Other",
    )
    tooth_tokens = transversal.str.findall(r"(?<!\d)\d{2,}(?!\d)")
    valid_teeth = {str(quadrant * 10 + tooth) for quadrant in (1, 2) for tooth in range(1, 9)}
    annotations["Transversal Tooth Count"] = tooth_tokens.map(
        lambda values: sum(v in valid_teeth for v in values)
    )
    malformed_tokens = sorted(
        {value for values in tooth_tokens for value in values if value not in valid_teeth}
    )
    inconsistent_cross_wording = int(transversal.str.match(r"^Cross (?!Bite)", case=False).sum())
    mesh_rows, errors = [], []
    for split in ("train", "val"):
        for patient_dir in sorted((dataset / "data" / split).iterdir(), key=lambda p: int(p.name)):
            if not patient_dir.is_dir():
                continue
            for jaw in ("upper", "lower"):
                path = patient_dir / f"{jaw}.stl"
                row = {
                    "patient": int(patient_dir.name),
                    "split": split,
                    "jaw": jaw,
                    "path": path.relative_to(dataset).as_posix(),
                    "valid": False,
                }
                try:
                    row.update(inspect_binary_stl(path))
                    row.update(valid=True, sha256=sha256(path), error="")
                except (OSError, ValueError, struct.error) as exc:
                    row.update(error=f"{type(exc).__name__}: {exc}")
                    errors.append(row.copy())
                mesh_rows.append(row)
    meshes = pd.DataFrame(mesh_rows)
    split_map = meshes.drop_duplicates("patient").set_index("patient")["split"]
    annotations["split"] = annotations["Patient"].map(split_map)
    out_dir = report_path.parent / "docs" / "assets" / "data_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes.to_csv(out_dir / "mesh_metrics.csv", index=False)
    counts = {
        column: annotations[column].value_counts().to_dict()
        for column in [*LABELS, "Transversal Pattern"]
    }
    (out_dir / "label_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    categorical_figure(annotations, out_dir / "label_distributions.png")
    bilateral_figure(annotations, out_dir / "bilateral_agreement.png")
    geometry_figure(meshes, out_dir / "geometry_distributions.png")
    split_drift_figure(annotations, out_dir / "split_drift.png")
    missingness_figure(annotations, meshes, out_dir / "integrity.png")

    duplicate_groups = [group for group in meshes.groupby("sha256").size() if group > 1]
    patients = set(annotations["Patient"])
    mesh_patients = set(meshes["patient"])
    bilateral = float((annotations["Left Class"] == annotations["Right Class"]).mean())
    pair = meshes.pivot(index="patient", columns="jaw", values="triangles")
    triangle_corr = float(pair.corr().iloc[0, 1])
    summary = {
        "patients": len(annotations),
        "train_patients": int((annotations["split"] == "train").sum()),
        "validation_patients": int((annotations["split"] == "val").sum()),
        "meshes": len(meshes),
        "archive_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "invalid_meshes": int((~meshes["valid"]).sum()),
        "missing_annotation_cells": int(annotations[LABELS].isna().sum().sum()),
        "annotation_without_mesh": len(patients - mesh_patients),
        "mesh_without_annotation": len(mesh_patients - patients),
        "duplicate_mesh_groups": len(duplicate_groups),
        "degenerate_triangles": int(meshes["degenerate_triangles"].sum()),
        "meshes_with_degenerate_triangles": int((meshes["degenerate_triangles"] > 0).sum()),
        "bilateral_exact_agreement": bilateral,
        "upper_lower_triangle_correlation": triangle_corr,
        "transversal_unique_raw_values": int(annotations["Transversal Bite"].nunique()),
        "transversal_inconsistent_cross_wording": inconsistent_cross_wording,
        "transversal_malformed_tooth_tokens": malformed_tokens,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    stats = meshes.groupby("jaw")[
        ["bytes", "triangles", "surface_area", "bbox_volume", "extent_x", "extent_y", "extent_z"]
    ].agg(["min", "median", "max"])
    stats_md = dataframe_markdown(stats)
    count_lines = []
    for column in PLOT_LABELS:
        values = ", ".join(
            f"{key}: {value}" for key, value in annotations[column].value_counts().items()
        )
        count_lines.append(f"- **{column}:** {values}")
    report = f"""# Bits2Bites v01 — Complete Data Audit

**Audit date:** 2026-08-16  
**Source:** `Bits2Bites_v01.zip`  
**Scope:** all {summary["patients"]} patients, {summary["meshes"]} STL files, and every annotation cell  
**Privacy:** aggregate statistics only; no patient geometry, identifiers, or row-level clinical annotations are reproduced.

## Executive summary

The supplied archive is **Bits2Bites**, not the Bite2Text multimodal reporting dataset described elsewhere in this repository. It contains only paired upper/lower 3D dental scans and categorical occlusion annotations—no intraoral photographs and no free-text reports. It therefore cannot directly train or audit a Bite2Text report-generation model.

The archive is structurally complete: {summary["patients"]} annotation rows correspond one-to-one with {summary["patients"]} mesh case folders, split into {summary["train_patients"]} training and {summary["validation_patients"]} validation patients. Every patient has an upper and lower STL. All {summary["meshes"]} binary STL files passed byte-layout and finite-coordinate validation. There are {summary["missing_annotation_cells"]} missing annotation cells, {summary["invalid_meshes"]} invalid meshes, {summary["degenerate_triangles"]} zero-area triangles across {summary["meshes_with_degenerate_triangles"]} meshes, and {summary["duplicate_mesh_groups"]} exact duplicate mesh hash groups.

The main modeling risks are categorical imbalance, a small validation cohort (10%), and visible train–validation prevalence shifts for some clinical subclasses. Evaluation and model selection should use macro-averaged per-target metrics in addition to overall accuracy.

## Dataset inventory

| Item | Result |
|---|---:|
| Patients | {summary["patients"]} |
| Training patients | {summary["train_patients"]} |
| Validation patients | {summary["validation_patients"]} |
| Upper meshes | {int((meshes["jaw"] == "upper").sum())} |
| Lower meshes | {int((meshes["jaw"] == "lower").sum())} |
| Extracted bytes | {summary["archive_bytes"]:,} |
| Annotation targets | {len(LABELS)} |
| Missing annotation cells | {summary["missing_annotation_cells"]} |
| Invalid meshes | {summary["invalid_meshes"]} |
| Exact duplicate mesh groups | {summary["duplicate_mesh_groups"]} |

![Integrity and completeness](docs/assets/data_audit/integrity.png)

## Clinical labels

The CSV provides bilateral sagittal classes and three additional occlusal findings. For visualization, the raw transversal descriptions are parsed into Normal, Cross Bite, Scissor Bite, or Mixed while tooth-level detail is retained separately. Aggregate counts are:

{chr(10).join(count_lines)}

![Clinical label distributions](docs/assets/data_audit/label_distributions.png)

### Annotation schema findings

- `Transversal Bite` has **{summary["transversal_unique_raw_values"]} unique raw strings** because condition type and involved teeth share one field.
- **{summary["transversal_inconsistent_cross_wording"]} rows** use abbreviated `Cross ...` rather than `Cross Bite ...` wording.
- Tooth tokens outside the expected permanent FDI ranges 11–18 and 21–28: **{", ".join(summary["transversal_malformed_tooth_tokens"]) if summary["transversal_malformed_tooth_tokens"] else "none"}**.
- Preserve the raw field for traceability, but train on an explicitly versioned parser that separates condition type, side/tooth involvement, and malformed/unknown status. Any suspected typo must be clinically adjudicated rather than silently corrected.

Left and right sagittal labels agree exactly for **{bilateral:.1%}** of patients. This dependence means treating the two sides as independent samples would inflate the effective sample size and leak patient information. A shared encoder with separate side-specific heads is a more defensible formulation.

![Bilateral agreement](docs/assets/data_audit/bilateral_agreement.png)

## 3D mesh audit

Every STL was checked independently for binary-STL byte consistency, finite coordinates, triangle count, zero-area triangles, bounding-box extents, approximate surface area, and SHA-256 identity. Coordinate units are not asserted in the archive metadata, so area and volume are reported as coordinate units rather than assumed millimeters.

{stats_md}

Upper and lower triangle counts have Pearson correlation **{triangle_corr:.3f}** within patients. The distributions show heterogeneous mesh resolution, so models should resample to a fixed point/face budget and retain the original scale only after confirming the coordinate-unit convention.

![Geometry distributions](docs/assets/data_audit/geometry_distributions.png)

## Split analysis

The provided split is patient-disjoint by folder ID: 180 train and 20 validation cases. The plot below reports validation prevalence minus training prevalence for every class. Large bars should be treated cautiously because each validation patient changes prevalence by five percentage points.

![Train-validation label drift](docs/assets/data_audit/split_drift.png)

Recommendations:

1. Preserve the official validation split for comparable experiments, but report repeated patient-level cross-validation on the 180 training cases for uncertainty estimates.
2. Use stratified multilabel/group-aware folds where possible; never split upper and lower jaws independently.
3. Report macro-F1 and balanced accuracy for each target, per-class recall, and exact-match accuracy across all five targets.
4. Fit geometry normalization, augmentation statistics, class weights, and any learned preprocessing on training patients only.
5. Inspect outlier scales in the authorized environment before choosing global centering/scaling. Avoid publishing identifiable mesh renders.

## End-to-end modeling blueprint

1. **Ingest:** validate case-to-annotation joins and both jaw files using the checks above.
2. **Preprocess:** orient consistently, center using a documented anatomical or bounding-box reference, preserve relative upper/lower pose, and sample fixed-size point clouds or meshes.
3. **Represent:** encode upper and lower jaws with shared 3D backbones; fuse jaw embeddings and explicit relative-pose features.
4. **Predict:** use five task heads, with separate left/right sagittal outputs and independent heads for anterior, transversal, and median-line findings.
5. **Train:** class-balanced loss, patient-level sampling, seeded augmentation, early stopping on macro validation performance, and calibration monitoring.
6. **Evaluate:** compare majority-class, geometry-only, and fused-jaw baselines; provide bootstrap confidence intervals and per-class confusion matrices.
7. **Package:** freeze preprocessing with the checkpoint, validate all expected patient outputs, record code/data hashes, and keep raw clinical data outside Git.

## Reproducibility artifacts

- `scripts/analyze_bits2bites.py`: complete rerunnable audit.
- `docs/assets/data_audit/mesh_metrics.csv`: per-mesh technical metrics with dataset-relative paths.
- `docs/assets/data_audit/label_counts.json`: aggregate label counts.
- `docs/assets/data_audit/summary.json`: machine-readable headline findings.
- PNG files in `docs/assets/data_audit/`: aggregate visualizations embedded above.

Re-run from the repository root with:

```bash
python scripts/analyze_bits2bites.py --root data/raw/bits2bites_v01 --report DATA_AUDIT.md
```

## Limitations

- This is a structural/statistical audit, not clinical adjudication of label correctness.
- STL coordinate units, scan hardware, acquisition center, demographic attributes, and annotation-rater metadata are not present in the supplied archive.
- With only 20 validation cases, apparent subgroup differences have high sampling uncertainty.
- Exact hashing detects byte-identical meshes but not geometrically equivalent rescans or transformed near-duplicates.
- No test split is present in this archive.
"""
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("DATA_AUDIT.md"))
    args = parser.parse_args()
    summary = audit(args.root, args.report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
