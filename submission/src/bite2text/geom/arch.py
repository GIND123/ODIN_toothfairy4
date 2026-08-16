"""Extract an occlusal ridge profile from an intraoral arch surface.

A dental arch is a U-shaped curve, so the natural coordinate along it is an angle rather
than a Cartesian axis. We place a pole inside the arch and describe every measurement as a
function of the **arch angle** ``phi``:

* ``phi = 0``   anterior midline (between the central incisors)
* ``phi > 0``   patient's right, increasing posteriorly
* ``phi < 0``   patient's left, increasing posteriorly

For each angular bin we recover the occlusal extreme (the cusp tip / incisal edge: the
*highest* point of the lower arch, the *lowest* point of the upper arch) plus the buccal and
lingual limits of the occlusal band. Those three series are enough to derive overbite,
overjet, transverse relationships and the occlusal curves, and they are cheap: everything is
a grouped reduction over the raw vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage

from .canonical import CanonicalFrame, canonicalize

__all__ = ["ArchProfile", "CaseGeometry", "extract_arch_profile", "load_arch_mesh"]

PHI_LIMIT_DEG = 150.0
PHI_STEP_DEG = 1.0
OCCLUSAL_BAND_MM = 2.0


def load_arch_mesh(path: str | Path) -> np.ndarray:
    """Load an arch surface and return its vertices as an (N, 3) array."""
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"No geometry in {path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1000:
        raise ValueError(f"Unusable arch mesh at {path}: shape {points.shape}")
    finite = np.isfinite(points).all(axis=1)
    if not finite.all():
        points = points[finite]
    return points


@dataclass
class ArchProfile:
    """Occlusal ridge of one arch, sampled on a fixed angular grid."""

    jaw: str  # "upper" | "lower"
    phi: np.ndarray  # (K,) arch angle in degrees
    valid: np.ndarray  # (K,) bool — bin had enough support
    z_occlusal: np.ndarray  # (K,) occlusal extreme height (mm)
    r_ridge: np.ndarray  # (K,) radius of the occlusal extreme
    x_ridge: np.ndarray  # (K,)
    y_ridge: np.ndarray  # (K,)
    r_buccal: np.ndarray  # (K,) outermost radius within the occlusal band
    r_lingual: np.ndarray  # (K,) innermost radius within the occlusal band
    pole: np.ndarray  # (2,) polar origin in the canonical XY plane

    def at(self, phi_deg: float) -> int:
        """Index of the grid bin nearest ``phi_deg``."""
        return int(np.argmin(np.abs(self.phi - phi_deg)))

    def window(self, lo_deg: float, hi_deg: float) -> np.ndarray:
        """Boolean mask of valid bins whose angle falls in [lo, hi]."""
        return self.valid & (self.phi >= lo_deg) & (self.phi <= hi_deg)


def _grouped_extreme(values: np.ndarray, bins: np.ndarray, nbins: int, mode: str) -> np.ndarray:
    idx = np.arange(nbins)
    fn = ndimage.maximum if mode == "max" else ndimage.minimum
    out = fn(values, labels=bins + 1, index=idx + 1)
    return np.asarray(out, dtype=np.float64)


def extract_arch_profile(
    points: np.ndarray,
    jaw: str,
    pole: np.ndarray | None = None,
    min_points_per_bin: int = 12,
) -> ArchProfile:
    """Build the occlusal ridge profile for one canonicalised arch."""
    if jaw not in ("upper", "lower"):
        raise ValueError(f"jaw must be 'upper' or 'lower', got {jaw!r}")
    points = np.asarray(points, dtype=np.float64)

    # Occlusal side: lower arch chews upward (+z), upper arch downward (-z).
    occlusal_mode = "max" if jaw == "lower" else "min"

    # Seed the pole from the occlusal quarter of the arch, which is dominated by the dental
    # ridge rather than by gingiva, then keep it fixed for both arches.
    #
    # The pole is the *midpoint of the ridge's extent*, not its median: a U-shaped point set
    # has a bimodal lateral distribution, so the median lands on whichever arm carries more
    # vertices and throws the angular origin off the midline by a centimetre or more.
    if pole is None:
        z = points[:, 2]
        cut = np.quantile(z, 0.75 if jaw == "lower" else 0.25)
        ridgeish = points[z >= cut] if jaw == "lower" else points[z <= cut]
        if len(ridgeish) < 500:
            ridgeish = points
        lo = np.quantile(ridgeish[:, :2], 0.01, axis=0)
        hi = np.quantile(ridgeish[:, :2], 0.99, axis=0)
        pole = 0.5 * (lo + hi)
    pole = np.asarray(pole, dtype=np.float64).reshape(2)

    dx = points[:, 0] - pole[0]
    dy = points[:, 1] - pole[1]
    phi = np.degrees(np.arctan2(dx, dy))
    radius = np.hypot(dx, dy)

    edges = np.arange(-PHI_LIMIT_DEG, PHI_LIMIT_DEG + PHI_STEP_DEG, PHI_STEP_DEG)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nbins = len(centers)

    bins = np.digitize(phi, edges) - 1
    inside = (bins >= 0) & (bins < nbins)
    bins = bins[inside]
    pts = points[inside]
    radius = radius[inside]
    if len(pts) < 1000:
        raise ValueError(f"Arch {jaw} has too few points inside the angular window")

    counts = np.bincount(bins, minlength=nbins)
    valid = counts >= min_points_per_bin

    z = pts[:, 2]
    z_occ = _grouped_extreme(z, bins, nbins, occlusal_mode)

    # Occlusal band: points within OCCLUSAL_BAND_MM of that bin's occlusal extreme.
    per_point_extreme = z_occ[bins]
    if jaw == "lower":
        in_band = z >= per_point_extreme - OCCLUSAL_BAND_MM
    else:
        in_band = z <= per_point_extreme + OCCLUSAL_BAND_MM

    band_bins = bins[in_band]
    band_r = radius[in_band]
    band_counts = np.bincount(band_bins, minlength=nbins)
    r_buccal = _grouped_extreme(band_r, band_bins, nbins, "max")
    r_lingual = _grouped_extreme(band_r, band_bins, nbins, "min")
    valid &= band_counts >= 3

    # Representative ridge point per bin: the in-band point closest to the occlusal extreme.
    ridge_x = np.full(nbins, np.nan)
    ridge_y = np.full(nbins, np.nan)
    ridge_r = np.full(nbins, np.nan)
    order = np.argsort(bins, kind="stable")
    sorted_bins = bins[order]
    starts = np.searchsorted(sorted_bins, np.arange(nbins), side="left")
    stops = np.searchsorted(sorted_bins, np.arange(nbins), side="right")
    for b in range(nbins):
        if not valid[b]:
            continue
        sel = order[starts[b] : stops[b]]
        if sel.size == 0:
            valid[b] = False
            continue
        best = sel[np.argmin(np.abs(z[sel] - z_occ[b]))]
        ridge_x[b] = pts[best, 0]
        ridge_y[b] = pts[best, 1]
        ridge_r[b] = radius[best]

    z_occ[~valid] = np.nan
    r_buccal[~valid] = np.nan
    r_lingual[~valid] = np.nan

    return ArchProfile(
        jaw=jaw,
        phi=centers,
        valid=valid,
        z_occlusal=z_occ,
        r_ridge=ridge_r,
        x_ridge=ridge_x,
        y_ridge=ridge_y,
        r_buccal=r_buccal,
        r_lingual=r_lingual,
        pole=pole,
    )


@dataclass
class CaseGeometry:
    """Both arches of one case, canonicalised and profiled on a shared pole."""

    case_id: str
    frame: CanonicalFrame
    upper_points: np.ndarray
    lower_points: np.ndarray
    upper: ArchProfile
    lower: ArchProfile

    @classmethod
    def from_meshes(cls, case_id: str, upper_path: str | Path, lower_path: str | Path) -> "CaseGeometry":
        upper_raw = load_arch_mesh(upper_path)
        lower_raw = load_arch_mesh(lower_path)
        frame = canonicalize(upper_raw, lower_raw)
        upper = frame.apply(upper_raw)
        lower = frame.apply(lower_raw)

        # A shared pole keeps the two profiles angularly comparable, which every
        # inter-arch measurement depends on. Derive it from the lower arch (its occlusal
        # ridge is the more reliable of the two) and reuse it for the upper.
        seed = extract_arch_profile(lower, "lower")
        return cls(
            case_id=case_id,
            frame=frame,
            upper_points=upper,
            lower_points=lower,
            upper=extract_arch_profile(upper, "upper", pole=seed.pole),
            lower=seed,
        )
