"""Put a pair of intraoral arches into a canonical frame without trusting file metadata.

The published Bite2Text scans are stated to be in RAS (x=right, y=anterior, z=superior),
and the Bits2Bites archive matches that convention. We nevertheless re-derive the frame
from the geometry itself, because a silent axis flip on the hidden test set would corrupt
every downstream measurement and would be invisible in the container logs.

Two of the three axes are recoverable from anatomy alone:

* **Superior** is the direction separating the two arch centroids (the upper arch sits
  above the lower one).
* **Anterior** is the direction in which the dental arch *closes*: a U-shaped arch has two
  separated arms posteriorly and a single merged body anteriorly.

The remaining left/right sign is a genuine mirror ambiguity that surface geometry cannot
resolve, so it is taken from the stated convention and audited separately against the
tooth-numbered Bits2Bites labels (see ``scripts/calibrate_geometry.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CanonicalFrame", "canonicalize"]


@dataclass(frozen=True)
class CanonicalFrame:
    """Rotation taking source coordinates into (right, anterior, superior)."""

    rotation: np.ndarray  # (3, 3); rows are the canonical basis vectors
    superior_axis: int
    anterior_axis: int
    superior_sign: int
    anterior_sign: int
    arch_separation_mm: float
    anterior_confidence: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ self.rotation.T


def _axis_closure_score(points: np.ndarray, axis: int, lateral: int) -> np.ndarray:
    """Per-end 'this end is anterior' score for the low and high end of ``axis``.

    Two independent cues are combined, because either alone fails on some scans:

    * **Narrowing** — a dental arch tapers anteriorly, so the anterior end is markedly
      narrower across the lateral axis than the posterior end. This is the dominant cue and
      is robust to how much soft tissue the scan includes.
    * **Closure** — the anterior end is a single merged body, whereas the posterior end is
      two separated arms with an empty middle.

    The closure cue must be measured on the **mandible only**: the maxillary scan includes
    the palate, which fills the central region at every depth and erases the signal
    entirely. That is what silently mis-oriented the Bite2Text scans.
    """
    coord = points[:, axis]
    lat = points[:, lateral]
    lo, hi = np.quantile(coord, [0.02, 0.98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([0.0, 0.0])

    lat_lo, lat_hi = np.quantile(lat, [0.02, 0.98])
    lat_span = lat_hi - lat_lo
    if lat_span <= 0:
        return np.array([0.0, 0.0])

    span = hi - lo
    widths, closures = [], []
    for end in (0, 1):
        # Outer 25% slab at this end of the axis.
        mask = coord <= lo + 0.25 * span if end == 0 else coord >= hi - 0.25 * span
        if mask.sum() < 200:
            widths.append(np.nan)
            closures.append(0.0)
            continue
        slab = lat[mask]
        w_lo, w_hi = np.quantile(slab, [0.02, 0.98])
        widths.append(float(w_hi - w_lo))

        hist, _ = np.histogram(slab, bins=21, range=(lat_lo, lat_hi))
        if hist.sum() == 0:
            closures.append(0.0)
            continue
        occupied = hist > 0.002 * hist.sum()
        closures.append(float(occupied[7:14].mean()))  # central third

    scores = np.zeros(2)
    if np.isfinite(widths).all():
        # Narrower end scores higher, normalised by the overall lateral span.
        narrowing = (widths[1] - widths[0]) / lat_span
        scores[0] += narrowing
        scores[1] -= narrowing
    scores[0] += closures[0] - closures[1]
    scores[1] += closures[1] - closures[0]
    return scores


def canonicalize(upper: np.ndarray, lower: np.ndarray) -> CanonicalFrame:
    """Derive the canonical frame from an upper/lower arch point pair."""
    upper = np.asarray(upper, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)

    # --- Superior axis: the arches separate along it. ---
    separation = np.median(upper, axis=0) - np.median(lower, axis=0)
    superior_axis = int(np.argmax(np.abs(separation)))
    superior_sign = 1 if separation[superior_axis] >= 0 else -1

    # --- Anterior axis: of the two remaining axes, the one along which the arch tapers. ---
    # Measured on the mandible alone; see _axis_closure_score for why the maxilla is unusable.
    remaining = [a for a in range(3) if a != superior_axis]
    best_axis, best_sign, best_margin = remaining[0], 1, -np.inf
    for axis in remaining:
        lateral = [a for a in remaining if a != axis][0]
        low_score, high_score = _axis_closure_score(lower, axis, lateral)
        margin = abs(high_score - low_score)
        if margin > best_margin:
            best_margin = margin
            best_axis = axis
            best_sign = 1 if high_score > low_score else -1

    anterior_axis, anterior_sign = best_axis, best_sign
    lateral_axis = [a for a in range(3) if a not in (superior_axis, anterior_axis)][0]

    sup = np.zeros(3)
    sup[superior_axis] = superior_sign
    ant = np.zeros(3)
    ant[anterior_axis] = anterior_sign
    # Right-handed: right = anterior x superior.
    right = np.cross(ant, sup)
    if abs(right[lateral_axis]) < 0.5:  # numerical guard; should never trigger
        right = np.zeros(3)
        right[lateral_axis] = 1.0

    rotation = np.vstack([right, ant, sup])
    return CanonicalFrame(
        rotation=rotation,
        superior_axis=superior_axis,
        anterior_axis=anterior_axis,
        superior_sign=superior_sign,
        anterior_sign=anterior_sign,
        arch_separation_mm=float(abs(separation[superior_axis])),
        anterior_confidence=float(best_margin),
    )
