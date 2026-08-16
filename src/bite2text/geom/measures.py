"""Derive orthodontic measurements from a registered arch pair.

Everything here is deterministic geometry. The scans are supplied *in occlusion* and
mutually registered, which means the quantities orthodontists actually report — overbite,
overjet, midline deviation, transverse relationships, the curve of Spee — are directly
measurable rather than something a model has to guess from appearance. That is the main
advantage this pipeline has over a photograph-only vision-language baseline.

Measurements are returned in millimetres and degrees with their sign conventions stated.
Categorical clinical labels are *not* assigned here: thresholds are fitted against the
Bits2Bites annotations in ``scripts/calibrate_geometry.py`` so that the cut-points are
evidence-based rather than textbook guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage, signal

from .arch import ArchProfile, CaseGeometry

__all__ = ["CaseMeasurements", "measure_case"]

ANTERIOR_WINDOW_DEG = 14.0
CANINE_WINDOW_DEG = (18.0, 38.0)
PREMOLAR_WINDOW_DEG = (38.0, 70.0)
MOLAR_WINDOW_DEG = (70.0, 115.0)
POSTERIOR_WINDOW_DEG = (35.0, 115.0)


def _smooth(values: np.ndarray, valid: np.ndarray, sigma_bins: float = 1.5) -> np.ndarray:
    """Gaussian-smooth a profile, bridging invalid bins by nearest-valid fill."""
    out = np.array(values, dtype=np.float64)
    if not valid.any():
        return out
    idx = np.arange(len(out))
    out = np.interp(idx, idx[valid], out[valid])
    return ndimage.gaussian_filter1d(out, sigma=sigma_bins, mode="nearest")


def _nan_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _incisal_edge(profile: ArchProfile) -> tuple[float, float, float]:
    """Return (z, y, phi) of the incisal edge region of one arch at the midline."""
    mask = profile.window(-ANTERIOR_WINDOW_DEG, ANTERIOR_WINDOW_DEG)
    if not mask.any():
        return (float("nan"),) * 3
    z = profile.z_occlusal[mask]
    y = profile.y_ridge[mask]
    phi = profile.phi[mask]
    # The incisal edge is the occlusal extreme of the anterior segment.
    pick = np.nanargmax(z) if profile.jaw == "lower" else np.nanargmin(z)
    return float(z[pick]), float(y[pick]), float(phi[pick])


def _dental_midline_x(profile: ArchProfile) -> float:
    """Locate the inter-incisal embrasure and return its x coordinate (mm).

    Two independent cues are combined: the small vertical notch between the central
    incisors (a dip in the occlusal height) and the corresponding recess in the labial
    outline (a dip in buccal radius).
    """
    mask = profile.window(-16.0, 16.0)
    if mask.sum() < 8:
        return float("nan")

    phi = profile.phi
    z = _smooth(profile.z_occlusal, profile.valid, sigma_bins=1.2)
    r = _smooth(profile.r_buccal, profile.valid, sigma_bins=1.2)

    # Notch = local minimum of occlusal height for the lower arch (edges point up),
    # local maximum for the upper arch (edges point down); the labial recess is a
    # minimum of radius for both.
    z_signal = -z if profile.jaw == "lower" else z
    candidates = []
    for sig in (z_signal, -r):
        seg = sig.copy()
        seg[~mask] = -np.inf
        peaks, props = signal.find_peaks(seg, prominence=0.02)
        if peaks.size:
            best = peaks[np.argmax(props["prominences"])]
            candidates.append(float(phi[best]))

    if not candidates:
        return float("nan")
    phi_mid = float(np.median(candidates))

    # Convert the angular position back to an x coordinate on the ridge.
    idx = int(np.argmin(np.abs(phi - phi_mid)))
    x = profile.x_ridge[idx]
    if not np.isfinite(x):
        near = profile.valid & (np.abs(phi - phi_mid) < 4.0)
        x = _nan_mean(profile.x_ridge[near])
    return float(x)


def _curve_of_spee(lower: ArchProfile, side: int) -> float:
    """Depth (mm) of the curve of Spee on one side; 0 is a flat plane.

    ``side`` is +1 for the patient's right, -1 for the left.
    """
    lo, hi = (5.0, MOLAR_WINDOW_DEG[1])
    mask = lower.window(lo, hi) if side > 0 else lower.window(-hi, -lo)
    if mask.sum() < 12:
        return float("nan")

    phi = lower.phi[mask]
    z = lower.z_occlusal[mask]
    # Arc position along the ridge, so depth is measured against a real chord.
    r = lower.r_ridge[mask]
    good = np.isfinite(z) & np.isfinite(r)
    if good.sum() < 12:
        return float("nan")
    phi, z, r = phi[good], z[good], r[good]
    s = np.radians(np.abs(phi)) * np.nanmedian(r)

    order = np.argsort(s)
    s, z = s[order], z[order]
    # Chord from the incisal end to the most distal cusp.
    chord = np.interp(s, [s[0], s[-1]], [z[0], z[-1]])
    return float(np.nanmax(chord - z))


def _curve_of_wilson(lower: ArchProfile, side: int) -> float:
    """Buccal-minus-lingual cusp height (mm) across the posterior segment."""
    lo, hi = POSTERIOR_WINDOW_DEG
    mask = lower.window(lo, hi) if side > 0 else lower.window(-hi, -lo)
    if mask.sum() < 8:
        return float("nan")
    # r_buccal/r_lingual bound the occlusal band; their radial spread stands in for the
    # bucco-lingual tilt of the posterior segment.
    return float(np.nanmedian(lower.r_buccal[mask] - lower.r_lingual[mask]))


def _transverse_series(geom: CaseGeometry) -> dict[str, np.ndarray]:
    """Per-angle transverse relationship between the arches.

    ``overlap`` is how far the upper arch sits buccal to the lower one. Positive is the
    normal relationship; negative means the upper is lingual to the lower — a crossbite.
    ``scissor`` is positive when the upper arch clears the lower one entirely buccally.
    """
    upper, lower = geom.upper, geom.lower
    both = upper.valid & lower.valid
    overlap = np.where(both, upper.r_buccal - lower.r_buccal, np.nan)
    scissor = np.where(both, upper.r_lingual - lower.r_buccal, np.nan)
    return {"phi": lower.phi, "valid": both, "overlap": overlap, "scissor": scissor}


def _side_mask(phi: np.ndarray, valid: np.ndarray, side: int, window: tuple[float, float]) -> np.ndarray:
    lo, hi = window
    if side > 0:
        return valid & (phi >= lo) & (phi <= hi)
    return valid & (phi <= -lo) & (phi >= -hi)


def _cusp_lag_deg(geom: CaseGeometry, side: int) -> float:
    """Angular offset that best aligns the upper and lower cusp patterns on one side.

    Interdigitation shifts mesiodistally with the sagittal (Angle) class, so the lag is a
    direct, segmentation-free proxy for it. Positive lag means the lower cusp pattern sits
    distal to the upper one.
    """
    lo, hi = (20.0, MOLAR_WINDOW_DEG[1])
    window = (lo, hi)
    up_mask = _side_mask(geom.upper.phi, geom.upper.valid, side, window)
    lo_mask = _side_mask(geom.lower.phi, geom.lower.valid, side, window)
    common = up_mask & lo_mask
    if common.sum() < 20:
        return float("nan")

    # Cusp pattern = occlusal height with the smooth arch trend removed.
    def detrend(profile: ArchProfile) -> np.ndarray:
        z = _smooth(profile.z_occlusal, profile.valid, sigma_bins=1.0)[common]
        trend = ndimage.gaussian_filter1d(z, sigma=8.0, mode="nearest")
        d = z - trend
        sd = d.std()
        return d / sd if sd > 1e-9 else d

    up = detrend(geom.upper)
    lw = detrend(geom.lower)
    if up.size < 20:
        return float("nan")

    max_lag = min(18, up.size // 3)
    lags = np.arange(-max_lag, max_lag + 1)
    scores = [float(np.dot(up, np.roll(lw, int(k)))) for k in lags]
    best = lags[int(np.argmax(scores))]
    return float(best * (geom.lower.phi[1] - geom.lower.phi[0])) * (1 if side > 0 else -1)


def _ap_offset(geom: CaseGeometry, side: int, window: tuple[float, float]) -> float:
    """Mean anteroposterior (y) offset between the arches over an angular window (mm)."""
    up_mask = _side_mask(geom.upper.phi, geom.upper.valid, side, window)
    lo_mask = _side_mask(geom.lower.phi, geom.lower.valid, side, window)
    common = up_mask & lo_mask
    if common.sum() < 5:
        return float("nan")
    return _nan_mean(geom.upper.y_ridge[common] - geom.lower.y_ridge[common])


def _arch_width(profile: ArchProfile, window: tuple[float, float]) -> float:
    """Inter-arch width (mm) across a symmetric angular window."""
    right = _side_mask(profile.phi, profile.valid, +1, window)
    left = _side_mask(profile.phi, profile.valid, -1, window)
    if not right.any() or not left.any():
        return float("nan")
    return float(np.nanmedian(profile.r_buccal[right]) + np.nanmedian(profile.r_buccal[left]))


def _anterior_irregularity(profile: ArchProfile) -> float:
    """Little-style irregularity of the anterior segment (mm).

    Sums the absolute deviation of the incisal ridge from its own smooth arch form; a
    well-aligned segment is near zero, crowding or rotation raises it.
    """
    mask = profile.window(-38.0, 38.0)
    if mask.sum() < 20:
        return float("nan")
    r = profile.r_buccal[mask]
    good = np.isfinite(r)
    if good.sum() < 20:
        return float("nan")
    r = r[good]
    smooth = ndimage.gaussian_filter1d(r, sigma=6.0, mode="nearest")
    return float(np.mean(np.abs(r - smooth)))


#: Clinically possible ranges (mm). A value outside these means landmark detection failed on
#: that scan, not that the patient is extraordinary. Such values are returned as NaN so the
#: downstream models treat the measurement as *missing* rather than learning from a wrong
#: number — gradient boosting handles NaN natively, so this degrades gracefully per field.
PLAUSIBLE_RANGE_MM = {
    "overbite_mm": (-12.0, 14.0),
    "overjet_mm": (-12.0, 16.0),
    "midline_deviation_mm": (-12.0, 12.0),
    "curve_of_spee_right_mm": (0.0, 12.0),
    "curve_of_spee_left_mm": (0.0, 12.0),
}


def _guard(name: str, value: float) -> float:
    lo, hi = PLAUSIBLE_RANGE_MM.get(name, (-np.inf, np.inf))
    if not np.isfinite(value) or value < lo or value > hi:
        return float("nan")
    return float(value)


@dataclass
class CaseMeasurements:
    """Named clinical measurements plus a flat feature vector for calibration."""

    case_id: str
    overbite_mm: float
    overjet_mm: float
    midline_deviation_mm: float
    upper_midline_x: float
    lower_midline_x: float
    curve_of_spee_right_mm: float
    curve_of_spee_left_mm: float
    curve_of_wilson_right_mm: float
    curve_of_wilson_left_mm: float
    transverse_min_overlap_right_mm: float
    transverse_min_overlap_left_mm: float
    crossbite_extent_right_deg: float
    crossbite_extent_left_deg: float
    scissor_extent_right_deg: float
    scissor_extent_left_deg: float
    cusp_lag_right_deg: float
    cusp_lag_left_deg: float
    upper_intermolar_width_mm: float
    lower_intermolar_width_mm: float
    upper_intercanine_width_mm: float
    lower_intercanine_width_mm: float
    upper_anterior_irregularity_mm: float
    lower_anterior_irregularity_mm: float
    frame_anterior_confidence: float
    arch_separation_mm: float
    extra: dict[str, float] = field(default_factory=dict)

    def features(self) -> dict[str, float]:
        out = {
            k: float(v)
            for k, v in self.__dict__.items()
            if k not in ("case_id", "extra") and isinstance(v, (int, float))
        }
        out.update({k: float(v) for k, v in self.extra.items()})
        return out


def measure_case(geom: CaseGeometry) -> CaseMeasurements:
    """Compute the full measurement set for one registered arch pair."""
    upper, lower = geom.upper, geom.lower

    up_z, up_y, _ = _incisal_edge(upper)
    lo_z, lo_y, _ = _incisal_edge(lower)
    # Overbite: how far the upper incisor descends past the lower incisal edge.
    overbite = lo_z - up_z
    # Overjet: how far the upper incisor sits anterior to the lower one.
    overjet = up_y - lo_y

    up_mid = _dental_midline_x(upper)
    lo_mid = _dental_midline_x(lower)
    midline_dev = up_mid - lo_mid

    tv = _transverse_series(geom)
    step = float(lower.phi[1] - lower.phi[0])

    def side_transverse(side: int) -> tuple[float, float, float]:
        m = _side_mask(tv["phi"], tv["valid"], side, POSTERIOR_WINDOW_DEG)
        if m.sum() < 5:
            return float("nan"), float("nan"), float("nan")
        overlap = tv["overlap"][m]
        scissor = tv["scissor"][m]
        min_overlap = float(np.nanmin(overlap))
        cross_extent = float(np.sum(overlap < 0) * step)
        scissor_extent = float(np.sum(scissor > 0) * step)
        return min_overlap, cross_extent, scissor_extent

    r_min, r_cross, r_sci = side_transverse(+1)
    l_min, l_cross, l_sci = side_transverse(-1)

    extra: dict[str, float] = {}
    for name, window in (
        ("canine", CANINE_WINDOW_DEG),
        ("premolar", PREMOLAR_WINDOW_DEG),
        ("molar", MOLAR_WINDOW_DEG),
    ):
        extra[f"ap_offset_right_{name}_mm"] = _ap_offset(geom, +1, window)
        extra[f"ap_offset_left_{name}_mm"] = _ap_offset(geom, -1, window)

    return CaseMeasurements(
        case_id=geom.case_id,
        overbite_mm=_guard("overbite_mm", overbite),
        overjet_mm=_guard("overjet_mm", overjet),
        midline_deviation_mm=_guard("midline_deviation_mm", midline_dev),
        upper_midline_x=up_mid,
        lower_midline_x=lo_mid,
        curve_of_spee_right_mm=_guard("curve_of_spee_right_mm", _curve_of_spee(lower, +1)),
        curve_of_spee_left_mm=_guard("curve_of_spee_left_mm", _curve_of_spee(lower, -1)),
        curve_of_wilson_right_mm=_curve_of_wilson(lower, +1),
        curve_of_wilson_left_mm=_curve_of_wilson(lower, -1),
        transverse_min_overlap_right_mm=r_min,
        transverse_min_overlap_left_mm=l_min,
        crossbite_extent_right_deg=r_cross,
        crossbite_extent_left_deg=l_cross,
        scissor_extent_right_deg=r_sci,
        scissor_extent_left_deg=l_sci,
        cusp_lag_right_deg=_cusp_lag_deg(geom, +1),
        cusp_lag_left_deg=_cusp_lag_deg(geom, -1),
        upper_intermolar_width_mm=_arch_width(upper, MOLAR_WINDOW_DEG),
        lower_intermolar_width_mm=_arch_width(lower, MOLAR_WINDOW_DEG),
        upper_intercanine_width_mm=_arch_width(upper, CANINE_WINDOW_DEG),
        lower_intercanine_width_mm=_arch_width(lower, CANINE_WINDOW_DEG),
        upper_anterior_irregularity_mm=_anterior_irregularity(upper),
        lower_anterior_irregularity_mm=_anterior_irregularity(lower),
        frame_anterior_confidence=geom.frame.anterior_confidence,
        arch_separation_mm=geom.frame.arch_separation_mm,
        extra=extra,
    )
