"""Tests for the geometry engine, report parser/renderer, and the scorer replica.

The geometry tests build synthetic arches rather than using patient data, so they run
anywhere and assert the properties the whole pipeline depends on: that the canonical frame is
recovered from geometry alone (including when the source frame is rotated or mirrored), and
that overbite/overjet carry the sign convention the renderer assumes.
"""

from __future__ import annotations

import numpy as np
import pytest

from bite2text.eval.gc_metrics import bleu_4_local, meteor_lite_batch, score_final, tokenize
from bite2text.geom.canonical import canonicalize
from bite2text.report.parse import parse_report
from bite2text.report.render import render_modal_report, render_report


def synthetic_arch(jaw: str, *, n: int = 40000, seed: int = 0) -> np.ndarray:
    """A U-shaped dental arch in (right, anterior, superior) millimetres.

    Anterior is +y and narrow; posterior is -y and wide. The occlusal surface faces the
    opposing arch, and the two arches are separated so that they overlap slightly, as
    scans supplied in occlusion do.
    """
    rng = np.random.default_rng(seed)
    # Parameterise the arch curve by angle about a pole behind the incisors.
    phi = rng.uniform(-np.deg2rad(135), np.deg2rad(135), n)
    radius = 22.0 + 6.0 * (np.abs(phi) / np.deg2rad(135)) ** 2
    x = radius * np.sin(phi)
    y = radius * np.cos(phi) - 4.0

    height = rng.uniform(0.0, 8.0, n)  # crown height, gingiva to occlusal surface
    if jaw == "lower":
        z = -8.0 + height  # occlusal surface near z = 0, pointing up
    else:
        z = 8.0 - height  # occlusal surface near z = 0, pointing down
    return np.stack([x, y, z], axis=1)


@pytest.fixture(scope="module")
def arches() -> tuple[np.ndarray, np.ndarray]:
    return synthetic_arch("upper", seed=1), synthetic_arch("lower", seed=2)


def test_canonicalize_recovers_frame_from_geometry(arches):
    upper, lower = arches
    frame = canonicalize(upper, lower)
    assert frame.superior_axis == 2 and frame.superior_sign == 1
    assert frame.anterior_axis == 1 and frame.anterior_sign == 1
    assert frame.anterior_confidence > 0.5


@pytest.mark.parametrize(
    "rotation,label",
    [
        (np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float), "anterior flipped"),
        (np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float), "superior on y"),
        (np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float), "axes swapped"),
    ],
)
def test_canonicalize_is_invariant_to_source_frame(arches, rotation, label):
    """Re-deriving the frame must undo whatever convention the file was saved in.

    The released datasets genuinely disagree here — Bits2Bites is anterior=+y while Bite2Text
    is anterior=-y — so this invariance is what keeps the measurements meaningful.
    """
    upper, lower = arches
    ru, rl = upper @ rotation.T, lower @ rotation.T
    frame = canonicalize(ru, rl)
    canon_upper, canon_lower = frame.apply(ru), frame.apply(rl)

    # After canonicalisation the upper arch is superior and the arch tapers anteriorly.
    assert np.median(canon_upper[:, 2]) > np.median(canon_lower[:, 2]), label
    front = canon_lower[canon_lower[:, 1] > np.quantile(canon_lower[:, 1], 0.80)]
    back = canon_lower[canon_lower[:, 1] < np.quantile(canon_lower[:, 1], 0.20)]
    assert np.ptp(front[:, 0]) < np.ptp(back[:, 0]), label


def test_parse_round_trips_a_representative_report():
    text = (
        "The patient presents a transverse constriction of the maxilla in the absence of "
        "crossbite. From a vertical standpoint, there is a deep bite. Sagittally, there is a "
        "full Class II molar and canine relationship on the right, and a Class I molar and "
        "edge-to-edge Class II canine relationship on the left, with increased overjet. The "
        "dental midlines are deviated relative to each other. The Curve of Spee and the Curve "
        "of Wilson are increased. There is mild crowding in the upper arch and moderate "
        "crowding in the lower arch."
    )
    f = parse_report(text)
    assert f.constriction == "present"
    assert f.crossbite == "absent"
    assert f.overbite == "increased"
    assert f.overjet == "increased"
    assert f.molar_right == "II full"
    assert f.molar_left == "I"
    assert f.canine_left == "II edge-to-edge"
    assert f.midlines == "deviated"
    assert f.spee == "increased" and f.wilson == "increased"
    assert f.crowding_upper == "mild" and f.crowding_lower == "moderate"


def test_render_covers_every_reporting_topic():
    report = render_modal_report()
    for topic in ("transverse", "vertical", "Sagittally", "midlines", "Curve of Spee", "crowding"):
        assert topic in report, topic
    assert report.count(".") >= 6


def test_render_states_both_sides_explicitly():
    """Both sides are named even when symmetric; this measured +0.012 captioning."""
    text = "Sagittally, there is a Class I molar and canine relationship bilaterally."
    rendered = render_report(parse_report(text))
    assert "on the right" in rendered and "on the left" in rendered


def test_meteor_lite_is_recall_weighted():
    """METEOR-lite's f-mean is 10PR/(R+9P), so recall dominates precision 9:1."""
    reference = ["the " + " ".join(f"w{i}" for i in range(20))]
    high_recall_low_precision = [" ".join(f"w{i}" for i in range(20)) + " " + " ".join(["pad"] * 20)]
    low_recall_high_precision = [" ".join(f"w{i}" for i in range(5))]
    assert meteor_lite_batch(high_recall_low_precision, reference) > meteor_lite_batch(
        low_recall_high_precision, reference
    )


def test_bleu_and_final_score_arithmetic():
    assert bleu_4_local(["a b c d e"], ["a b c d e"]) == pytest.approx(1.0, abs=1e-6)
    assert tokenize("Class II, full.") == ["class", "ii", ",", "full", "."]
    assert score_final(0.5, 0.6) == pytest.approx(0.8 * 0.5 + 0.2 * 0.6)


def test_dental_health_round_trip():
    """Photograph findings parse out of the corpus's own phrasings."""
    from bite2text.report.dental_health import parse_dental_health

    f = parse_dental_health(
        "The gingivae are inflamed. Restorations are present on teeth 36-46. "
        "No evident active carious processes are noted. "
        "Pit-and-fissure sealants are present on the upper and lower first molars. "
        "Plaque/tartar is present in the posterior sectors."
    )
    assert f.gingival_inflammation is True
    assert f.restorations is True
    assert f.caries is False
    assert f.sealants is True
    assert f.plaque is True


def test_dental_health_reads_negations():
    from bite2text.report.dental_health import parse_dental_health

    f = parse_dental_health(
        "No clear signs of gingival inflammation are observed. "
        "No restorative treatments or evident ongoing carious processes are present."
    )
    assert f.gingival_inflammation is False
    assert f.restorations is False


def test_render_appends_dental_health_only_when_given():
    """IOS reports stop after the occlusal section; photograph reports continue."""
    from bite2text.compose import DEFAULT_DENTAL_HEALTH
    from bite2text.report.render import MODAL_FINDINGS

    occlusal_only = render_report(MODAL_FINDINGS)
    with_photos = render_report(MODAL_FINDINGS, DEFAULT_DENTAL_HEALTH)
    assert "gingiv" not in occlusal_only.lower()
    assert "gingiv" in with_photos.lower()
    assert with_photos.startswith(occlusal_only)
    # Length should land near the photograph corpus median of 146 tokens.
    assert 120 < len(tokenize(with_photos)) < 175
