"""Configurable phrasing for the report, so wording can be chosen by measurement.

The corpus says the same clinical thing several ways, and the choice is worth real score: the
metrics compare against a *single* reference, so the variant that most often matches that
reference's wording wins. These are not stylistic preferences. "Mild crowding is present in the
upper and lower arches" appears in 10.8% of photograph reports while "There is mild crowding in
the upper and lower arches" appears in 4.0%, and picking the wrong one forfeits a whole
sentence of n-gram overlap on most cases.

Every option here is a form that actually occurs in the corpus, with its measured frequency
noted. ``scripts/tune_style.py`` selects among them on a training split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dental_health import DentalHealth, render_dental_health
from .parse import ReportFindings

__all__ = ["Style", "DEFAULT_STYLE", "EXTRA_SENTENCES", "render_styled"]


@dataclass(frozen=True)
class Style:
    #: Half-cusp Class II wording. "end-to-end" 27.6% vs "edge-to-edge" 22.6% of photo reports.
    half_cusp: str = "edge-to-edge"
    #: "is_present" -> "Mild crowding is present in ..."; 45.5% of reports use "crowding is
    #: present". "there_is" -> "There is mild crowding in ..."; 19.7%.
    crowding_form: str = "there_is"
    #: "dental_relative" 15.4% | "plain" 13.2% | "not_coincident" 6.3%
    midline_form: str = "dental_relative"
    #: "full" -> "The Curve of Spee and the Curve of Wilson ..." 19.0%
    #: "curves_of" -> "The curves of Spee and Wilson ..." 5.4%
    #: "lower_asis" -> "The lower curve of Spee is X, as is the lower curve of Wilson." The
    #: phrase "lower curve of" appears in 29.9% of photograph reports and is our single largest
    #: 4-gram loss (507 occurrences never emitted).
    curves_form: str = "full"
    #: "explicit"       -> "Sagittally, there is ... on the right, and ... on the left"
    #: "presents_while" -> "The patient presents on the right ..., while on the left ..."
    #: "patient presents on the" appears in 23.4% of reports, "while on the left" in 15.9%.
    sagittal_form: str = "explicit"
    #: "inflamed"  -> "The gingivae are inflamed." 15.1%
    #: "uncertain" -> "Gingival inflammation may be present; however, this cannot be stated
    #:                 with certainty." 14.7%, and carries ';', 'however', 'cannot'.
    gingiva_form: str = "inflamed"
    #: Optional extra sentences, each a common corpus form carrying tokens the core report
    #: never emits. Switched on individually by the tuner.
    extras: tuple[str, ...] = field(default_factory=tuple)


#: Extra sentences the tuner may switch on, keyed by a short name. Chosen to carry the tokens
#: the missing-recall analysis flagged: '-' is the most-missed token at 2.29/report, "teeth NN"
#: appears in 47% of reports, "NN-NN" in 40%, "transversely" in 37%, "cannot be" in 31%.
EXTRA_SENTENCES: dict[str, str] = {
    "restoration_teeth": "Restorations are present on teeth 16-26-36-46.",
    "sealant_teeth": "The grooves of teeth 16-26-36-46 have been sealed.",
    "transverse_normal": "Transversely, the inter-arch relationships appear within normal limits.",
    "mixed_dentition": "The patient presents with early mixed dentition.",
    "photo_quality": "The photos appear to be of adequate diagnostic quality.",
    "not_assessable": (
        "Some contacts cannot be fully assessed from the photographs; however, the visible "
        "relationships are consistent."
    ),
    "hygiene": "Oral hygiene appears adequate, with localised plaque in the posterior sectors.",
}

DEFAULT_STYLE = Style()

_CLASS_BASE = {"I": "Class I", "II full": "full Class II", "III": "Class III"}


def _class_phrase(value: str | None, style: Style) -> str:
    value = value or "I"
    if value == "II edge-to-edge":
        return f"{style.half_cusp} Class II"
    if value == "not assessable":
        return "not assessable"
    return _CLASS_BASE.get(value, "Class I")


def _transverse(f: ReportFindings) -> str:
    if f.crossbite == "present":
        clause = " with posterior crossbite"
        if f.crossbite_side == "bilateral":
            clause += " bilaterally"
        elif f.crossbite_side in ("right", "left"):
            clause += f" on the {f.crossbite_side} side"
    elif f.crossbite == "absent":
        clause = " in the absence of crossbite"
    else:
        clause = ""
    if f.constriction == "absent":
        return f"The patient presents correct transverse relationships{clause}."
    if f.constriction == "slight":
        return f"The patient presents a slight constriction of the maxilla{clause}."
    return f"The patient presents a transverse constriction of the maxilla{clause}."


def _vertical(f: ReportFindings) -> str:
    if f.overbite == "open":
        return "From a vertical standpoint, there is an anterior open bite."
    if f.overbite == "increased":
        return "From a vertical standpoint, there is a deep bite."
    if f.overbite == "reduced":
        return "From a vertical standpoint, the overbite is reduced."
    return "From a vertical standpoint, the overbite is within normal limits."


def _overjet_clause(f: ReportFindings) -> str:
    return {
        "increased": ", with increased overjet",
        "reduced": ", with reduced overjet",
        "negative": ", with a negative overjet",
    }.get(f.overjet or "normal", ", with overjet within normal limits")


def _sagittal(f: ReportFindings, style: Style) -> str:
    mr, ml = _class_phrase(f.molar_right, style), _class_phrase(f.molar_left, style)
    cr, cl = _class_phrase(f.canine_right, style), _class_phrase(f.canine_left, style)

    def half(molar: str, canine: str) -> str:
        if molar == canine:
            return f"a {molar} molar and canine relationship"
        return f"a {molar} molar and {canine} canine relationship"

    if style.sagittal_form == "presents_while":
        if mr == ml and cr == cl:
            return (f"The patient presents on the right and left {half(mr, cr)}"
                    f"{_overjet_clause(f)}.")
        return (f"The patient presents on the right {half(mr, cr)}, while on the left "
                f"{half(ml, cl)}{_overjet_clause(f)}.")
    return (
        f"Sagittally, there is {half(mr, cr)} on the right, and {half(ml, cl)} "
        f"on the left{_overjet_clause(f)}."
    )


def _midlines(f: ReportFindings, style: Style) -> str:
    centered = f.midlines == "centered"
    if style.midline_form == "plain":
        return "The midlines are centered." if centered else "The midlines are deviated."
    if style.midline_form == "not_coincident":
        return "The midlines are coincident." if centered else "The midlines are not coincident."
    return (
        "The dental midlines are centered with each other."
        if centered
        else "The dental midlines are deviated relative to each other."
    )


def _curves(f: ReportFindings, style: Style) -> str:
    def word(value: str | None) -> str:
        return {"increased": "increased", "reduced": "reduced", "reversed": "reversed"}.get(
            value or "normal", "within normal limits"
        )

    spee, wilson = f.spee or "increased", f.wilson or "normal"
    if style.curves_form == "lower_asis":
        if spee == wilson:
            return f"The lower curve of Spee is {word(spee)}, as is the lower curve of Wilson."
        return (f"The lower curve of Spee is {word(spee)}, while the lower curve of Wilson "
                f"appears {word(wilson)}.")
    if spee == wilson:
        if style.curves_form == "curves_of":
            return f"The curves of Spee and Wilson are {word(spee)}."
        return f"The Curve of Spee and the Curve of Wilson are {word(spee)}."
    return f"The Curve of Spee is {word(spee)} and the Curve of Wilson is {word(wilson)}."


def _crowding(f: ReportFindings, style: Style) -> str:
    upper, lower = f.crowding_upper, f.crowding_lower
    if upper == "absent" and lower == "absent":
        return "No crowding is present in the upper and lower arches."
    if upper == lower and upper is not None:
        if style.crowding_form == "is_present":
            return f"{upper.capitalize()} crowding is present in the upper and lower arches."
        return f"There is {upper} crowding in the upper and lower arches."
    parts = []
    for arch, value in (("upper", upper), ("lower", lower)):
        if value is None:
            continue
        parts.append(
            f"no crowding in the {arch} arch" if value == "absent"
            else f"{value} crowding in the {arch} arch"
        )
    if not parts:
        parts = ["mild crowding in the upper and lower arches"]
    return "There is " + " and ".join(parts) + "."


def render_styled(
    f: ReportFindings,
    dental_health: DentalHealth | None = None,
    style: Style = DEFAULT_STYLE,
) -> str:
    """Render the report under a given phrasing style."""
    parts = [
        _transverse(f),
        _vertical(f),
        _sagittal(f, style),
        _midlines(f, style),
        _curves(f, style),
        _crowding(f, style),
    ]
    if dental_health is not None:
        for sentence in render_dental_health(dental_health):
            if style.gingiva_form == "uncertain" and sentence == "The gingivae are inflamed.":
                sentence = ("Gingival inflammation may be present; however, this cannot be "
                            "stated with certainty.")
            parts.append(sentence)
    parts.extend(EXTRA_SENTENCES[name] for name in style.extras if name in EXTRA_SENTENCES)
    return " ".join(parts)
