"""Render structured findings as a Bite2Text-style clinical narrative.

Bite2Text reports follow a fixed dictation order — transverse, vertical, sagittal, midlines,
occlusal curves, crowding — and reuse a small set of stock phrasings. Reproducing that idiom
is worth real score on both halves of the objective: shared 4-grams drive BLEU-4 directly, and
covering all six topics is what RadFact recall measures.

Wording here is transcribed from the corpus rather than invented. ``scripts/tune_phrasing.py``
selects among the observed variants by measured score.
"""

from __future__ import annotations

from .parse import ReportFindings

__all__ = ["render_report", "render_modal_report", "MODAL_FINDINGS"]

#: Corpus phrasing for each occlusion class, as it appears inside a relationship clause.
_CLASS_PHRASE = {
    "I": "Class I",
    "II edge-to-edge": "edge-to-edge Class II",
    "II full": "full Class II",
    "III": "Class III",
    "not assessable": "not assessable",
}


def _transverse_sentence(f: ReportFindings) -> str:
    crossbite_clause = ""
    if f.crossbite == "present":
        if f.crossbite_teeth:
            teeth = f.crossbite_teeth
            listed = str(teeth[0]) if len(teeth) == 1 else ", ".join(str(t) for t in teeth[:-1]) + f" and {teeth[-1]}"
            noun = "tooth" if len(teeth) == 1 else "teeth"
            crossbite_clause = f" with crossbite on {noun} {listed}"
        else:
            side = {
                "right": " on the right side",
                "left": " on the left side",
                "bilateral": " bilaterally",
            }.get(f.crossbite_side or "", "")
            crossbite_clause = f" with posterior crossbite{side}"
    elif f.crossbite == "absent":
        crossbite_clause = " in the absence of crossbite"

    if f.constriction in ("present", None):
        return f"The patient presents a transverse constriction of the maxilla{crossbite_clause}."
    if f.constriction == "slight":
        return f"The patient presents a slight constriction of the maxilla{crossbite_clause}."
    # constriction == "absent"
    if crossbite_clause:
        return f"The patient presents correct transverse relationships{crossbite_clause}."
    return "The patient presents correct transverse relationships."


def _vertical_sentence(f: ReportFindings) -> str:
    if f.overbite == "open":
        return "From a vertical standpoint, there is an anterior open bite."
    if f.overbite == "increased":
        return "From a vertical standpoint, there is a deep bite."
    if f.overbite == "reduced":
        return "From a vertical standpoint, the overbite is reduced."
    return "From a vertical standpoint, the overbite is within normal limits."


def _overjet_clause(f: ReportFindings) -> str:
    if f.overjet == "increased":
        return ", with increased overjet"
    if f.overjet == "reduced":
        return ", with reduced overjet"
    if f.overjet == "negative":
        return ", with a negative overjet"
    return ", with overjet within normal limits"


def _sagittal_sentence(f: ReportFindings) -> str:
    mr = _CLASS_PHRASE.get(f.molar_right or "I", "Class I")
    ml = _CLASS_PHRASE.get(f.molar_left or "I", "Class I")
    cr = _CLASS_PHRASE.get(f.canine_right or "I", "Class I")
    cl = _CLASS_PHRASE.get(f.canine_left or "I", "Class I")
    overjet = _overjet_clause(f)

    # Both sides are always stated explicitly, even when they agree. 42% of the corpus uses
    # the shorter "bilaterally" form, but measured against real references the explicit form
    # scores higher (+0.012 captioning): it shares more 4-grams with the majority phrasing,
    # and METEOR-lite weights recall 9:1, so covering more reference tokens pays.
    def half(molar: str, canine: str) -> str:
        if molar == canine:
            return f"a {molar} molar and canine relationship"
        return f"a {molar} molar and {canine} canine relationship"

    return (
        f"Sagittally, there is {half(mr, cr)} on the right, and {half(ml, cl)} "
        f"on the left{overjet}."
    )


def _midline_sentence(f: ReportFindings) -> str:
    if f.midlines == "centered":
        return "The dental midlines are centered with each other."
    return "The dental midlines are deviated relative to each other."


def _curves_sentence(f: ReportFindings) -> str:
    spee = f.spee or "increased"
    wilson = f.wilson or "normal"
    if spee == wilson:
        state = "increased" if spee == "increased" else "within normal limits"
        if spee == "reduced":
            state = "reduced"
        elif spee == "reversed":
            state = "reversed"
        return f"The Curve of Spee and the Curve of Wilson are {state}."

    def phrase(value: str) -> str:
        return {
            "increased": "increased",
            "normal": "within normal limits",
            "reduced": "reduced",
            "reversed": "reversed",
        }.get(value, "within normal limits")

    return f"The Curve of Spee is {phrase(spee)} and the Curve of Wilson is {phrase(wilson)}."


def _crowding_sentence(f: ReportFindings) -> str:
    upper = f.crowding_upper
    lower = f.crowding_lower
    spacing_clause = ", with the presence of spaces" if f.spacing else ""

    if upper == "absent" and lower == "absent":
        return f"No crowding is present in the upper and lower arches{spacing_clause}."
    if upper == lower and upper is not None:
        return f"There is {upper} crowding in the upper and lower arches{spacing_clause}."

    parts = []
    for arch, value in (("upper", upper), ("lower", lower)):
        if value is None:
            continue
        if value == "absent":
            parts.append(f"no crowding in the {arch} arch")
        else:
            parts.append(f"{value} crowding in the {arch} arch")
    if not parts:
        return f"There is mild crowding in the upper and lower arches{spacing_clause}."
    return "There is " + " and ".join(parts) + spacing_clause + "."


def render_report(f: ReportFindings, dental_health: "DentalHealth | None" = None) -> str:
    """Compose the report: the six-part occlusal narrative, then photograph findings.

    ``dental_health`` is optional because the two report families differ. Intraoral-scan
    reports stop after the occlusal section; photograph reports continue for another ~3.6
    sentences about gingival status, restorations and caries. Pass the findings to include
    that section — see ``bite2text.report.dental_health``.
    """
    parts = [
        _transverse_sentence(f),
        _vertical_sentence(f),
        _sagittal_sentence(f),
        _midline_sentence(f),
        _curves_sentence(f),
        _crowding_sentence(f),
    ]
    if dental_health is not None:
        from .dental_health import render_dental_health

        parts.extend(render_dental_health(dental_health))
    return " ".join(parts)


#: Modal values measured over the 1,496 parsed IOS reports. Used as the uninformed fallback
#: and as the prior baseline in ``scripts/baseline_experiment.py``.
MODAL_FINDINGS = ReportFindings(
    constriction="present",
    crossbite="absent",
    overbite="increased",
    overjet="increased",
    molar_right="I",
    molar_left="I",
    canine_right="I",
    canine_left="I",
    midlines="deviated",
    spee="increased",
    wilson="increased",
    crowding_upper="mild",
    crowding_lower="mild",
)


def render_modal_report() -> str:
    """The report a system would emit knowing nothing about the individual patient."""
    return render_report(MODAL_FINDINGS)
