"""Parse Bite2Text free-text reports into structured findings.

The corpus follows a consistent clinical dictation order — transverse, vertical, sagittal,
midlines, occlusal curves, crowding — so a rule-based parser recovers the underlying fields
without an LLM in the loop. That matters for two reasons: it turns 1,496 narrative reports
into supervised labels for the geometry models, and it lets us measure per-field base rates
directly instead of guessing them.

The parser is deliberately conservative. Any field it cannot read confidently is left as
``None`` rather than defaulted, so downstream training can drop uncertain labels instead of
learning from parser mistakes.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

__all__ = ["ReportFindings", "parse_report", "SEVERITY_ORDER", "OCCLUSION_CLASSES"]

SEVERITY_ORDER = ("absent", "mild", "mild-to-moderate", "moderate", "moderate-to-severe", "severe")
OCCLUSION_CLASSES = ("I", "II edge-to-edge", "II full", "III", "not assessable")

_CLASS_III = re.compile(r"class\s*(?:iii|3)\b", re.I)
_CLASS_II = re.compile(r"class\s*(?:ii|2)\b(?!\s*i)", re.I)
_CLASS_I = re.compile(r"class\s*(?:i|1)\b(?!\s*[iv1-9])", re.I)
_EDGE = re.compile(r"edge[-\s]?to[-\s]?edge|end[-\s]?to[-\s]?end|head[-\s]?to[-\s]?head", re.I)
_FULL = re.compile(r"\bfull\b|\bcomplete\b", re.I)
_NOT_ASSESSABLE = re.compile(r"cannot be (?:assessed|evaluated)|not assessable|due to the (?:absence|lack)", re.I)


def _severity(text: str) -> str | None:
    """Read a crowding severity, preferring compound grades over their components."""
    t = text.lower()
    if re.search(r"\bno crowding|absence of crowding|without crowding|no significant crowding", t):
        return "absent"
    for pattern, value in (
        (r"mild[-\s]?(?:to)?[-\s]?moderate", "mild-to-moderate"),
        (r"moderate[-\s]?(?:to)?[-\s]?severe", "moderate-to-severe"),
        (r"\bsevere\b", "severe"),
        (r"\bmoderate\b", "moderate"),
        (r"\bmild\b|\bslight\b", "mild"),
    ):
        if re.search(pattern, t):
            return value
    return None


def _split_sides(sentence: str) -> dict[str, str]:
    """Split a sagittal sentence into right/left/bilateral segments.

    Reports use both word orders — "a full Class II relationship *on the right*" and "*on the
    left side* the molar relationship cannot be assessed" — so a side marker may be either
    preceded or followed by the finding it qualifies. We therefore split the sentence at the
    markers and attach each marker to whichever neighbouring chunk actually carries a class
    statement, preferring the preceding chunk when both do.
    """
    text = sentence
    markers = [
        (m.start(), m.end(), "right" if "right" in m.group(0).lower() else "left")
        for m in re.finditer(r"on the (?:right|left)(?:\s*side)?", text, re.I)
    ]
    if not markers:
        return {"both": text}

    # Chunks between (and around) the markers.
    bounds = [0] + [b for m in markers for b in (m[0], m[1])] + [len(text)]
    chunks: list[str] = []
    for i in range(0, len(bounds) - 1, 2):
        chunks.append(text[bounds[i] : bounds[i + 1]])
    # chunks[i] is the text before markers[i]; chunks[i+1] the text after it.

    def informative(chunk: str) -> bool:
        return bool(chunk.strip(" ,;.")) and _classify_occlusion(chunk) is not None

    segments: dict[str, str] = {}
    claimed: set[int] = set()
    for i, (_, _, side) in enumerate(markers):
        before, after = i, i + 1
        choice = None
        if before not in claimed and informative(chunks[before]):
            choice = before
        elif after < len(chunks) and after not in claimed and informative(chunks[after]):
            choice = after
        elif before not in claimed and chunks[before].strip(" ,;."):
            choice = before
        elif after < len(chunks) and after not in claimed:
            choice = after
        if choice is None:
            continue
        claimed.add(choice)
        segments[side] = chunks[choice]

    # A side never named inherits any unclaimed descriptive chunk, else the whole sentence.
    for side in ("right", "left"):
        if side in segments:
            continue
        spare = [c for i, c in enumerate(chunks) if i not in claimed and informative(c)]
        segments[side] = spare[0] if spare else text
    return segments


def _classify_occlusion(segment: str) -> str | None:
    if _NOT_ASSESSABLE.search(segment):
        return "not assessable"
    if _CLASS_III.search(segment):
        return "III"
    if _CLASS_II.search(segment):
        if _EDGE.search(segment):
            return "II edge-to-edge"
        if _FULL.search(segment):
            return "II full"
        return "II full"
    if _CLASS_I.search(segment):
        return "I"
    return None


_CLASS_MENTION = re.compile(r"class\s*(?:iii|ii|i|3|2|1)\b", re.I)


def _tooth_specific(segment: str, keyword: str) -> str | None:
    """Class stated specifically for molar or canine within a segment.

    English puts the class before the tooth it qualifies ("a Class I molar", "an edge-to-edge
    Class II canine"), so each keyword takes the *nearest preceding* class mention. A naive
    fixed window around the keyword instead reaches into the neighbouring tooth's class and
    silently mislabels sentences like "a Class I molar and edge-to-edge Class II canine".
    """
    mentions = list(_CLASS_MENTION.finditer(segment))
    if not mentions:
        return None

    best: str | None = None
    for m in re.finditer(keyword, segment, re.I):
        anchor = m.start()
        preceding = [c for c in mentions if c.start() <= anchor]
        chosen = preceding[-1] if preceding else mentions[0]
        # Include the words around the class token so "full" / "edge-to-edge" are picked up.
        lo = max(0, chosen.start() - 30)
        hi = min(len(segment), chosen.end() + 30)
        found = _classify_occlusion(segment[lo:hi])
        if found:
            best = found
    if best is None and _NOT_ASSESSABLE.search(segment):
        return "not assessable"
    return best


@dataclass
class ReportFindings:
    """Structured view of one narrative report. ``None`` means 'not stated / unreadable'."""

    constriction: str | None = None  # "absent" | "slight" | "present"
    crossbite: str | None = None  # "absent" | "present"
    crossbite_teeth: list[int] = field(default_factory=list)
    crossbite_side: str | None = None  # "right" | "left" | "bilateral"
    scissor_bite: bool = False
    overbite: str | None = None  # "normal" | "increased" | "reduced" | "open"
    overjet: str | None = None  # "normal" | "increased" | "reduced" | "negative"
    molar_right: str | None = None
    molar_left: str | None = None
    canine_right: str | None = None
    canine_left: str | None = None
    midlines: str | None = None  # "centered" | "deviated"
    spee: str | None = None  # "normal" | "increased" | "reduced" | "reversed"
    wilson: str | None = None
    crowding_upper: str | None = None
    crowding_lower: str | None = None
    spacing: bool = False
    missing_teeth: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def n_parsed(self) -> int:
        keys = (
            "constriction", "crossbite", "overbite", "overjet", "molar_right", "molar_left",
            "canine_right", "canine_left", "midlines", "spee", "wilson",
            "crowding_upper", "crowding_lower",
        )
        return sum(1 for k in keys if getattr(self, k) is not None)


def parse_report(text: str) -> ReportFindings:
    """Extract structured findings from one English Bite2Text report."""
    out = ReportFindings()
    low = text.lower()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    # --- Transverse: maxillary constriction and crossbite ---
    if re.search(r"correct transverse|transverse relationships? (?:are|is) (?:correct|normal)|normal transverse", low):
        out.constriction = "absent"
    elif re.search(r"(?:slight|mild|modest)\s+(?:transverse\s+)?(?:constriction|contraction)", low):
        out.constriction = "slight"
    elif re.search(r"constriction|contraction", low):
        out.constriction = "present"

    if re.search(r"absence of (?:posterior |anterior )?cross-?bite|without (?:posterior )?cross-?bite|no (?:posterior )?cross-?bite", low):
        out.crossbite = "absent"
    elif re.search(r"cross-?bite", low):
        out.crossbite = "present"
        teeth = re.findall(r"\b(\d{2})\b", " ".join(s for s in sents if re.search(r"cross-?bite", s, re.I)))
        out.crossbite_teeth = [int(t) for t in teeth if 11 <= int(t) <= 48]
        quadrants = {t // 10 for t in out.crossbite_teeth}
        right, left = bool(quadrants & {1, 4}), bool(quadrants & {2, 3})
        if re.search(r"bilateral", low) or (right and left):
            out.crossbite_side = "bilateral"
        elif right:
            out.crossbite_side = "right"
        elif left:
            out.crossbite_side = "left"
        elif re.search(r"cross-?bite[^.]{0,40}\bright\b", low):
            out.crossbite_side = "right"
        elif re.search(r"cross-?bite[^.]{0,40}\bleft\b", low):
            out.crossbite_side = "left"
    out.scissor_bite = bool(re.search(r"scissor", low))

    # --- Vertical ---
    if re.search(r"open bite", low):
        out.overbite = "open"
    elif re.search(r"deep bite|overbite is increased|increased overbite|increase[d]? in overbite", low):
        out.overbite = "increased"
    elif re.search(r"overbite is (?:reduced|decreased)|reduced overbite|decreased overbite", low):
        out.overbite = "reduced"
    elif re.search(r"correct vertical|vertical relationships? (?:are|is) (?:correct|normal)|overbite (?:is )?(?:normal|within norm)", low):
        out.overbite = "normal"

    if re.search(r"overjet[^.]{0,30}(?:increased|increase)|increased overjet", low):
        out.overjet = "increased"
    elif re.search(r"overjet[^.]{0,30}(?:reduced|decreased)|reduced overjet", low):
        out.overjet = "reduced"
    elif re.search(r"negative overjet|inverted overjet", low):
        out.overjet = "negative"
    elif re.search(r"overjet[^.]{0,40}(?:within normal|normal limits|is normal)", low):
        out.overjet = "normal"

    # --- Sagittal ---
    sag = " ".join(s for s in sents if re.search(r"sagittal|class\s*(?:i|ii|iii|1|2|3)\b|molar|canine", s, re.I))
    if sag:
        sides = _split_sides(sag)
        both = sides.get("both")
        for side in ("right", "left"):
            segment = sides.get(side) or both
            if not segment:
                continue
            molar = _tooth_specific(segment, r"molar")
            canine = _tooth_specific(segment, r"canine")
            overall = _classify_occlusion(segment)
            setattr(out, f"molar_{side}", molar or overall)
            setattr(out, f"canine_{side}", canine or overall)

    # --- Midlines ---
    if re.search(r"midlines?[^.]{0,60}(?:centered|centred|coincident)", low):
        out.midlines = "centered"
    elif re.search(r"midlines?[^.]{0,60}(?:deviated|deviation|not coincident|off)", low):
        out.midlines = "deviated"

    # --- Occlusal curves ---
    for name, key in (("spee", "spee"), ("wilson", "wilson")):
        # A shared clause ("The curves of Spee and Wilson are increased") covers both.
        shared = re.search(
            r"curves? of spee and (?:the )?(?:curve of )?wilson[^.]{0,60}?"
            r"(increased|accentuated|normal|within normal|reduced|decreased|reversed)",
            low,
        )
        specific = re.search(
            rf"curve of {name}[^.]{{0,60}}?(increased|accentuated|normal|within normal|reduced|decreased|reversed)",
            low,
        )
        m = specific or shared
        if not m:
            continue
        word = m.group(1)
        value = (
            "increased" if word in ("increased", "accentuated")
            else "reduced" if word in ("reduced", "decreased")
            else "reversed" if word == "reversed"
            else "normal"
        )
        setattr(out, key, value)

    # --- Crowding ---
    crowd_sents = [s for s in sents if re.search(r"crowd|spac|diastem|align", s, re.I)]
    crowd_text = " ".join(crowd_sents).lower()
    if crowd_text:
        out.spacing = bool(re.search(r"spac|diastem", crowd_text))
        both_arches = re.search(
            r"(\w[\w\s-]*?)\s+crowding (?:is )?(?:is )?present in the upper and lower arch(?:es)?|"
            r"there is ([\w\s-]*?) crowding in the upper and lower arch(?:es)?",
            crowd_text,
        )
        if re.search(r"upper and lower arch", crowd_text) and not re.search(r"upper arch[^.]{0,40}lower arch[^.]{0,40}(mild|moderate|severe)", crowd_text):
            sev = _severity(crowd_text)
            out.crowding_upper = sev
            out.crowding_lower = sev
        else:
            for arch, key in (("upper", "crowding_upper"), ("lower", "crowding_lower")):
                m = re.search(rf"([\w\s-]{{0,40}})crowding[^.]{{0,40}}{arch} arch", crowd_text)
                if not m:
                    m = re.search(rf"{arch} arch[^.]{{0,40}}?([\w\s-]{{0,30}})crowding", crowd_text)
                if m:
                    setattr(out, key, _severity(m.group(0)))
            if out.crowding_upper is None and out.crowding_lower is None:
                sev = _severity(crowd_text)
                out.crowding_upper = sev
                out.crowding_lower = sev
        if re.search(r"\bno crowding|absence of crowding", crowd_text):
            out.crowding_upper = out.crowding_upper or "absent"
            out.crowding_lower = out.crowding_lower or "absent"

    missing = re.findall(r"(?:absence|lack|missing|agenesis)(?:\s+of)?\s+(?:tooth|teeth)?\s*(\d{2})", low)
    out.missing_teeth = sorted({int(t) for t in missing if 11 <= int(t) <= 48})
    return out
