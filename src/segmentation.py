"""Split a contract's raw text into clauses using heading / clause markers.

We recognise three kinds of headings:
  * numbered:  "1.", "1.1", "2)"  (the common case)
  * named:     "Section 4", "ARTICLE V", "Clause 1. ASSIGNMENT"
  * all-caps:  a short ALL CAPS line acting as a heading

Numbered and named headings are tried first. All-caps detection is only used as a
fallback for documents that have no numbered/named structure — otherwise a document
*title* (also all caps) would be mis-read as a clause.
"""
from __future__ import annotations

import re
from typing import List, Optional, Pattern, Tuple

from .schema import Clause

# "1." / "1.1" / "2)"  followed by a capitalised heading word.
_NUMBERED: Pattern[str] = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+([A-Z].*)$")

# "Section 4", "ARTICLE V", "Article 12:", "Clause 1. ASSIGNMENT", "CLAUSE 2" ...
# Tolerant of OCR damage: case-insensitive, flexible separator, and the classic
# l<->I<->1<->| glyph confusion in the keyword (so "ClauSe1.", "CIause 3" — capital i
# for the L — and "C1ause 2" are all still recognised as clause headings).
_NAMED: Pattern[str] = re.compile(
    r"^((?:se[ci]tion|artic[li1|]e|c[li1|]ause)\s*[0-9ivxlcdm]+)[.:)\-]?\s*(.*)$",
    re.IGNORECASE,
)

# A short ALL-CAPS line, e.g. "CONFIDENTIALITY". Used only as a fallback.
_ALLCAPS: Pattern[str] = re.compile(r"^([A-Z][A-Z0-9 ,'&/\-]{3,60})$")

# Document furniture (titles, watermarks, stamps) that OCR picks up but that must
# NOT be treated as clause headings — otherwise "OLD VERSION" or "FINAL DRAFT 2024"
# become spurious clauses.
_NON_CLAUSE_MARKERS: Pattern[str] = re.compile(
    r"\b(version|draft|final|revised|original|confidential|copy|sample|specimen"
    r"|page|exhibit|annex|appendix)\b",
    re.IGNORECASE,
)


def _match_heading(line: str, patterns: List[Pattern[str]]) -> Optional[Tuple[str, str]]:
    """Return (clause_id, heading) if the line starts a clause, else None."""
    s = line.strip()
    if not s:
        return None

    for pat in patterns:
        m = pat.match(s)
        if not m:
            continue
        if pat is _NUMBERED:
            number = m.group(1)
            rest = m.group(2).strip()
            # The heading is the title up to the first sentence break.
            heading = rest.split(".")[0].strip() if rest else number
            return number, heading
        if pat is _NAMED:
            label = m.group(1).strip()
            tail = m.group(2).strip()
            heading = tail.split(".")[0].strip() if tail else label
            return label, heading
        # _ALLCAPS — skip document furniture (titles / watermarks / stamps).
        if _NON_CLAUSE_MARKERS.search(s):
            return None
        return s, s.title()
    return None


def _segment_with(text: str, patterns: List[Pattern[str]]) -> List[Clause]:
    clauses: List[Clause] = []
    current_id: Optional[str] = None
    current_heading: str = ""
    current_lines: List[str] = []

    def flush() -> None:
        if current_id is not None:
            clauses.append(
                Clause(
                    id=current_id,
                    heading=current_heading,
                    text="\n".join(current_lines).strip(),
                )
            )

    for line in text.splitlines():
        match = _match_heading(line, patterns)
        if match:
            flush()
            current_id, current_heading = match
            current_lines = [line.strip()]
        elif current_id is not None:
            current_lines.append(line)
        # text before the first heading (preamble/title) is ignored

    flush()
    return clauses


def segment(text: str) -> List[Clause]:
    """Segment contract text into clauses.

    Tries numbered/named headings first; only falls back to all-caps heading
    detection for documents that have no numbered structure at all.
    """
    clauses = _segment_with(text, [_NUMBERED, _NAMED])
    if len(clauses) >= 2:
        return clauses
    return _segment_with(text, [_NUMBERED, _NAMED, _ALLCAPS])
