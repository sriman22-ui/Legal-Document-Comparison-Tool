"""Align clauses between two contracts.

MVP strategy: match by heading/number similarity. Each template clause is paired
with the best remaining revised clause whose normalized heading is similar enough.
Unmatched template clauses are deletions; unmatched revised clauses are additions.

Heading similarity dominates the score so that renumbered clauses (common after a
deletion shifts the numbering) still align. The clause id is only a tiny tiebreaker
— never enough on its own to force a match — otherwise a deleted clause would wrongly
glom onto whatever clause inherited its old number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional

from .schema import Clause

# Minimum heading similarity for two clauses to be considered the same. Renamed
# clauses (e.g. "Late Charges" -> "Late Payment Fees") only partly overlap, so the
# bar is moderate; the greedy best-match still picks the strongest pair per row.
HEADING_MATCH_THRESHOLD = 0.5

# Small bonus when the clause numbers happen to match too.
_ID_MATCH_BONUS = 0.05

# Dropped before token-overlap scoring so filler words don't fake a match.
_STOPWORDS = {"and", "or", "the", "of", "to", "a", "an", "in", "for"}


@dataclass
class AlignedPair:
    """A template clause paired with its revised counterpart.

    Exactly one side may be None: template-only => deleted, revised-only => added.
    """

    template: Optional[Clause]
    revised: Optional[Clause]

    @property
    def is_deleted(self) -> bool:
        return self.template is not None and self.revised is None

    @property
    def is_added(self) -> bool:
        return self.template is None and self.revised is not None


def _normalize(heading: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", heading.lower()).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _content_tokens(heading: str) -> set[str]:
    return {w for w in _normalize(heading).split() if w not in _STOPWORDS}


def _containment(a: str, b: str) -> float:
    """Fraction of the shorter heading's content words shared with the longer.

    1.0 when one heading's words are a subset of the other's, so an expanded
    heading ("Assignment" -> "Assignment and Subletting") still scores as a match
    even though character-level similarity is dragged down by the extra words.
    """
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _heading_similarity(a: str, b: str) -> float:
    return max(_similarity(_normalize(a), _normalize(b)), _containment(a, b))


def _score(template: Clause, revised: Clause) -> float:
    base = _heading_similarity(template.heading, revised.heading)
    bonus = _ID_MATCH_BONUS if template.id.lower() == revised.id.lower() else 0.0
    return min(base + bonus, 1.0)


def align(
    template_clauses: List[Clause], revised_clauses: List[Clause]
) -> List[AlignedPair]:
    """Produce aligned pairs plus unmatched (added / deleted) clauses."""
    pairs: List[AlignedPair] = []
    used_revised: set[int] = set()

    for template in template_clauses:
        best_idx: Optional[int] = None
        best_score = 0.0
        for idx, revised in enumerate(revised_clauses):
            if idx in used_revised:
                continue
            score = _score(template, revised)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score >= HEADING_MATCH_THRESHOLD:
            used_revised.add(best_idx)
            pairs.append(AlignedPair(template=template, revised=revised_clauses[best_idx]))
        else:
            pairs.append(AlignedPair(template=template, revised=None))  # deleted

    for idx, revised in enumerate(revised_clauses):
        if idx not in used_revised:
            pairs.append(AlignedPair(template=None, revised=revised))  # added

    return pairs
