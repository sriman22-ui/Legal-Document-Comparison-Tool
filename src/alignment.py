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

# Minimum normalized-heading similarity for two clauses to be considered the same.
HEADING_MATCH_THRESHOLD = 0.6

# Small bonus when the clause numbers happen to match too.
_ID_MATCH_BONUS = 0.05


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


def _score(template: Clause, revised: Clause) -> float:
    base = _similarity(_normalize(template.heading), _normalize(revised.heading))
    bonus = _ID_MATCH_BONUS if template.id == revised.id else 0.0
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
