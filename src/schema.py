"""Typed data models shared across the pipeline."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

# Classification of how a clause changed between template and revised contract.
ChangeType = Literal[
    "unchanged",
    "reworded_same_meaning",
    "meaning_changed",
    "added",
    "deleted",
]

# Risk to the party relying on the original template.
RiskLevel = Literal["none", "low", "medium", "high"]


class Clause(BaseModel):
    """A single segmented clause from a contract."""

    id: str
    heading: str
    text: str


class ClauseVerdict(BaseModel):
    """The comparison result for one clause (the unit rendered in the report)."""

    clause_id: str
    heading: str
    change_type: ChangeType
    risk_level: RiskLevel
    explanation: str  # one plain-English sentence: what changed and why it matters
    template_text: Optional[str] = None
    revised_text: Optional[str] = None
