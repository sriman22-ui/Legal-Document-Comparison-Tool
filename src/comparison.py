"""Per-clause comparison.

For each ALIGNED pair we make a SEPARATE LLM call (never the whole document) and
parse the reply into a ClauseVerdict. Added/deleted clauses are labelled without an
LLM call. An offline text-similarity heuristic is provided so the UI is fully
browsable even before an API key is configured.

The `openai` SDK is imported lazily (only inside get_client) so this module — and the
test suite, which mocks the client — imports without the SDK installed.
"""
from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Optional

from .alignment import AlignedPair
from .schema import Clause, ClauseVerdict

SYSTEM_PROMPT = (
    "You are a contracts analyst. You compare a clause from an original template "
    "against the corresponding clause in a revised contract. Decide whether the "
    "meaning changed, not just the wording. Assess risk to the party using the "
    "template. Reply with ONLY a JSON object, no prose, no markdown fences."
)

# Retry / backoff for rate-limited free providers.
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 1.0


def get_client() -> Any:
    """Build an OpenAI-compatible client from env vars (works with any provider)."""
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    )


def get_model() -> str:
    return os.environ["LLM_MODEL"]


def build_user_prompt(heading: str, template_text: str, revised_text: str) -> str:
    return (
        f"Clause heading: {heading}\n\n"
        f'ORIGINAL TEMPLATE CLAUSE:\n"""{template_text}"""\n\n'
        f'REVISED CONTRACT CLAUSE:\n"""{revised_text}"""\n\n'
        "Compare the revised clause against the template clause. Decide whether the "
        "MEANING changed (not just the wording), and assess the risk to the party "
        "relying on the template.\n\n"
        "Return ONLY a JSON object with exactly these fields:\n"
        "{\n"
        '  "change_type": one of ["unchanged", "reworded_same_meaning", "meaning_changed"],\n'
        '  "risk_level": one of ["none", "low", "medium", "high"],\n'
        '  "explanation": "one plain-English sentence: what changed and why it matters"\n'
        "}"
    )


def _strip_fences(text: str) -> str:
    """Remove accidental ```json ... ``` fences before JSON parsing."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _backoff_seconds(attempt: int) -> float:
    return _BACKOFF_BASE_SECONDS * (2 ** attempt)


def compare_clause(
    client: Any,
    model: str,
    pair: AlignedPair,
    max_attempts: int = _MAX_ATTEMPTS,
) -> ClauseVerdict:
    """Compare one aligned pair via the LLM, with retry and a fail-safe verdict."""
    template = pair.template
    revised = pair.revised
    assert template is not None and revised is not None, "compare_clause needs an aligned pair"

    user_prompt = build_user_prompt(template.heading, template.text, revised.text)
    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content
            data = json.loads(_strip_fences(content))
            return ClauseVerdict(
                clause_id=template.id,
                heading=template.heading,
                change_type=data["change_type"],
                risk_level=data["risk_level"],
                explanation=data["explanation"],
                template_text=template.text,
                revised_text=revised.text,
            )
        except Exception as exc:  # noqa: BLE001 — free providers raise many error types
            last_error = exc
            if attempt < max_attempts - 1:
                time.sleep(_backoff_seconds(attempt))

    # Fail safe: never crash. Flag for manual review at medium risk.
    return ClauseVerdict(
        clause_id=template.id,
        heading=template.heading,
        change_type="meaning_changed",
        risk_level="medium",
        explanation=(
            "Automated comparison could not parse a valid response after retries "
            f"({last_error}); this clause is flagged for manual review."
        ),
        template_text=template.text,
        revised_text=revised.text,
    )


def verdict_for_unmatched(pair: AlignedPair) -> ClauseVerdict:
    """Label an added or deleted clause without calling the LLM."""
    if pair.is_deleted:
        clause: Clause = pair.template  # type: ignore[assignment]
        return ClauseVerdict(
            clause_id=clause.id,
            heading=clause.heading,
            change_type="deleted",
            risk_level="high",
            explanation=(
                "This clause from the template is absent in the revised contract; "
                "removing it may eliminate a protection the template relied on."
            ),
            template_text=clause.text,
            revised_text=None,
        )

    clause = pair.revised  # type: ignore[assignment]
    return ClauseVerdict(
        clause_id=clause.id,
        heading=clause.heading,
        change_type="added",
        risk_level="medium",
        explanation=(
            "This clause is new in the revised contract and had no counterpart in "
            "the template; review whether it introduces new obligations."
        ),
        template_text=None,
        revised_text=clause.text,
    )


def heuristic_verdict(pair: AlignedPair) -> ClauseVerdict:
    """Offline, no-LLM verdict based on text similarity.

    Used when no API key is configured so the report is still browsable. It cannot
    tell 'reworded same meaning' from 'meaning changed' reliably — that is exactly
    what the LLM is for — so it is conservative and clearly labelled.
    """
    template = pair.template
    revised = pair.revised
    assert template is not None and revised is not None

    a = re.sub(r"\s+", " ", template.text.strip().lower())
    b = re.sub(r"\s+", " ", revised.text.strip().lower())
    ratio = SequenceMatcher(None, a, b).ratio()

    if a == b:
        change_type, risk = "unchanged", "none"
        note = "Text is identical."
    elif ratio > 0.97:
        change_type, risk = "reworded_same_meaning", "none"
        note = "Only trivial wording differences."
    elif ratio > 0.6:
        change_type, risk = "meaning_changed", "medium"
        note = "Wording differs substantially; review for a change in meaning."
    else:
        change_type, risk = "meaning_changed", "high"
        note = "Text differs heavily; likely a material change."

    return ClauseVerdict(
        clause_id=template.id,
        heading=template.heading,
        change_type=change_type,  # type: ignore[arg-type]
        risk_level=risk,  # type: ignore[arg-type]
        explanation=f"[offline heuristic — no LLM configured] {note}",
        template_text=template.text,
        revised_text=revised.text,
    )
