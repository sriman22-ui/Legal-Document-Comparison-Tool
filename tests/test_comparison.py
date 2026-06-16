"""Tests for parsing, segmentation, alignment, and the comparison call.

The live LLM API is never called — compare_clause is exercised with a mock client.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.alignment import align
from src.comparison import compare_clause, verdict_for_unmatched
from src.parsing import MIN_CHARS_PER_PAGE, classify_pdf_source, extract_text
from src.schema import Clause
from src.segmentation import segment

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEMPLATE = DATA_DIR / "sample_template.txt"
REVISED = DATA_DIR / "sample_revised.txt"


# ------------------------------------------------------------------- parsing


def test_txt_parsing_reads_directly_as_text():
    text, source = extract_text(str(TEMPLATE))
    assert source == "text"
    assert "Confidential Information" in text


def test_digital_vs_scanned_threshold():
    # Plenty of selectable text per page -> digital.
    dense = "x" * (MIN_CHARS_PER_PAGE + 50)
    assert classify_pdf_source(dense * 3, num_pages=3) == "digital"

    # Near-empty extraction -> scanned (needs OCR).
    assert classify_pdf_source("   \n  ", num_pages=2) == "scanned"

    # Exactly at the threshold is NOT above it -> scanned.
    at_threshold = "y" * MIN_CHARS_PER_PAGE
    assert classify_pdf_source(at_threshold, num_pages=1) == "scanned"


def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        extract_text("contract.docx")


# -------------------------------------------------------------- segmentation


def test_segmentation_finds_all_clauses():
    template_text, _ = extract_text(str(TEMPLATE))
    clauses = segment(template_text)
    # 8 numbered clauses; the all-caps title must NOT become a clause.
    assert len(clauses) == 8
    assert clauses[0].id == "1"
    assert clauses[0].heading == "Definitions"
    headings = {c.heading for c in clauses}
    assert "Remedies and Injunctive Relief" in headings
    assert "MUTUAL NON-DISCLOSURE AGREEMENT" not in headings


def test_revised_segmentation_has_seven_clauses():
    revised_text, _ = extract_text(str(REVISED))
    assert len(segment(revised_text)) == 7


# ----------------------------------------------------------------- alignment


def test_alignment_flags_the_deleted_injunctive_relief_clause():
    template = segment(extract_text(str(TEMPLATE))[0])
    revised = segment(extract_text(str(REVISED))[0])
    pairs = align(template, revised)

    deleted = [p for p in pairs if p.is_deleted]
    assert len(deleted) == 1
    assert "Injunctive Relief" in deleted[0].template.heading

    # Renumbered survivors still align by heading.
    matched_headings = {
        p.template.heading for p in pairs if p.template and p.revised
    }
    assert "Governing Law" in matched_headings
    assert "Entire Agreement" in matched_headings

    # No spurious additions for this edit set.
    assert not any(p.is_added for p in pairs)


def test_verdict_for_deleted_clause_is_high_risk():
    template = [Clause(id="6", heading="Remedies and Injunctive Relief", text="...")]
    pairs = align(template, [])
    verdict = verdict_for_unmatched(pairs[0])
    assert verdict.change_type == "deleted"
    assert verdict.risk_level == "high"


# -------------------------------------------------- comparison (mocked LLM)


def _mock_client_returning(payload: dict) -> MagicMock:
    client = MagicMock()
    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


def test_compare_clause_parses_mocked_response():
    template = Clause(id="3", heading="Term", text="five (5) years")
    revised = Clause(id="3", heading="Term", text="two (2) years")
    pair = align([template], [revised])[0]

    client = _mock_client_returning(
        {
            "change_type": "meaning_changed",
            "risk_level": "medium",
            "explanation": "The term was shortened from five to two years.",
        }
    )
    verdict = compare_clause(client, "mock-model", pair)
    assert verdict.change_type == "meaning_changed"
    assert verdict.risk_level == "medium"
    assert verdict.clause_id == "3"
    assert verdict.template_text == "five (5) years"


def test_compare_clause_strips_json_fences():
    template = Clause(id="1", heading="Definitions", text="a")
    revised = Clause(id="1", heading="Definitions", text="a")
    pair = align([template], [revised])[0]

    client = MagicMock()
    message = MagicMock()
    message.content = (
        '```json\n{"change_type": "unchanged", "risk_level": "none", '
        '"explanation": "No change."}\n```'
    )
    choice = MagicMock()
    choice.message = message
    client.chat.completions.create.return_value = MagicMock(choices=[choice])

    verdict = compare_clause(client, "mock-model", pair)
    assert verdict.change_type == "unchanged"


def test_compare_clause_fails_safe_on_bad_json():
    template = Clause(id="2", heading="Confidentiality", text="x")
    revised = Clause(id="2", heading="Confidentiality", text="y")
    pair = align([template], [revised])[0]

    client = MagicMock()
    message = MagicMock()
    message.content = "not json at all"
    choice = MagicMock()
    choice.message = message
    client.chat.completions.create.return_value = MagicMock(choices=[choice])

    verdict = compare_clause(client, "mock-model", pair, max_attempts=1)
    assert verdict.change_type == "meaning_changed"
    assert verdict.risk_level == "medium"
    assert "manual review" in verdict.explanation
