"""Streamlit UI for the Legal Document Comparison Tool.

Pipeline: upload -> parse/OCR -> segment -> align -> per-clause LLM verdict -> report.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.alignment import AlignedPair, align
from src.comparison import (
    compare_clause,
    get_client,
    get_model,
    heuristic_verdict,
    verdict_for_unmatched,
)
from src.parsing import extract_text
from src.schema import ClauseVerdict
from src.segmentation import segment

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_TEMPLATE = DATA_DIR / "sample_template.txt"
SAMPLE_REVISED = DATA_DIR / "sample_revised.txt"

RISK_ORDER = {"high": 3, "medium": 2, "low": 1, "none": 0}
RISK_COLORS = {
    "high": "#b00020",
    "medium": "#b86e00",
    "low": "#5a6570",
    "none": "#3a7d44",
}
CHANGE_TINTS = {
    "deleted": "#fdecea",
    "meaning_changed": "#fff4e5",
    "added": "#e8f0fe",
    "reworded_same_meaning": "#eef7ee",
    "unchanged": "#f3f4f6",
}
SOURCE_LABEL = {
    "text": "plain text",
    "digital": "digital PDF — OCR skipped",
    "scanned": "scanned PDF — OCR applied",
}


def llm_configured() -> bool:
    return bool(os.environ.get("LLM_API_KEY") and os.environ.get("LLM_BASE_URL"))


# --------------------------------------------------------------------------- IO


def _read_upload(upload) -> tuple[str, str]:
    """Persist an uploaded file to a temp path and run it through extract_text."""
    suffix = Path(upload.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        tmp_path = tmp.name
    try:
        text, source = extract_text(tmp_path)
    finally:
        os.unlink(tmp_path)
    return text, source


def _load_sample() -> tuple[tuple[str, str], tuple[str, str]]:
    t_text, t_src = extract_text(str(SAMPLE_TEMPLATE))
    r_text, r_src = extract_text(str(SAMPLE_REVISED))
    return (t_text, t_src), (r_text, r_src)


# --------------------------------------------------------------- comparison run


def _verdict_cache() -> dict:
    if "verdict_cache" not in st.session_state:
        st.session_state.verdict_cache = {}
    return st.session_state.verdict_cache


def run_comparison(pairs: List[AlignedPair], use_llm: bool) -> List[ClauseVerdict]:
    """Produce a verdict per pair, caching aligned-pair verdicts across reruns."""
    cache = _verdict_cache()
    client = get_client() if use_llm else None
    model = get_model() if use_llm else ""

    verdicts: List[ClauseVerdict] = []
    progress = st.progress(0.0, text="Comparing clauses…")
    total = len(pairs)

    for i, pair in enumerate(pairs, start=1):
        if pair.template is None or pair.revised is None:
            verdicts.append(verdict_for_unmatched(pair))
        else:
            key = (pair.template.text, pair.revised.text)
            if key in cache:
                verdicts.append(cache[key])
            else:
                if use_llm:
                    verdict = compare_clause(client, model, pair)
                else:
                    verdict = heuristic_verdict(pair)
                cache[key] = verdict
                verdicts.append(verdict)
        progress.progress(i / total, text=f"Comparing clauses… ({i}/{total})")

    progress.empty()
    return verdicts


# ------------------------------------------------------------------- rendering


def _risk_style(val: str) -> str:
    color = RISK_COLORS.get(val, "#000000")
    weight = "700" if val in ("high", "medium") else "400"
    return f"color: {color}; font-weight: {weight};"


def render_report(verdicts: List[ClauseVerdict]) -> None:
    high = sum(1 for v in verdicts if v.risk_level == "high")
    meaning = sum(1 for v in verdicts if v.change_type == "meaning_changed")
    added = sum(1 for v in verdicts if v.change_type == "added")
    deleted = sum(1 for v in verdicts if v.change_type == "deleted")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High-risk changes", high)
    c2.metric("Meaning changes", meaning)
    c3.metric("Added clauses", added)
    c4.metric("Deleted clauses", deleted)

    st.subheader("Risk summary")
    ordered = sorted(
        verdicts,
        key=lambda v: (RISK_ORDER.get(v.risk_level, 0), v.change_type == "meaning_changed"),
        reverse=True,
    )
    df = pd.DataFrame(
        {
            "heading": [v.heading for v in ordered],
            "change_type": [v.change_type for v in ordered],
            "risk_level": [v.risk_level for v in ordered],
        }
    )
    styled = df.style.map(_risk_style, subset=["risk_level"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Clause-by-clause detail")
    for v in ordered:
        tint = CHANGE_TINTS.get(v.change_type, "#f3f4f6")
        label = f"{v.heading} — {v.change_type} / {v.risk_level} risk"
        with st.expander(label, expanded=v.risk_level == "high"):
            st.markdown(
                f"<div style='background:{tint};color:#1a1a1a;padding:0.6rem 0.8rem;"
                f"border-radius:6px;'>"
                f"<b>What changed:</b> {v.explanation}</div>",
                unsafe_allow_html=True,
            )
            left, right = st.columns(2)
            with left:
                st.caption("Template")
                st.write(v.template_text or "_(not present in template)_")
            with right:
                st.caption("Revised")
                st.write(v.revised_text or "_(not present in revised)_")


# ------------------------------------------------------------------------ main


def _process(template_text: str, revised_text: str) -> None:
    template_clauses = segment(template_text)
    revised_clauses = segment(revised_text)
    st.caption(
        f"Segmented {len(template_clauses)} template clauses and "
        f"{len(revised_clauses)} revised clauses."
    )
    pairs = align(template_clauses, revised_clauses)
    verdicts = run_comparison(pairs, use_llm=llm_configured())
    render_report(verdicts)


def main() -> None:
    st.set_page_config(page_title="Legal Document Comparison Tool", layout="wide")
    st.title("⚖️ Legal Document Comparison Tool")

    configured = llm_configured()
    if configured:
        st.success(f"LLM configured — model `{os.environ.get('LLM_MODEL', '?')}`.")
    else:
        st.warning(
            "Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL in your .env to run "
            "comparisons. You can still browse the UI with sample data loaded "
            "(clauses are compared with an offline text-similarity heuristic)."
        )

    col_a, col_b = st.columns(2)
    template_upload = col_a.file_uploader("Template (original)", type=["pdf", "txt"])
    revised_upload = col_b.file_uploader("Revised contract", type=["pdf", "txt"])

    run_clicked = st.button("Compare", type="primary")
    sample_clicked = st.button("Load sample contracts")

    if sample_clicked:
        (t_text, t_src), (r_text, r_src) = _load_sample()
        st.info(
            f"Loaded sample NDA. Template detected as **{SOURCE_LABEL[t_src]}**; "
            f"revised detected as **{SOURCE_LABEL[r_src]}**."
        )
        _process(t_text, r_text)
        return

    if run_clicked:
        if not template_upload or not revised_upload:
            st.error("Please upload both a template and a revised document, or load the sample.")
            return
        t_text, t_src = _read_upload(template_upload)
        r_text, r_src = _read_upload(revised_upload)
        st.info(
            f"Template detected as **{SOURCE_LABEL[t_src]}**; "
            f"revised detected as **{SOURCE_LABEL[r_src]}**."
        )
        _process(t_text, r_text)


if __name__ == "__main__":
    main()
