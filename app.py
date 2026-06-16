"""Streamlit UI for the Legal Document Comparison Tool.

Pipeline: upload -> parse/OCR -> segment -> align -> per-clause LLM verdict -> report.
"""
from __future__ import annotations

import os
import tempfile
import time
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

# override=True so editing .env (e.g. switching LLM_MODEL) takes effect on the
# next rerun — without it, python-dotenv keeps the value first loaded this process.
load_dotenv(override=True)

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
    "image": "image — OCR applied",
}

UPLOAD_TYPES = ["pdf", "txt", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]


def llm_configured() -> bool:
    return bool(os.environ.get("LLM_API_KEY") and os.environ.get("LLM_BASE_URL"))


# ------------------------------------------------------ processing animation

# Abstract morphing shapes: three blobs that bend/expand/rotate and blend, plus a
# slowly spinning dashed ring. Rendered ONCE into its own placeholder so the CSS
# animation runs continuously; only the percentage text below it is re-rendered.
_PROCESSING_SHAPES = """
<style>
.ldc-stage{display:flex;justify-content:center;align-items:center;height:168px;margin:6px 0 2px;}
.ldc-orbit{position:relative;width:148px;height:148px;}
.ldc-ring{position:absolute;inset:-12px;border:2px dashed rgba(150,168,200,.40);
  border-radius:46% 54% 52% 48%/48% 46% 54% 52%;animation:ldc-spin 7s linear infinite;}
.ldc-blob{position:absolute;inset:0;margin:auto;width:98px;height:98px;mix-blend-mode:screen;
  opacity:.88;animation:ldc-morph 3.4s ease-in-out infinite;}
.ldc-b1{background:#5b8def;}
.ldc-b2{background:#c8821a;animation-duration:4.2s;animation-delay:-1.3s;}
.ldc-b3{background:#3a9d57;animation-duration:5.0s;animation-delay:-2.4s;}
@keyframes ldc-morph{
  0%,100%{border-radius:42% 58% 70% 30%/45% 45% 55% 55%;transform:rotate(0deg) scale(1) translate(0,0);}
  25%{border-radius:70% 30% 46% 54%/30% 60% 40% 70%;transform:rotate(90deg) scale(1.14) translate(7px,-5px);}
  50%{border-radius:34% 66% 56% 44%/64% 44% 56% 36%;transform:rotate(180deg) scale(.90) translate(-6px,6px);}
  75%{border-radius:58% 42% 38% 62%/52% 56% 44% 48%;transform:rotate(270deg) scale(1.08) translate(5px,7px);}
}
@keyframes ldc-spin{to{transform:rotate(360deg);}}
.ldc-pct{text-align:center;font-size:2.5rem;font-weight:800;letter-spacing:.5px;line-height:1.1;
  background:linear-gradient(90deg,#5b8def,#3a9d57);-webkit-background-clip:text;
  background-clip:text;color:transparent;}
.ldc-sub{text-align:center;color:#9aa7b8;font-size:.9rem;margin-top:-2px;}
</style>
<div class="ldc-stage"><div class="ldc-orbit">
  <div class="ldc-ring"></div>
  <div class="ldc-blob ldc-b1"></div>
  <div class="ldc-blob ldc-b2"></div>
  <div class="ldc-blob ldc-b3"></div>
</div></div>
"""


def _render_percent(placeholder, frac: float, i: int, total: int) -> None:
    pct = int(round(frac * 100))
    placeholder.markdown(
        f"<div class='ldc-pct'>{pct}%</div>"
        f"<div class='ldc-sub'>Comparing clauses… ({i}/{total})</div>",
        unsafe_allow_html=True,
    )


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


def run_comparison(
    pairs: List[AlignedPair],
    use_llm: bool,
    anim_ph,
    pct_ph,
) -> List[ClauseVerdict]:
    """Produce a verdict per pair, caching aligned-pair verdicts across reruns.

    Drives the morphing-shapes animation: ``anim_ph`` holds the shapes (rendered
    once so the CSS keeps running) and ``pct_ph`` shows the live percentage.
    """
    cache = _verdict_cache()
    client = get_client() if use_llm else None
    model = get_model() if use_llm else ""

    verdicts: List[ClauseVerdict] = []
    total = len(pairs)

    anim_ph.markdown(_PROCESSING_SHAPES, unsafe_allow_html=True)
    _render_percent(pct_ph, 0.0, 0, total)

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
                    # The offline heuristic is near-instant; pace it slightly so
                    # the processing animation is actually perceptible.
                    time.sleep(0.35)
                cache[key] = verdict
                verdicts.append(verdict)
        _render_percent(pct_ph, i / total, i, total)

    anim_ph.empty()
    pct_ph.empty()
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
    anim_ph = st.empty()
    pct_ph = st.empty()
    verdicts = run_comparison(pairs, llm_configured(), anim_ph, pct_ph)
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
    template_upload = col_a.file_uploader("Template (original)", type=UPLOAD_TYPES)
    revised_upload = col_b.file_uploader("Revised contract", type=UPLOAD_TYPES)

    # Centre the action buttons (width-independent), and keep them in a placeholder
    # we can clear so they disappear the moment a comparison starts.
    st.markdown(
        "<style>"
        "div[data-testid='stElementContainer']:has(>.stButton){width:100% !important;}"
        ".stButton{display:flex;justify-content:center;width:100%;}"
        ".stButton>button{min-width:260px;max-width:420px;}"
        "</style>",
        unsafe_allow_html=True,
    )
    controls_ph = st.empty()
    with controls_ph.container():
        run_clicked = st.button("Compare", type="primary")
        sample_clicked = st.button("Load sample contracts")

    if sample_clicked:
        controls_ph.empty()
        (t_text, t_src), (r_text, r_src) = _load_sample()
        st.info(
            f"Loaded sample NDA.  \n"
            f"Template detected as **{SOURCE_LABEL[t_src]}**.  \n"
            f"Revised detected as **{SOURCE_LABEL[r_src]}**."
        )
        _process(t_text, r_text)
        return

    if run_clicked:
        if not template_upload or not revised_upload:
            st.error("Please upload both a template and a revised document, or load the sample.")
            return
        controls_ph.empty()
        t_text, t_src = _read_upload(template_upload)
        r_text, r_src = _read_upload(revised_upload)
        st.info(
            f"Template detected as **{SOURCE_LABEL[t_src]}**.  \n"
            f"Revised detected as **{SOURCE_LABEL[r_src]}**."
        )
        _process(t_text, r_text)


if __name__ == "__main__":
    main()
