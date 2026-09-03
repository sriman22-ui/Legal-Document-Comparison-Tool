"""Gradio UI for the Legal Document Comparison Tool.

Pipeline: upload -> parse/OCR -> segment -> align -> per-clause LLM verdict -> report.

Gradio (rather than Streamlit) because Hugging Face Spaces' free tier offers Gradio
but not Streamlit, and the OCR path needs more RAM than other free hosts allow.
Everything under ``src/`` is UI-agnostic and is untouched by this module.
"""
from __future__ import annotations

import html
import os
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import gradio as gr
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

# override=True so editing .env (e.g. switching LLM_MODEL) takes effect on a
# restart — without it, python-dotenv keeps the value first loaded this process.
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

UPLOAD_TYPES = [
    ".pdf", ".txt", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
]

# Cost control: every aligned clause pair costs one LLM call, so an unbounded
# upload fans out into unbounded spend. Enforced in _check_size, since Gradio has
# no built-in per-file size cap.
MAX_UPLOAD_MB = 30


def llm_configured() -> bool:
    return bool(os.environ.get("LLM_API_KEY") and os.environ.get("LLM_BASE_URL"))


# ------------------------------------------------------------------ page style

# Warm amber gradient wash: soft blurred colour concentrated at the top of the
# page, fading to near-white further down so the report stays legible. Layered
# radial gradients (rather than an image) keep it resolution-independent.
_CSS = """
.gradio-container, .gradio-container .main {
  background:
    radial-gradient(58% 46% at 8% 10%,  rgba(245,168,38,.62) 0%, rgba(245,168,38,0) 60%),
    radial-gradient(50% 40% at 92% 2%,  rgba(242,104,60,.40) 0%, rgba(242,104,60,0) 64%),
    radial-gradient(95% 60% at 50% -10%,rgba(255,206,128,.66) 0%, rgba(255,206,128,0) 72%),
    linear-gradient(180deg,#FFF6E8 0%,#FFFCF6 42%,#FFFDF9 100%) !important;
  background-attachment: fixed !important;
}
.ldc-title{font-size:2.1rem;font-weight:800;color:#1F2328;margin:.2rem 0 .1rem;}
.ldc-note{border-radius:10px;background:rgba(255,255,255,.86);padding:.7rem .9rem;
  box-shadow:0 4px 16px rgba(190,120,50,.07);color:#1F2328;margin:.4rem 0;}
.ldc-note.ok{border-left:4px solid #3a7d44;}
.ldc-note.warn{border-left:4px solid #E08A1E;}
.ldc-note.err{border-left:4px solid #b00020;}
.ldc-note.info{border-left:4px solid #F2A03C;}

/* Metric row: the big coral numbers. */
.ldc-metrics{display:flex;gap:1.4rem;flex-wrap:wrap;margin:.9rem 0 .3rem;}
.ldc-metric{flex:1 1 150px;background:rgba(255,255,255,.86);border-radius:12px;
  padding:.7rem .9rem;box-shadow:0 6px 22px rgba(190,120,50,.07);}
.ldc-metric .lbl{font-size:.82rem;color:#6b7280;}
.ldc-metric .val{font-size:2rem;font-weight:800;color:#EF5A28;line-height:1.15;}

/* Risk summary table. */
.ldc-table{width:100%;border-collapse:collapse;background:rgba(255,255,255,.9);
  border-radius:12px;overflow:hidden;box-shadow:0 6px 22px rgba(190,120,50,.07);}
.ldc-table th,.ldc-table td{text-align:left;padding:.5rem .8rem;font-size:.92rem;
  border-bottom:1px solid rgba(226,168,100,.22);}
.ldc-table th{color:#6b7280;font-weight:600;}

/* Per-clause detail. */
.ldc-clause{background:rgba(255,255,255,.92);border:1px solid rgba(226,168,100,.30);
  border-radius:12px;margin:.55rem 0;overflow:hidden;
  box-shadow:0 6px 22px rgba(190,120,50,.07);}
.ldc-clause>summary{cursor:pointer;padding:.65rem .9rem;font-weight:600;color:#1F2328;}
.ldc-body{padding:0 .9rem .9rem;}
.ldc-expl{padding:.6rem .8rem;border-radius:6px;color:#1a1a1a;margin-bottom:.6rem;}
.ldc-cols{display:flex;gap:1rem;flex-wrap:wrap;}
.ldc-col{flex:1 1 260px;min-width:240px;}
.ldc-col .cap{font-size:.78rem;color:#6b7280;margin-bottom:.2rem;}
.ldc-col .txt{font-size:.9rem;white-space:pre-wrap;color:#1F2328;}
"""

# ------------------------------------------------------ processing animation

# Abstract morphing shapes: three blobs that bend/expand/rotate and blend, plus a
# slowly spinning dashed ring. multiply (not screen) so they read as deeper pools
# of the warm background rather than washing out to white on a light page.
_PROCESSING_SHAPES = """
<style>
.ldc-stage{display:flex;justify-content:center;align-items:center;height:168px;margin:6px 0 2px;}
.ldc-orbit{position:relative;width:148px;height:148px;}
.ldc-ring{position:absolute;inset:-12px;border:2px dashed rgba(226,150,60,.45);
  border-radius:46% 54% 52% 48%/48% 46% 54% 52%;animation:ldc-spin 7s linear infinite;}
.ldc-blob{position:absolute;inset:0;margin:auto;width:98px;height:98px;mix-blend-mode:multiply;
  opacity:.72;animation:ldc-morph 3.4s ease-in-out infinite;}
.ldc-b1{background:#F5A826;}
.ldc-b2{background:#F2683C;animation-duration:4.2s;animation-delay:-1.3s;}
.ldc-b3{background:#FFCE80;animation-duration:5.0s;animation-delay:-2.4s;}
@keyframes ldc-morph{
  0%,100%{border-radius:42% 58% 70% 30%/45% 45% 55% 55%;transform:rotate(0deg) scale(1) translate(0,0);}
  25%{border-radius:70% 30% 46% 54%/30% 60% 40% 70%;transform:rotate(90deg) scale(1.14) translate(7px,-5px);}
  50%{border-radius:34% 66% 56% 44%/64% 44% 56% 36%;transform:rotate(180deg) scale(.90) translate(-6px,6px);}
  75%{border-radius:58% 42% 38% 62%/52% 56% 44% 48%;transform:rotate(270deg) scale(1.08) translate(5px,7px);}
}
@keyframes ldc-spin{to{transform:rotate(360deg);}}
.ldc-pct{text-align:center;font-size:2.5rem;font-weight:800;letter-spacing:.5px;line-height:1.1;
  background:linear-gradient(90deg,#F5A826,#EF5A28);-webkit-background-clip:text;
  background-clip:text;color:transparent;}
.ldc-sub{text-align:center;color:#9a8778;font-size:.9rem;margin-top:-2px;}
</style>
<div class="ldc-stage"><div class="ldc-orbit">
  <div class="ldc-ring"></div>
  <div class="ldc-blob ldc-b1"></div>
  <div class="ldc-blob ldc-b2"></div>
  <div class="ldc-blob ldc-b3"></div>
</div></div>
"""


# Gradio follows the OS dark-mode preference, but this app's palette is light-only
# (a warm amber wash with dark text), so dark mode renders near-black uploaders and
# unreadable text over it. Gradio's supported switch is the ?__theme=light URL
# param; pinning it from <head> runs before the app boots, so there is no flash.
_FORCE_LIGHT = """
<script>
  (function () {
    var url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'light') {
      url.searchParams.set('__theme', 'light');
      window.location.replace(url.href);
    }
  })();
</script>
"""


def _progress_html(frac: float, i: int, total: int) -> str:
    pct = int(round(frac * 100))
    return (
        f"{_PROCESSING_SHAPES}"
        f"<div class='ldc-pct'>{pct}%</div>"
        f"<div class='ldc-sub'>Comparing clauses… ({i}/{total})</div>"
    )


def _note(kind: str, message: str) -> str:
    return f"<div class='ldc-note {kind}'>{message}</div>"


# --------------------------------------------------------------------------- IO


def _check_size(path: str, label: str) -> Optional[str]:
    """Return an error message if the file exceeds the upload cap, else None."""
    mb = os.path.getsize(path) / (1024 * 1024)
    if mb > MAX_UPLOAD_MB:
        return f"{label} is {mb:.1f}MB, above the {MAX_UPLOAD_MB}MB limit."
    return None


def _check_type(path: str, label: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext not in UPLOAD_TYPES:
        return f"{label} has unsupported type '{ext}'. Accepted: {', '.join(UPLOAD_TYPES)}."
    return None


# --------------------------------------------------------------- comparison run


def run_comparison(
    pairs: List[AlignedPair],
    use_llm: bool,
    cache: dict,
) -> Iterator[Tuple[List[ClauseVerdict], str]]:
    """Yield (verdicts_so_far, progress_html) after each clause.

    Streaming rather than returning once keeps the morphing-shapes animation and
    the live percentage on screen while the LLM calls run.
    """
    client = get_client() if use_llm else None
    model = get_model() if use_llm else ""

    verdicts: List[ClauseVerdict] = []
    total = len(pairs)
    yield verdicts, _progress_html(0.0, 0, total)

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
        yield verdicts, _progress_html(i / total, i, total)


# ------------------------------------------------------------------- rendering


def render_report(verdicts: List[ClauseVerdict]) -> str:
    high = sum(1 for v in verdicts if v.risk_level == "high")
    meaning = sum(1 for v in verdicts if v.change_type == "meaning_changed")
    added = sum(1 for v in verdicts if v.change_type == "added")
    deleted = sum(1 for v in verdicts if v.change_type == "deleted")

    metrics = "".join(
        f"<div class='ldc-metric'><div class='lbl'>{lbl}</div>"
        f"<div class='val'>{val}</div></div>"
        for lbl, val in (
            ("High-risk changes", high),
            ("Meaning changes", meaning),
            ("Added clauses", added),
            ("Deleted clauses", deleted),
        )
    )

    ordered = sorted(
        verdicts,
        key=lambda v: (RISK_ORDER.get(v.risk_level, 0), v.change_type == "meaning_changed"),
        reverse=True,
    )

    rows = ""
    for v in ordered:
        color = RISK_COLORS.get(v.risk_level, "#000000")
        weight = "700" if v.risk_level in ("high", "medium") else "400"
        rows += (
            f"<tr><td>{html.escape(v.heading)}</td>"
            f"<td>{html.escape(v.change_type)}</td>"
            f"<td style='color:{color};font-weight:{weight};'>"
            f"{html.escape(v.risk_level)}</td></tr>"
        )
    table = (
        "<table class='ldc-table'><thead><tr><th>heading</th><th>change_type</th>"
        f"<th>risk_level</th></tr></thead><tbody>{rows}</tbody></table>"
    )

    details = ""
    for v in ordered:
        tint = CHANGE_TINTS.get(v.change_type, "#f3f4f6")
        label = f"{v.heading} — {v.change_type} / {v.risk_level} risk"
        open_attr = " open" if v.risk_level == "high" else ""
        left = html.escape(v.template_text or "(not present in template)")
        right = html.escape(v.revised_text or "(not present in revised)")
        details += (
            f"<details class='ldc-clause'{open_attr}>"
            f"<summary>{html.escape(label)}</summary>"
            f"<div class='ldc-body'>"
            f"<div class='ldc-expl' style='background:{tint};'>"
            f"<b>What changed:</b> {html.escape(v.explanation)}</div>"
            f"<div class='ldc-cols'>"
            f"<div class='ldc-col'><div class='cap'>Template</div>"
            f"<div class='txt'>{left}</div></div>"
            f"<div class='ldc-col'><div class='cap'>Revised</div>"
            f"<div class='txt'>{right}</div></div>"
            f"</div></div></details>"
        )

    return (
        f"<div class='ldc-metrics'>{metrics}</div>"
        f"<h3>Risk summary</h3>{table}"
        f"<h3>Clause-by-clause detail</h3>{details}"
    )


# ------------------------------------------------------------------------ main


def _compare(
    template_path: Optional[str],
    revised_path: Optional[str],
    cache: dict,
    use_sample: bool,
) -> Iterator[Tuple[str, str, dict]]:
    """Drive the whole pipeline, streaming (status_html, report_html, cache)."""
    if use_sample:
        template_path, revised_path = str(SAMPLE_TEMPLATE), str(SAMPLE_REVISED)
    else:
        if not template_path or not revised_path:
            yield _note("err", "Upload both a template and a revised document, "
                               "or load the sample."), "", cache
            return
        for path, label in ((template_path, "Template"), (revised_path, "Revised contract")):
            problem = _check_type(path, label) or _check_size(path, label)
            if problem:
                yield _note("err", html.escape(problem)), "", cache
                return

    yield _note("info", "Reading documents…"), "", cache

    try:
        t_text, t_src = extract_text(template_path)
        r_text, r_src = extract_text(revised_path)
    except Exception as exc:  # noqa: BLE001 — surface parse/OCR failure in the UI
        yield _note("err", f"Could not read the documents: {html.escape(str(exc))}"), "", cache
        return

    header = (
        ("Loaded sample NDA.<br>" if use_sample else "")
        + f"Template detected as <b>{SOURCE_LABEL[t_src]}</b>.<br>"
        + f"Revised detected as <b>{SOURCE_LABEL[r_src]}</b>."
    )

    template_clauses = segment(t_text)
    revised_clauses = segment(r_text)
    header += (
        f"<br>Segmented {len(template_clauses)} template clauses and "
        f"{len(revised_clauses)} revised clauses."
    )
    pairs = align(template_clauses, revised_clauses)

    if not pairs:
        yield _note("err", "No clauses were detected in these documents."), "", cache
        return

    verdicts: List[ClauseVerdict] = []
    for verdicts, progress in run_comparison(pairs, llm_configured(), cache):
        yield _note("info", header) + progress, "", cache

    yield _note("info", header), render_report(verdicts), cache


def _on_compare(template_path, revised_path, cache):
    yield from _compare(template_path, revised_path, cache, use_sample=False)


def _on_sample(cache):
    yield from _compare(None, None, cache, use_sample=True)


def build_demo() -> gr.Blocks:
    # NB: Gradio 6 moved `theme` and `css` off the Blocks constructor — they are
    # passed to launch() below. Setting them here is silently ignored (warning only).
    with gr.Blocks(title="Legal Document Comparison Tool") as demo:
        gr.HTML("<div class='ldc-title'>⚖️ Legal Document Comparison Tool</div>")

        if llm_configured():
            gr.HTML(_note("ok", f"LLM configured — model "
                                f"<code>{html.escape(os.environ.get('LLM_MODEL', '?'))}</code>."))
        else:
            gr.HTML(_note(
                "warn",
                "Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL to run comparisons. "
                "You can still browse the UI with sample data loaded (clauses are "
                "compared with an offline text-similarity heuristic).",
            ))

        with gr.Row():
            template_in = gr.File(
                label="Template (original)", file_types=UPLOAD_TYPES, type="filepath"
            )
            revised_in = gr.File(
                label="Revised contract", file_types=UPLOAD_TYPES, type="filepath"
            )

        with gr.Row():
            compare_btn = gr.Button("Compare", variant="primary")
            sample_btn = gr.Button("Load sample contracts")

        cache_state = gr.State({})
        status_out = gr.HTML()
        report_out = gr.HTML()

        compare_btn.click(
            _on_compare,
            inputs=[template_in, revised_in, cache_state],
            outputs=[status_out, report_out, cache_state],
        )
        sample_btn.click(
            _on_sample,
            inputs=[cache_state],
            outputs=[status_out, report_out, cache_state],
        )

    return demo


if __name__ == "__main__":
    # Spaces serves on 7860; Render (and anything else injecting $PORT) overrides it.
    build_demo().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="stone"),
        css=_CSS,
        head=_FORCE_LIGHT,
    )
