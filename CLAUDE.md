# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit app that compares two versions of a contract and produces a risk-aware
change report per clause (not a plain text diff). Six-stage pipeline, each stage its
own module in `src/`:

1. **Upload** — `.pdf`, `.txt`, or image (`.png`/`.jpg`/`.jpeg`/`.bmp`/`.tif`/`.tiff`/`.webp`).
2. **Parse & OCR** (`src/parsing.py`) — `extract_text(path) -> (text, source_type)`.
   `.txt` is read directly. PDFs are extracted natively with `pymupdf4llm` first;
   `classify_pdf_source` decides digital vs. scanned by measuring non-whitespace
   chars per page against `MIN_CHARS_PER_PAGE` — only genuinely scanned PDFs (and
   raw images) fall through to `rapidocr-onnxruntime` OCR. `fitz`/`pymupdf4llm`/
   `rapidocr_onnxruntime` are imported lazily inside the functions that need them so
   the module (and the test suite) stays importable without those heavy deps.
3. **Segment** (`src/segmentation.py`) — `segment(text) -> List[Clause]`. Recognises
   numbered (`1.`, `1.1`), named (`Section 4`, `ARTICLE V`, `Clause 1.` — tolerant of
   OCR glyph damage like l/I/1 confusion), and all-caps headings; all-caps is only a
   fallback when there's no numbered/named structure, so document titles aren't
   mis-read as clauses. `_NON_CLAUSE_MARKERS` filters OCR furniture (watermarks,
   "DRAFT", "CONFIDENTIAL", etc.) out of the all-caps fallback.
4. **Align** (`src/alignment.py`) — `align(template_clauses, revised_clauses) -> List[AlignedPair]`.
   Greedy best-match per template clause by heading similarity (`SequenceMatcher`
   ratio, plus a token-containment score so an expanded heading like "Assignment" →
   "Assignment and Subletting" still matches). The clause id is only a small
   tiebreaker (`_ID_MATCH_BONUS`) — never enough alone to force a match, so a
   renumbered clause after a deletion still aligns correctly. Unmatched template
   clauses are deletions; unmatched revised clauses are additions.
5. **Compare** (`src/comparison.py`) — one LLM call per aligned clause pair (never
   whole documents) via an OpenAI-compatible client (`get_client`/`get_model`, built
   from `LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL` env vars — works with Groq,
   OpenRouter, Gemini's OpenAI-compat endpoint, local Ollama, etc). Retries with
   exponential backoff (`_MAX_ATTEMPTS`, `_backoff_seconds`); if JSON still can't be
   parsed, fails safe into a `meaning_changed`/`medium` verdict flagged for manual
   review rather than crashing. `heuristic_verdict` is a no-LLM offline fallback
   (pure text-similarity via `SequenceMatcher`) used when no API key is configured,
   so the UI stays fully browsable. Added/deleted clauses are labelled by
   `verdict_for_unmatched` without any LLM call.
6. **Report** (`app.py`) — risk-summary table sorted high→low plus an expandable
   colour-coded side-by-side view per clause.

`src/schema.py` defines the shared pydantic models: `Clause` and `ClauseVerdict`
(`change_type`: unchanged/reworded_same_meaning/meaning_changed/added/deleted;
`risk_level`: none/low/medium/high).

## Commands

```bash
# setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # fill in LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

# run the app
streamlit run app.py

# run all tests
pytest

# run a single test
pytest tests/test_comparison.py::test_alignment_flags_the_deleted_injunctive_relief_clause -v
```

There's a Claude Code launch config at `.claude/launch.json` (`legal-doc-compare`,
port 8501) for previewing the app.

## Deployment

Deployed to Render as a native Python web service, defined by `render.yaml`
(Blueprint). The start command binds to Render's injected `$PORT`; headless mode,
XSRF, and a 30MB `maxUploadSize` cap come from `.streamlit/config.toml`. The three
`LLM_` env vars are `sync: false` in `render.yaml` and must be set in the Render
dashboard. `requirements.txt` is pinned to exact versions for reproducible builds.
The `starter` plan is specified deliberately — the free tier's 512MB RAM is not
enough once the OCR (`rapidocr-onnxruntime`) models load. There is no CI/CD pipeline
and no authentication on the deployed app by design.

## Key implementation notes

- **`.env` is read at import time and cached by `python-dotenv`.** `app.py` loads it
  with `override=True` so editing `.env` mid-session (e.g. switching `LLM_MODEL`)
  takes effect on the next Streamlit rerun — but the app process itself must still be
  restarted to pick up changes made *before* first launch. Without all three
  `LLM_` vars set, the app silently falls back to the offline heuristic
  (`llm_configured()` in `app.py` gates this).
- **The live LLM API is never called in tests.** `tests/test_comparison.py` mocks the
  OpenAI client entirely (`MagicMock`) — `compare_clause` is unit-tested against
  canned JSON responses, including malformed ones to exercise the fail-safe path.
- **Verdicts are cached in `st.session_state`** keyed by `(template_text,
  revised_text)` so Streamlit reruns (e.g. from widget interaction) don't re-call the
  LLM for clause pairs already compared.
- **`conftest.py`** exists solely so pytest adds the project root to `sys.path`,
  making `from src...` imports resolve when running `pytest` from the repo root.
- Sample data lives in `data/`: `sample_template.txt`/`sample_revised.txt` (an 8→7
  clause NDA with deliberate edits of each `change_type`/`risk_level` combination —
  see README for exactly which clause exercises which case) and
  `sample_service_template.png`/`sample_service_revised.png` (a rendered image MSA,
  for exercising the OCR path end-to-end).
