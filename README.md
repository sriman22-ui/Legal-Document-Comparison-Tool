#  Legal Document Comparison Tool

Compare two versions of a contract and produce a **risk-aware change report** — not
just a text diff. For each clause the tool classifies the change by **meaning** and by
**risk**, so you can ignore *"reworded but same meaning"* edits and focus on the ones
where the *meaning materially changed*. This is the distinction Microsoft Word's
Compare cannot make.

## What it does

A six-stage pipeline:

1. **Upload** two documents (template vs. revised). Accepts `.pdf`, `.txt`, and image
   files (`.png`, `.jpg`/`.jpeg`, `.bmp`, `.tif`/`.tiff`, `.webp`).
2. **Parse & OCR** — detects digital vs scanned PDFs *before* doing any OCR. Digital
   PDFs use native text extraction; only genuinely scanned/image PDFs are OCR'd.
   Image uploads are always OCR'd directly.
3. **Segment** each document into clauses using numbered/named/all-caps headings.
4. **Align** clauses between the two documents by heading similarity (renumbered
   clauses still match; missing ones are flagged).
5. **Compare** each aligned clause pair with a *separate* LLM call returning typed
   JSON — only clause pairs are sent to the model, never whole documents.
6. **Report** a risk-summary table (sorted high→low) plus an expandable, colour-coded
   side-by-side view per clause.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env          # then fill in the three LLM_ values
streamlit run app.py
```

> **⚠️ Before running real comparisons, you must set all three `LLM_` values in `.env`:**
> `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`. The app reads these on startup, so
> open `.env` and fill them in *before* you launch (`streamlit run app.py`). If any of
> the three is blank, the app falls back to the offline heuristic and will **not** call
> the LLM. See [Choosing a free provider](#choosing-a-free-provider) for the exact
> values to paste in. If you change `.env` while the app is already running, restart it.

Without any keys configured you can still click **Load sample contracts** to browse the
full UI — aligned clauses are compared with an offline text-similarity heuristic, and
added/deleted clauses are labelled exactly as they are in LLM mode.

## Choosing a free provider

Configure the client entirely from environment variables. The `openai` SDK works with
any OpenAI-compatible endpoint, which is what nearly every free provider exposes. Pick
one and paste its values into `.env`:

| Provider | `LLM_BASE_URL` | Sample `LLM_MODEL` | Notes |
|---|---|---|---|
| **Groq** | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` | Fast, generous free tier. Get a key at console.groq.com. |
| **OpenRouter** | `https://openrouter.ai/api/v1` | `meta-llama/llama-3.3-70b-instruct:free` | Many `:free` model ids. Key at openrouter.ai/keys. |
| **Google Gemini** (OpenAI-compat) | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` | Free tier via Google AI Studio. |
| **Ollama** (local) | `http://localhost:11434/v1` | `llama3.1` | Fully local; `LLM_API_KEY` can be any non-empty string. |

Example `.env` for Groq:

```env
LLM_API_KEY=gsk_your_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

## Sample data

`data/sample_template.txt` is a short 8-clause NDA. `data/sample_revised.txt` is the
same NDA with exactly these deliberate edits:

- **Confidentiality Obligation** weakened (allows disclosure to affiliates/advisors
  without consent, lowers the care standard) → *meaning_changed / high*.
- **Term** changed from five years to two years → *meaning_changed / medium*.
- **Remedies and Injunctive Relief** clause **deleted** (and the doc renumbered) →
  *deleted / high*.
- **No License** reworded with the same meaning (a control) → *reworded_same_meaning /
  none*.

### Scanned-image samples (to try OCR)

`data/sample_service_template.png` and `data/sample_service_revised.png` are a 3-clause
Master Service Agreement rendered as **images** (no selectable text), so uploading them
exercises the OCR path. The revised version changes clause 1 (payment window 30→60 days,
interest removed) and clause 2 (term 5→2 years) in *meaning*, and merely rewords clause 3
(Governing Law) — a quick way to see the risk report on OCR'd input.

## Running the tests

```bash
pytest
```

Tests cover `.txt` parsing, the digital-vs-scanned threshold, segmentation into the
right number of clauses, alignment correctly flagging the deleted clause, and the
comparison call (with a **mocked** LLM client — the live API is never called).

## Key design decisions

- **Detect before OCR.** PDFs are first extracted natively with `pymupdf4llm`. We only
  fall back to `rapidocr-onnxruntime` when the average non-whitespace characters per
  page falls below `MIN_CHARS_PER_PAGE` (in `src/parsing.py`). OCR is slow; most PDFs
  don't need it.
- **Heading-similarity alignment.** Headings dominate the alignment score so that
  renumbered clauses (common after a deletion) still align; the clause number is only a
  tiny tiebreaker, never enough on its own to force a wrong match.
- **One clause per LLM call.** Each aligned pair is compared in isolation. Whole
  documents are never sent to the model — cheaper, more reliable JSON, and no context
  bleed between clauses.
- **Fail safe, never crash.** The LLM call retries with exponential backoff (free
  providers rate-limit). If the response still can't be parsed, the clause is returned
  as `meaning_changed / medium` flagged for manual review rather than crashing the run.
- **Typed structured output.** Responses are validated into a pydantic `ClauseVerdict`;
  an invalid `change_type`/`risk_level` triggers a retry.
- **Caching.** Verdicts are cached in `st.session_state` keyed by
  `(template_text, revised_text)` so Streamlit reruns don't re-call the API.

## Stretch goals

The MVP above is complete and runnable on the sample data. Stretch items from the spec
(OCR verification on a scanned sample, embedding-based semantic alignment, evaluation
script with precision/recall, PDF export) are intended to be added behind their own
flags without disturbing the default heading-based path.

## Project layout

```
legal-doc-compare/
  app.py                 # Streamlit UI
  src/
    parsing.py           # PDF/OCR/txt -> raw text + detected source type
    segmentation.py      # text -> List[Clause]
    alignment.py         # two clause lists -> aligned pairs + unmatched
    comparison.py        # LLM per-clause comparison -> ClauseVerdict
    schema.py            # pydantic models
  data/                  # sample NDA (template + revised)
  tests/                 # parsing + segmentation + alignment + comparison tests
  requirements.txt
  .env.example
  README.md
```
