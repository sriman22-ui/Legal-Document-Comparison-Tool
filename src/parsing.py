"""Document parsing: PDF / OCR / txt -> raw text plus a detected source type.

Design notes
------------
* .txt files are read directly and are never OCR'd.
* .pdf files are first extracted with pymupdf4llm (fast, native, selectable text).
  We then DECIDE whether the PDF is digital or scanned by measuring how much
  selectable text we got per page. Only genuinely scanned/image PDFs fall through
  to the (much slower) rapidocr-onnxruntime OCR path.

Heavy optional dependencies (pymupdf4llm, fitz, rapidocr) are imported lazily inside
the functions that need them so that segmentation/alignment/tests stay importable
without them installed.
"""
from __future__ import annotations

import os
import re
from typing import Literal, Tuple

# Average non-whitespace characters per page above which a PDF is treated as a
# DIGITAL document (use the extracted text directly, skip OCR). Below it, the
# document is treated as SCANNED and routed through OCR. Tunable in one place.
MIN_CHARS_PER_PAGE = 100

SourceType = Literal["digital", "scanned", "text", "image"]

# Raster image formats are always OCR'd directly (no digital text to detect).
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# pymupdf4llm returns Markdown; these strip it back to plain text (see
# _clean_pdf_markdown) so heading detection isn't blocked by '#'/'**' prefixes.
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*")
_MD_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?")
_MD_EMPHASIS = re.compile(r"\*\*|__|\*|_")


def _clean_pdf_markdown(md: str) -> str:
    """Flatten pymupdf4llm Markdown to plain text.

    pymupdf4llm extracts digital PDFs as Markdown, so a heading arrives as
    ``## **Clause 1. ASSIGNMENT.**``. Segmentation keys off the *start* of each
    line, so the leading ``#``/``**`` would hide every heading (only a stray
    all-caps watermark would slip through the fallback). We strip heading markers,
    emphasis, ``<br>`` soft breaks, and the library's image-placeholder
    scaffolding so a digital PDF segments just like a plain ``.txt`` file.
    """
    md = md.replace("<br>", "\n")
    lines = []
    for raw in md.splitlines():
        if "intentionally omitted" in raw or "picture text" in raw:
            continue  # pymupdf4llm image placeholder scaffolding
        line = _MD_HEADING.sub("", raw)
        line = _MD_BLOCKQUOTE.sub("", line)
        line = _MD_EMPHASIS.sub("", line)
        lines.append(line.rstrip())
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _count_non_whitespace(text: str) -> int:
    """Number of non-whitespace characters — our proxy for 'real' selectable text."""
    return sum(1 for ch in text if not ch.isspace())


def classify_pdf_source(extracted_text: str, num_pages: int) -> SourceType:
    """Decide digital vs scanned from already-extracted text and the page count.

    Factored out from extract_text so the threshold logic can be unit-tested
    without a real PDF on disk.
    """
    pages = max(num_pages, 1)
    avg_chars_per_page = _count_non_whitespace(extracted_text) / pages
    return "digital" if avg_chars_per_page > MIN_CHARS_PER_PAGE else "scanned"


def extract_text(path: str) -> Tuple[str, SourceType]:
    """Extract raw text from a .txt or .pdf file.

    Returns (text, source_type) where source_type is one of
    "text" (a .txt file), "digital" (a PDF with selectable text),
    "scanned" (an image PDF that required OCR), or "image" (a raster image
    such as .png/.jpg that was OCR'd directly).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read(), "text"

    if ext == ".pdf":
        return _extract_pdf(path)

    if ext in IMAGE_EXTS:
        return _ocr_image(path), "image"

    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: .txt, .pdf, and images "
        f"({', '.join(sorted(IMAGE_EXTS))})."
    )


def _extract_pdf(path: str) -> Tuple[str, SourceType]:
    """Extract a PDF, choosing native extraction or OCR based on text density."""
    import fitz  # PyMuPDF, pulled in by pymupdf4llm
    import pymupdf4llm

    extracted = _clean_pdf_markdown(pymupdf4llm.to_markdown(path))

    doc = fitz.open(path)
    num_pages = doc.page_count
    doc.close()

    source = classify_pdf_source(extracted, num_pages)
    if source == "digital":
        return extracted, "digital"

    # Near-empty selectable text -> almost certainly a scanned/image PDF.
    return _ocr_pdf(path), "scanned"


def _ocr_rows_to_text(result) -> str:
    """Join RapidOCR rows into text, one detected line per row.

    Each result row is [box, text, confidence]; preserving one row per line keeps
    clause headings on their own line so segmentation can find them.
    """
    if not result:
        return ""
    return "\n".join(row[1] for row in result)


def _ocr_pdf(path: str) -> str:
    """Render each page to an image and OCR it with rapidocr-onnxruntime."""
    import fitz
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    doc = fitz.open(path)
    page_texts: list[str] = []
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            png_bytes = pix.tobytes("png")
            result, _elapsed = ocr(png_bytes)
            text = _ocr_rows_to_text(result)
            if text:
                page_texts.append(text)
    finally:
        doc.close()

    return "\n\n".join(page_texts)


def _ocr_image(path: str) -> str:
    """OCR a raster image file (.png/.jpg/...) directly with rapidocr-onnxruntime."""
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    result, _elapsed = ocr(path)
    return _ocr_rows_to_text(result)
