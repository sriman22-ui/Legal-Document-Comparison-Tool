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
from typing import Literal, Tuple

# Average non-whitespace characters per page above which a PDF is treated as a
# DIGITAL document (use the extracted text directly, skip OCR). Below it, the
# document is treated as SCANNED and routed through OCR. Tunable in one place.
MIN_CHARS_PER_PAGE = 100

SourceType = Literal["digital", "scanned", "text"]


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
    "text" (a .txt file), "digital" (a PDF with selectable text), or
    "scanned" (an image PDF that required OCR).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read(), "text"

    if ext == ".pdf":
        return _extract_pdf(path)

    raise ValueError(f"Unsupported file type '{ext}'. Only .txt and .pdf are supported.")


def _extract_pdf(path: str) -> Tuple[str, SourceType]:
    """Extract a PDF, choosing native extraction or OCR based on text density."""
    import fitz  # PyMuPDF, pulled in by pymupdf4llm
    import pymupdf4llm

    extracted = pymupdf4llm.to_markdown(path)

    doc = fitz.open(path)
    num_pages = doc.page_count
    doc.close()

    source = classify_pdf_source(extracted, num_pages)
    if source == "digital":
        return extracted, "digital"

    # Near-empty selectable text -> almost certainly a scanned/image PDF.
    return _ocr_pdf(path), "scanned"


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
            if result:
                # Each result row is [box, text, confidence].
                page_texts.append("\n".join(row[1] for row in result))
    finally:
        doc.close()

    return "\n\n".join(page_texts)
