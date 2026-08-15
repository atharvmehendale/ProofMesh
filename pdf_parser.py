"""
core/pdf_parser.py

PDF ingestion for ProofMesh. Two-tier strategy:
  1. Try the PDF's actual text layer (pypdf) - fast, free, no model call.
  2. Only if that comes back empty (scanned/photographed PDF, no text
     layer) fall back to rendering pages as images and running vision-model
     OCR (models/featherless_client.run_ocr) - slower and costs credits,
     so it's a fallback, not the default path.

get_derivation_text() is the one function app.py should call - it handles
the tier decision internally so the caller doesn't need to know which path
was used, though it does report back which one it took (useful to show in
the UI so the user isn't confused about why a scanned PDF is slower).
"""

from __future__ import annotations

import base64

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import pymupdf as fitz  # PyMuPDF's current import name - 'import fitz' is
                             # deprecated and warns on the pymupdf version this
                             # was tested against; renders PDF pages to images
                             # for OCR
except ImportError:
    fitz = None

from featherless_client import run_ocr

PDF_CHAR_LIMIT = 15000   # how much extracted text we actually pass to extraction
OCR_MAX_PAGES = 6        # cap on scanned pages sent through vision OCR -
                          # otherwise rate limits (and credits) get blown fast


def extract_pdf_text(file, max_chars: int = PDF_CHAR_LIMIT) -> tuple[str, int]:
    """Reads text from the PDF's actual text layer. Returns
    (text_truncated_to_max_chars, total_length_before_truncation) so the
    caller can tell if it got cut off. Empty string means no usable text
    layer (scanned pages) - caller should fall back to OCR, not treat this
    as an error."""
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf")
    file.seek(0)  # Streamlit reruns the whole script on every widget
                  # interaction - without seeking back to 0, a second read
                  # of the same uploaded file object returns nothing
    reader = PdfReader(file)
    chunks = [page.extract_text() or "" for page in reader.pages]
    full = "\n".join(chunks).strip()
    return full[:max_chars], len(full)


def ocr_pdf_text(secrets: dict, file, max_chars: int = PDF_CHAR_LIMIT,
                  max_pages: int = OCR_MAX_PAGES) -> tuple[str, int, int]:
    """Fallback for scanned/photographed PDFs with no text layer. Renders
    each page to a PNG and asks a vision model to transcribe it - avoids
    needing a separate OCR engine installed on top of everything else.
    Returns (text, pages_actually_read, total_pages_in_doc)."""
    if fitz is None:
        raise RuntimeError("pymupdf is not installed. Run: pip install pymupdf")
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    total_pages = len(doc)
    pages_to_read = min(total_pages, max_pages)

    transcripts = []
    for i in range(pages_to_read):
        page = doc[i]
        pix = page.get_pixmap(dpi=150)
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        transcripts.append(run_ocr(secrets, b64))

    full = "\n\n".join(transcripts).strip()
    return full[:max_chars], pages_to_read, total_pages


def get_derivation_text(secrets: dict, file, max_chars: int = PDF_CHAR_LIMIT) -> tuple[str, dict]:
    """Single entry point for app.py. Tries the text layer first and only
    pays for OCR if that comes back empty. Returns (text, info) where info
    reports which path was taken - app.py should surface this in the UI
    (e.g. 'OCR'd 4 of 12 pages') so a slow scanned-PDF run doesn't look
    like the app is just being slow for no reason."""
    text, total_len = extract_pdf_text(file, max_chars)
    if text.strip():
        return text, {"method": "text_layer", "chars_found": total_len}

    ocr_text, pages_read, total_pages = ocr_pdf_text(secrets, file, max_chars)
    if not ocr_text.strip():
        raise RuntimeError(
            "No text layer found, and OCR found no readable text on the "
            "pages checked. This PDF may be blank, corrupted, or use "
            "notation the vision model couldn't transcribe."
        )
    info = {"method": "ocr", "pages_read": pages_read, "total_pages": total_pages}
    return ocr_text, info
