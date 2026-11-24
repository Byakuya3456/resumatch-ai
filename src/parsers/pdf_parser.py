"""
pdf_parser.py — Multi-strategy PDF text extraction.

Strategy order (best quality first):
  1. pymupdf4llm  → Markdown-rich output, preserves layout / tables
  2. pdfplumber   → Structured text + table extraction
  3. PyPDF2       → Legacy fallback (old behaviour, kept for safety)

If none extract meaningful text (scanned PDF), a warning is printed
and an empty string is returned. Add surya-ocr here in a future sprint
for full OCR support on image-only PDFs.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Attempt to import preferred libraries ──────────────────────────────────

try:
    import pymupdf4llm  # type: ignore
    _HAS_PYMUPDF4LLM = True
except ImportError:
    _HAS_PYMUPDF4LLM = False
    logger.warning(
        "pymupdf4llm not installed — falling back to pdfplumber/PyPDF2. "
        "Run: pip install pymupdf4llm"
    )

try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False
    logger.warning(
        "pdfplumber not installed. Run: pip install pdfplumber"
    )

try:
    import PyPDF2  # type: ignore
    _HAS_PYPDF2 = True
except ImportError:
    _HAS_PYPDF2 = False


# ── OCR placeholder ────────────────────────────────────────────────────────

def _try_ocr_fallback(file_path: Path) -> str:
    """
    Placeholder for surya-ocr / Tesseract OCR on scanned PDFs.
    Install surya-ocr and uncomment the implementation block below
    when you need to handle image-only resume PDFs.

    pip install surya-ocr
    """
    # ── surya-ocr (uncomment when installed) ──────────────────────────
    # try:
    #     from surya.ocr import run_ocr
    #     from surya.model.detection import segformer
    #     from surya.model.recognition.model import load_model
    #     from surya.model.recognition.processor import load_processor
    #     import fitz  # PyMuPDF
    #
    #     doc = fitz.open(str(file_path))
    #     images = [page.get_pixmap() for page in doc]
    #     det_model, det_processor = segformer.load_model(), segformer.load_processor()
    #     rec_model = load_model()
    #     rec_processor = load_processor()
    #     results = run_ocr(images, [["en"]] * len(images), det_model, det_processor,
    #                       rec_model, rec_processor)
    #     return "\n".join(
    #         line.text for page in results for line in page.text_lines
    #     )
    # except Exception as exc:
    #     logger.error("surya-ocr failed: %s", exc)
    #     return ""

    logger.warning(
        "PDF appears to be scanned (image-only). "
        "Install surya-ocr for OCR support: pip install surya-ocr"
    )
    return ""


# ── Main extraction function ───────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> dict:
    """
    Extract text from a PDF using the best available library.

    Returns:
        {
            "text":   str   — extracted text (Markdown when pymupdf4llm),
            "method": str   — which library succeeded,
            "tables": list  — extracted tables (pdfplumber only),
            "pages":  int   — page count,
        }
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # ── Strategy 1: pymupdf4llm ────────────────────────────────────────
    if _HAS_PYMUPDF4LLM:
        try:
            md_text: str = pymupdf4llm.to_markdown(str(path))
            md_text = md_text.strip()
            if len(md_text) >= 100:
                logger.info("PDF extracted via pymupdf4llm (%d chars)", len(md_text))
                return {
                    "text": md_text,
                    "method": "pymupdf4llm",
                    "tables": [],
                    "pages": _count_pages_fast(path),
                }
            logger.debug("pymupdf4llm yielded too little text, trying next strategy")
        except Exception as exc:
            logger.warning("pymupdf4llm failed: %s", exc)

    # ── Strategy 2: pdfplumber ─────────────────────────────────────────
    if _HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(path) as pdf:
                pages_text: list[str] = []
                all_tables: list[list] = []

                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    pages_text.append(page_text)
                    all_tables.extend(page.extract_tables() or [])

                combined = "\n\n".join(pages_text).strip()
                if len(combined) >= 100:
                    logger.info(
                        "PDF extracted via pdfplumber (%d chars, %d tables)",
                        len(combined), len(all_tables),
                    )
                    return {
                        "text": combined,
                        "method": "pdfplumber",
                        "tables": all_tables,
                        "pages": len(pdf.pages),
                    }
            logger.debug("pdfplumber yielded too little text, trying PyPDF2")
        except Exception as exc:
            logger.warning("pdfplumber failed: %s", exc)

    # ── Strategy 3: PyPDF2 (legacy fallback) ──────────────────────────
    if _HAS_PYPDF2:
        try:
            text_parts: list[str] = []
            with open(path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
            combined = "\n".join(text_parts).strip()
            if len(combined) >= 100:
                logger.info("PDF extracted via PyPDF2 (%d chars)", len(combined))
                return {
                    "text": combined,
                    "method": "PyPDF2",
                    "tables": [],
                    "pages": len(reader.pages),
                }
        except Exception as exc:
            logger.warning("PyPDF2 failed: %s", exc)

    # ── Strategy 4: OCR fallback (scanned PDF) ─────────────────────────
    ocr_text = _try_ocr_fallback(path)
    return {
        "text": ocr_text,
        "method": "ocr_fallback",
        "tables": [],
        "pages": _count_pages_fast(path),
    }


def _count_pages_fast(path: Path) -> int:
    """Quick page-count using PyMuPDF if available, else 0."""
    try:
        import fitz  # PyMuPDF (installed as dependency of pymupdf4llm)
        with fitz.open(str(path)) as doc:
            return len(doc)
    except Exception:
        return 0
