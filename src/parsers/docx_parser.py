"""
docx_parser.py — DOCX text extraction.

Strategy order:
  1. docling (IBM)   → Preserves tables, columns, images alt-text → Markdown
  2. python-docx     → Reliable, extracts paragraphs + tables cleanly
  3. docx2txt        → Minimal legacy fallback
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Library availability checks ────────────────────────────────────────────

try:
    from docling.document_converter import DocumentConverter  # type: ignore
    _HAS_DOCLING = True
    _docling_converter = DocumentConverter()
except ImportError:
    _HAS_DOCLING = False
    logger.warning(
        "docling not installed — falling back to python-docx. "
        "Run: pip install docling"
    )

try:
    from docx import Document as DocxDocument  # type: ignore  (python-docx)
    _HAS_PYTHON_DOCX = True
except ImportError:
    _HAS_PYTHON_DOCX = False

try:
    import docx2txt  # type: ignore
    _HAS_DOCX2TXT = True
except ImportError:
    _HAS_DOCX2TXT = False


# ── Main extraction function ───────────────────────────────────────────────

def extract_text_from_docx(file_path: str) -> dict:
    """
    Extract text from a DOCX file using the best available library.

    Returns:
        {
            "text":   str  — extracted text (Markdown when docling),
            "method": str  — which library succeeded,
            "tables": list — extracted table rows (python-docx only),
        }
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX not found: {file_path}")

    # ── Strategy 1: docling ────────────────────────────────────────────
    if _HAS_DOCLING:
        try:
            result = _docling_converter.convert(str(path))
            md_text = result.document.export_to_markdown().strip()
            if len(md_text) >= 100:
                logger.info("DOCX extracted via docling (%d chars)", len(md_text))
                return {
                    "text": md_text,
                    "method": "docling",
                    "tables": [],   # docling embeds table content in markdown
                }
            logger.debug("docling yielded too little text, trying python-docx")
        except Exception as exc:
            logger.warning("docling failed: %s", exc)

    # ── Strategy 2: python-docx ────────────────────────────────────────
    if _HAS_PYTHON_DOCX:
        try:
            doc = DocxDocument(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

            # Extract table cells row-by-row
            tables_text: list[str] = []
            raw_tables: list[list[list[str]]] = []
            for table in doc.tables:
                table_rows: list[list[str]] = []
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        table_rows.append(cells)
                        tables_text.append(" | ".join(cells))
                raw_tables.append(table_rows)

            full_text = "\n".join(paragraphs + tables_text).strip()
            if len(full_text) >= 50:
                logger.info(
                    "DOCX extracted via python-docx (%d chars, %d tables)",
                    len(full_text), len(raw_tables),
                )
                return {
                    "text": full_text,
                    "method": "python-docx",
                    "tables": raw_tables,
                }
            logger.debug("python-docx yielded too little text, trying docx2txt")
        except Exception as exc:
            logger.warning("python-docx failed: %s", exc)

    # ── Strategy 3: docx2txt (minimal fallback) ────────────────────────
    if _HAS_DOCX2TXT:
        try:
            text = docx2txt.process(str(path)).strip()
            if text:
                logger.info("DOCX extracted via docx2txt (%d chars)", len(text))
                return {"text": text, "method": "docx2txt", "tables": []}
        except Exception as exc:
            logger.warning("docx2txt failed: %s", exc)

    logger.error("All DOCX extraction strategies failed for: %s", path)
    return {"text": "", "method": "failed", "tables": []}
