"""
Document processing: text extraction from various file formats.

Strategy: for each format a lean, fast extractor based on libraries that
are installed anyway (pdfminer, python-docx, python-pptx, openpyxl) is
tried first. Only if it fails does processing fall back to the heavy
unstructured library (optional, see requirements-optional.txt).
"""

import os
import re
import html as html_lib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import ProcessorConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    filename: str
    content: str
    token_count: int
    page_count: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class DocumentProcessor:
    """Extract text from documents — fast, with a fallback to unstructured."""

    def __init__(self, config: ProcessorConfig):
        self.config = config

    def process_document(self, file_path: str) -> ProcessedDocument:
        filename = os.path.basename(file_path)
        suffix = Path(file_path).suffix.lower()
        t0 = time.monotonic()

        try:
            # ── Step 1: fast extraction ──────────────────────
            content = self._extract_fast(file_path, suffix)

            # ── Step 2: fall back to unstructured ────────────────
            if content is None:
                logger.info(f"Fast extractor for {suffix} unavailable/failed, "
                            f"using unstructured for {filename}")
                content = self._extract_with_unstructured(file_path)

            # ── Clean-up ─────────────────────────────────────────
            content = self._clean_content(content)

            # Token estimate (~3 characters per token for German text)
            token_count = max(1, len(content) // 3)

            elapsed = time.monotonic() - t0
            logger.info(f"Processed: {filename} ({suffix}) in {elapsed:.1f}s → {token_count:,} tokens")

            return ProcessedDocument(
                filename=filename,
                content=content,
                token_count=token_count,
            )

        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(f"Error processing {filename} after {elapsed:.1f}s: {e}")
            raise ValueError(f"Document could not be processed: {e}")

    # =================================================================
    # Fast extractors (no unstructured)
    # =================================================================

    def _extract_fast(self, file_path: str, suffix: str) -> Optional[str]:
        """Try text extraction with lightweight libraries.
        Returns None if there is no suitable extractor or it fails.
        """
        extractors = {
            ".txt":  self._read_text_file,
            ".md":   self._read_text_file,
            ".rst":  self._read_text_file,
            ".py":   self._read_text_file,
            ".csv":  self._read_text_file,
            ".pdf":  self._extract_pdf_fast,
            ".docx": self._extract_docx_fast,
            ".doc":  None,  # no fast extractor → unstructured
            ".pptx": self._extract_pptx_fast,
            ".ppt":  None,  # no fast extractor → unstructured
            ".xlsx": self._extract_xlsx_fast,
            ".xls":  None,  # no fast extractor → unstructured
            ".html": self._extract_html_fast,
            ".htm":  self._extract_html_fast,
            ".rtf":  self._read_text_file,  # read RTF as plain text (markup remains, but fast)
        }

        extractor = extractors.get(suffix)
        if extractor is None:
            return None

        try:
            content = extractor(file_path)
            if content and content.strip():
                return content
            logger.warning(f"Fast extractor for {suffix} returned empty content")
            return None
        except Exception as e:
            logger.warning(f"Fast extractor for {suffix} failed: {e}")
            return None

    def _read_text_file(self, file_path: str) -> str:
        encodings = ["utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError("File encoding not recognised.")

    def _extract_pdf_fast(self, file_path: str) -> str:
        """PDF text extraction with pdfminer.six (fast, no layout model)."""
        from pdfminer.high_level import extract_text
        text = extract_text(file_path)
        return text

    def _extract_docx_fast(self, file_path: str) -> str:
        """DOCX text extraction with python-docx."""
        from docx import Document
        doc = Document(file_path)
        parts = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)

        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        return "\n\n".join(parts)

    def _extract_pptx_fast(self, file_path: str) -> str:
        """PPTX text extraction with python-pptx."""
        from pptx import Presentation
        prs = Presentation(file_path)
        parts = []

        for i, slide in enumerate(prs.slides, 1):
            slide_texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            slide_texts.append(text)
                if shape.has_table:
                    for row in shape.table.rows:
                        cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if cells:
                            slide_texts.append(" | ".join(cells))

            if slide_texts:
                parts.append(f"--- Slide {i} ---\n" + "\n".join(slide_texts))

        return "\n\n".join(parts)

    def _extract_xlsx_fast(self, file_path: str) -> str:
        """XLSX text extraction with openpyxl."""
        from openpyxl import load_workbook
        wb = load_workbook(file_path, read_only=True, data_only=True)
        parts = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            sheet_parts = [f"--- Sheet: {sheet_name} ---"]
            for row in ws.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    sheet_parts.append(" | ".join(cells))
            if len(sheet_parts) > 1:  # more than just the header
                parts.append("\n".join(sheet_parts))

        wb.close()
        return "\n\n".join(parts)

    def _extract_html_fast(self, file_path: str) -> str:
        """HTML text extraction: remove tags, keep text."""
        raw = self._read_text_file(file_path)

        # Remove script and style blocks
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode HTML entities
        text = html_lib.unescape(text)
        # Clean up multiple whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    # =================================================================
    # Fallback: unstructured (heavy, but universal)
    # =================================================================

    def _extract_with_unstructured(self, file_path: str) -> str:
        """Extract text with the format-specific unstructured partitioners."""
        suffix = Path(file_path).suffix.lower()
        elements = None

        try:
            if suffix == ".pdf":
                from unstructured.partition.pdf import partition_pdf
                elements = partition_pdf(
                    filename=file_path,
                    strategy=self.config.pdf_strategy,
                )

            elif suffix in (".docx", ".doc"):
                from unstructured.partition.docx import partition_docx
                elements = partition_docx(filename=file_path)

            elif suffix in (".pptx", ".ppt"):
                from unstructured.partition.pptx import partition_pptx
                elements = partition_pptx(filename=file_path)

            elif suffix in (".xlsx", ".xls"):
                from unstructured.partition.xlsx import partition_xlsx
                elements = partition_xlsx(filename=file_path)

            elif suffix in (".html", ".htm"):
                from unstructured.partition.html import partition_html
                elements = partition_html(filename=file_path)

            elif suffix == ".rtf":
                from unstructured.partition.rtf import partition_rtf
                elements = partition_rtf(filename=file_path)

            else:
                # Fallback: auto-partition for unknown formats
                from unstructured.partition.auto import partition
                elements = partition(filename=file_path)

        except ImportError as e:
            raise ImportError(
                f"Dependency for processing {suffix} is missing: {e}. "
                f"Install with: pip install 'unstructured[{suffix.lstrip('.')}]'"
            )

        if not elements:
            raise ValueError("No content found in the document.")

        parts = []
        for el in elements:
            text = str(el).strip()
            if text:
                parts.append(text)

        return "\n\n".join(parts)

    # =================================================================
    # Clean-up
    # =================================================================

    def _clean_content(self, content: str) -> str:
        if not content:
            return ""

        # Reduce multiple empty lines
        content = re.sub(r"\n{4,}", "\n\n\n", content)

        # Reduce multiple spaces
        content = re.sub(r"[ \t]{3,}", "  ", content)

        # Remove header/footer (optional)
        if self.config.remove_page_numbers:
            content = re.sub(
                r"(?m)^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", "", content
            )

        # Remove paragraphs that are too short (optional)
        if self.config.min_paragraph_length > 0:
            paragraphs = content.split("\n\n")
            filtered = []
            for p in paragraphs:
                stripped = p.strip()
                if len(stripped) >= self.config.min_paragraph_length or not stripped:
                    filtered.append(p)
                elif re.match(r"^[#=\-]", stripped):
                    # keep headings
                    filtered.append(p)
            content = "\n\n".join(filtered)

        return content.strip()
