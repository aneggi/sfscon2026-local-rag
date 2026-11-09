"""Extract text and split it into overlapping passages with page references."""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int | None


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Prefer a whitespace boundary; hard-split long strings so progress is guaranteed."""
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size.")
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start + overlap:
                end = boundary
        passage = text[start:end].strip()
        if passage:
            chunks.append(passage)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def extract_chunks(path: Path, size: int, overlap: int) -> list[Chunk]:
    # A source may have a misleading filename (for example, report.pdf.md).
    with path.open("rb") as file:
        has_pdf_header = file.read(5) == b"%PDF-"
    if has_pdf_header or path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are unsupported. Supply a decrypted document.")
        pages = [(page.extract_text() or "", number) for number, page in enumerate(reader.pages, 1)]
    else:
        text = path.read_text(encoding="utf-8-sig")
        if text.lstrip().lower().startswith(("<!doctype html", "<html")):
            raise ValueError("This file is HTML. Configure a direct TXT, Markdown, or PDF URL.")
        pages = [(text, None)]
    chunks = [
        Chunk(passage, page) for text, page in pages for passage in split_text(text, size, overlap)
    ]
    if not chunks:
        raise ValueError("No extractable text. Scanned PDFs need OCR before ingestion.")
    return chunks
