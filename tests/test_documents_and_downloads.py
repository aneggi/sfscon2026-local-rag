from dataclasses import replace

import httpx
import pytest

from local_rag.config import load_sources
from local_rag.documents import extract_chunks, split_text
from local_rag.download import download_documents


@pytest.mark.parametrize("size,overlap", [(30, 8), (1, 0), (30, 29)])
def test_chunking_preserves_text_with_whitespace_or_very_long_words(size, overlap):
    for text in ("a " * 80, "x" * 300):
        chunks = split_text(text, size, overlap)
        assert chunks
        assert all(0 < len(chunk) <= size for chunk in chunks)
        assert chunks[-1].endswith(text.strip()[-1])


def test_scanned_pdf_reports_that_ocr_is_needed(tmp_path):
    from pypdf import PdfWriter

    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(path)
    with pytest.raises(ValueError, match="OCR"):
        extract_chunks(path, 1000, 150)


@pytest.mark.parametrize("filename", ["two-pages.pdf", "two-pages.pdf.md"])
def test_pdf_chunks_retain_page_numbers(tmp_path, filename):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    for text in ("First page contains open data.", "Second page describes redistribution."):
        page = writer.add_blank_page(width=200, height=200)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            }
        )
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(content)
    path = tmp_path / filename
    writer.write(path)
    chunks = extract_chunks(path, 1000, 150)
    assert [(chunk.page, chunk.text) for chunk in chunks] == [
        (1, "First page contains open data."),
        (2, "Second page describes redistribution."),
    ]


@pytest.mark.parametrize(
    "filename", ["../outside.txt", "a/b.md", "a\\b.pdf", ".hidden.txt", "a.html"]
)
def test_source_filename_cannot_escape_documents_folder(settings, filename):
    escaped_name = filename.replace("\\", "\\\\")
    settings.sources_file.write_text(
        f'[[documents]]\nfilename = "{escaped_name}"\nurl = "https://example.org/test"\n'
    )
    with pytest.raises(ValueError):
        load_sources(settings.sources_file)


def configure_download(settings):
    settings.sources_file.write_text(
        '[[documents]]\nfilename = "source.txt"\nurl = "https://example.org/source.txt"\n'
    )


def mock_download(monkeypatch, response):
    original_client = httpx.Client
    monkeypatch.setattr(
        "local_rag.download.httpx.Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(lambda request: response)),
    )


def test_download_requires_explicit_overwrite(settings, monkeypatch):
    configure_download(settings)
    path = settings.documents_dir / "source.txt"
    path.write_text("Original content")
    mock_download(monkeypatch, httpx.Response(200, content=b"Updated content"))
    assert download_documents(settings)[0].startswith("Skipped")
    assert path.read_text() == "Original content"
    assert download_documents(settings, overwrite=True)[0].startswith("Downloaded")
    assert path.read_text() == "Updated content"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, content=b"<html>error</html>", headers={"content-type": "text/html"}),
        httpx.Response(200, content=b"x" * 100),
        httpx.Response(200, content=b""),
    ],
)
def test_failed_download_preserves_existing_document(settings, monkeypatch, response):
    configure_download(settings)
    path = settings.documents_dir / "source.txt"
    path.write_text("Original content")
    mock_download(monkeypatch, response)
    settings = replace(settings, maximum_download_bytes=20)
    assert download_documents(settings, overwrite=True)[0].startswith("ERROR")
    assert path.read_text() == "Original content"
    assert not list(settings.documents_dir.glob("*.part"))
