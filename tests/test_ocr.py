from pathlib import Path

import pytest

from src.ocr.extract_text import extract_document, extract_text_from_docx, page_needs_ocr
from src.ocr.ingest import ingest_directory


def test_page_needs_ocr_flags_short_text():
    assert page_needs_ocr("", min_chars=20) is True
    assert page_needs_ocr("a b", min_chars=20) is True


def test_page_needs_ocr_accepts_normal_text():
    assert page_needs_ocr("This Agreement is entered into as of January 5, 2019.", min_chars=20) is False


def test_extract_text_from_docx(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("This Master Services Agreement is entered into by Acme Corp.")
    document.add_paragraph("Governing Law: State of Delaware.")
    docx_path = tmp_path / "sample.docx"
    document.save(docx_path)

    text = extract_text_from_docx(docx_path)
    assert "Master Services Agreement" in text
    assert "State of Delaware" in text


def test_extract_document_rejects_unsupported_suffix(tmp_path):
    bogus = tmp_path / "contract.txt"
    bogus.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_document(bogus)


def test_ingest_directory_raises_on_missing_input_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_directory(input_dir=tmp_path / "does_not_exist", output_dir=tmp_path / "out")


def test_ingest_directory_writes_manifest_for_docx(tmp_path, monkeypatch):
    docx = pytest.importorskip("docx")
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    document = docx.Document()
    document.add_paragraph("Sample clause text for ingestion test.")
    document.save(input_dir / "sample.docx")

    manifest_path = tmp_path / "manifest.jsonl"
    monkeypatch.setattr("src.ocr.ingest.OCR_MANIFEST_JSONL", manifest_path)

    manifest = ingest_directory(input_dir=input_dir, output_dir=output_dir)

    assert len(manifest) == 1
    assert manifest[0]["status"] == "ok"
    assert (output_dir / "sample.txt").exists()
    assert manifest_path.exists()
