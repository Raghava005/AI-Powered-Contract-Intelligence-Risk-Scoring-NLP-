"""CLI: walk a folder of contract PDFs/DOCXs, extract text, write outputs.

Usage:
    python -m src.ocr.ingest --input-dir data/raw/documents --output-dir data/processed/ocr_text
"""

import argparse
import json
import logging
from pathlib import Path

from src.common.config import OCR_MANIFEST_JSONL, OCR_TEXT_DIR, RAW_DOCS_DIR
from src.ocr.extract_text import OcrDependencyError, extract_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def ingest_directory(input_dir: Path = RAW_DOCS_DIR, output_dir: Path = OCR_TEXT_DIR) -> list[dict]:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"{input_dir} does not exist. Drop source PDF/DOCX contracts there first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    doc_paths = sorted(
        p for p in input_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not doc_paths:
        logger.warning("No PDF/DOCX files found under %s", input_dir)

    for doc_path in doc_paths:
        record = {"source_path": str(doc_path), "status": "ok", "char_count": 0, "error": None}
        try:
            text = extract_document(doc_path)
            out_path = output_dir / f"{doc_path.stem}.txt"
            out_path.write_text(text, encoding="utf-8")
            record["output_path"] = str(out_path)
            record["char_count"] = len(text)
            logger.info("Extracted %d chars from %s -> %s", len(text), doc_path.name, out_path.name)
        except OcrDependencyError as exc:
            record["status"] = "missing_dependency"
            record["error"] = str(exc)
            logger.error("Skipping %s: %s", doc_path.name, exc)
        except Exception as exc:  # keep ingesting the rest of the batch on a single bad file
            record["status"] = "error"
            record["error"] = str(exc)
            logger.exception("Failed to extract %s", doc_path.name)

        manifest.append(record)

    with open(OCR_MANIFEST_JSONL, "w", encoding="utf-8") as f:
        for record in manifest:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    ok = sum(1 for r in manifest if r["status"] == "ok")
    logger.info("Ingested %d/%d documents successfully. Manifest -> %s", ok, len(manifest), OCR_MANIFEST_JSONL)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Extract text from contract PDFs/DOCXs.")
    parser.add_argument("--input-dir", type=Path, default=RAW_DOCS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OCR_TEXT_DIR)
    args = parser.parse_args()
    ingest_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
