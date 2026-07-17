# AI-Powered Contract Intelligence & Risk Scoring (NLP)

An NLP platform for legal/compliance teams that ingests contracts (PDF/Word), extracts key
entities (dates, parties, jurisdictions), identifies clauses (termination, confidentiality,
auto-renewal, etc.), and flags anomalous or high-risk language.

Built on the [CUAD (Contract Understanding Atticus Dataset)](https://www.atticusprojectai.org/cuad),
500+ commercial contracts annotated across 41 legal clause categories.

## Tech stack

- **NLP/ML**: Hugging Face Transformers (BERT/RoBERTa), spaCy, PyTorch
- **Retrieval**: Pinecone/Milvus, LangChain
- **Backend**: FastAPI, Uvicorn, Celery
- **Deployment**: Docker, AWS EC2

## Project structure

```
src/
  common/       shared config, constants (CUAD category list, paths)
  data/         CUAD download, preprocessing, tokenization  (Week 1, Day 1-2)
  ocr/          PDF/DOCX ingestion -> raw text                (Week 1, Day 3-5)
  ner/          baseline spaCy NER training/eval              (Week 1, Day 6-7)
scripts/        environment setup scripts
tests/          pytest unit tests (run against small fixtures, no network needed)
data/
  raw/          downloaded CUAD_v1.json + source PDFs (gitignored)
  processed/    flattened/tokenized training data (gitignored)
  sample/       tiny committed fixtures used by tests
```

## Setup

```powershell
# from the project root
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Tesseract OCR binary (needed for Day 3-5) is a separate system install, not a pip package:
https://github.com/UB-Mannheim/tesseract/wiki (Windows installer). After installing, either
add it to PATH or set `TESSERACT_CMD` in a `.env` file to the `tesseract.exe` path.

## Week 1 progress

- [x] **Day 1-2** — Environment setup; download CUAD and preprocess into flattened
      JSONL records + tokenized (RoBERTa) training format.
      See `src/data/download_cuad.py`, `src/data/preprocess.py`, `src/data/tokenize_dataset.py`.
- [ ] **Day 3-5** — OCR ingestion pipeline for raw PDFs (digital-text extraction with OCR
      fallback via Tesseract) plus DOCX support.
      See `src/ocr/`.
- [ ] **Day 6-7** — Baseline spaCy NER model to extract organizations, dates, and monetary
      values from contract text.
      See `src/ner/`.

## Running Day 1-2

```powershell
# 1. Download CUAD_v1.json (SQuAD-style QA export) into data/raw/
python -m src.data.download_cuad

# 2. Flatten into per-clause JSONL records
python -m src.data.preprocess

# 3. Tokenize for transformer fine-tuning (Week 2)
python -m src.data.tokenize_dataset

# Run tests (uses data/sample/, no network required)
pytest tests/ -v
```
