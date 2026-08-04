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
- [x] **Day 3-5** — OCR ingestion pipeline for raw PDFs (digital-text extraction with OCR
      fallback via Tesseract) plus DOCX support.
      See `src/ocr/`.
- [x] **Day 6-7** — Baseline spaCy NER model to extract organizations, dates, and monetary
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

## Running Day 3-5

```powershell
# Drop source PDFs/DOCXs into data/raw/documents/, then:
python -m src.ocr.ingest

# Extracted text lands in data/processed/ocr_text/, with a per-document
# manifest (status, char count, digital vs. OCR method) at
# data/processed/ocr_manifest.jsonl
```

## Running Day 6-7

```powershell
# 1. Build weakly-labeled training data from CUAD's Parties/Date clause
#    annotations (Parties -> ORG, Agreement/Effective/Expiration Date -> DATE)
python -m src.ner.build_training_data

# 2. Train a baseline NER model (blank English pipeline + "ner" pipe,
#    5 epochs - CUAD contracts are long, un-truncated documents, so a single
#    epoch takes several minutes on a laptop CPU)
python -m src.ner.train_baseline_ner

# 3. Evaluate against the training data (no held-out split yet - this is a
#    train-fit sanity check, not a generalization measure; on the full CUAD
#    corpus this baseline scores P=0.48 R=0.40 F1=0.44 overall, ORG F1=0.43,
#    DATE F1=0.46). A real held-out dev split is Week 2 scope alongside the
#    transformer fine-tuning.
python -m src.ner.evaluate

# 4. Extract entities (ORG/DATE from the trained model, MONEY via regex)
python -m src.ner.baseline_extract "Total fees shall not exceed $2,500,000."
```
# 🚀 Week 2 Deliverables

✔ Integrated the Official CUAD Dataset

✔ Implemented text preprocessing pipeline

✔ Cleaned and normalized contract clauses

✔ Encoded legal clause labels

✔ Generated dataset statistics

✔ Created Train / Validation / Test split

✔ Fine-tuned a Transformer model

✔ Evaluated model performance

✔ Built an interactive prediction system

---

# ✨ Features

- 📄 Official CUAD Dataset Support
- 🧹 Automated Text Preprocessing
- 🏷 Label Encoding
- 📊 Dataset Statistics
- 🤖 Transformer-based Legal Clause Classification
- 📈 Accuracy, Precision, Recall & F1 Evaluation
- 💾 Model Saving & Loading
- 🔍 Interactive Clause Prediction
- 🔁 Reproducible Training Pipeline

---

# 🛠 Tech Stack

## Programming Language

- Python

## Machine Learning

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

## NLP

- RoBERTa / DistilRoBERTa
- Tokenization
- Text Normalization

## Evaluation

- Scikit-learn

## Dataset

- Official CUAD Dataset

---

# 📂 Project Structure

```text
AI-Powered-Contract-Intelligence-Risk-Scoring-NLP/

│
├── data/
│   ├── raw/
│   │   └── CUAD_v1.json
│   ├── processed/
│
├── models/
│   └── legal_clause_classifier/
│
├── results/
│   └── evaluation_results.json
│
├── src/
│   ├── classification/
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   ├── ner/
│   ├── ocr/
│   └── common/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 📚 Dataset

This project uses the **Contract Understanding Atticus Dataset (CUAD)**.

### Dataset Highlights

- 500+ Commercial Contracts
- 41 Legal Clause Categories
- Real-world Legal Documents
- Expert Legal Annotations

---

# 🔄 Data Preprocessing Pipeline

```
Official CUAD Dataset
          │
          ▼
Load JSON
          │
          ▼
Remove Impossible Clauses
          │
          ▼
Text Cleaning
          │
          ▼
Whitespace Normalization
          │
          ▼
Label Extraction
          │
          ▼
Label Encoding
          │
          ▼
Dataset Statistics
```

---

# 🤖 Model Training Pipeline

```
Official CUAD Dataset
          │
          ▼
Preprocessing
          │
          ▼
Tokenization
          │
          ▼
Train / Validation / Test Split
          │
          ▼
Transformer Training
          │
          ▼
Model Evaluation
          │
          ▼
Saved Model
```

---

# ⚙ Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | DistilRoBERTa / RoBERTa |
| Epochs | 3 |
| Batch Size | 4 |
| Learning Rate | 2e-5 |
| Max Sequence Length | 256 |
| Dataset | Official CUAD |

---

# 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

### Sample Evaluation

| Metric | Score |
|---------|------:|
| Accuracy | 52% |
| Precision | 44.9% |
| Recall | 52% |
| F1 Score | 44.5% |

> *Results shown are from the current Week 2 training configuration using a subset of the official CUAD dataset.*

---

# ▶ Running the Project

## Train

```bash
python src/classification/train.py
```

## Evaluate

```bash
python src/classification/evaluate.py
```

## Predict

```bash
python src/classification/predict.py
```

---

# 💡 Sample Prediction

### Input

```
Either party may terminate this agreement by giving thirty days written notice.
```

### Output

```
Predicted Clause:
Termination For Convenience

Confidence:
84.7%
```

*Sample output for illustration.*

---

# 🔮 Future Improvements

- Train on the complete CUAD dataset using GPU resources.
- Hyperparameter optimization.
- Multi-label legal clause classification.
- Named Entity Recognition (NER).
- FastAPI REST API.
- Semantic contract search using vector databases.
- Web dashboard for contract analysis.

## Week 3 - Day 1

- Integrated Sentence Transformers for semantic embeddings.
- Generated dense vector embeddings for legal contract clauses.
- Stored embeddings and metadata for semantic retrieval.

## Week 3 - Day 2

- Built a FAISS vector database from generated sentence embeddings.
- Indexed all legal clause vectors for efficient similarity search.
- Saved the searchable FAISS index for semantic retrieval.

## Week 3 - Day 3

- Implemented semantic search using FAISS.
- Added natural language query support.
- Retrieved top matching legal clauses with similarity scores.

## Week 3 - Day 4

- Initialized FastAPI backend.
- Added REST API structure.
- Implemented health and version endpoints.
- Enabled automatic Swagger documentation.

## Week 3 - Day 5

- Implemented document upload API using FastAPI.
- Added validation for PDF and DOCX files.
- Saved uploaded documents for downstream OCR and NLP processing.

Week 3 - Day 6: Contract Intelligence APIs
Objective

## Implemented REST APIs for contract intelligence using FastAPI.

Features
Upload contract files
Predict legal clause category
Perform semantic search using FAISS
Health monitoring endpoint
Version endpoint
Implemented APIs
POST /upload

Uploads PDF and DOCX contract files.

POST /predict

Predicts the legal clause category using the trained RoBERTa classifier.

POST /search

Performs semantic search using Sentence Transformers and the FAISS vector database.

GET /health

Checks whether the API service is running.

GET /version

Returns the current API version.

Technologies Used
FastAPI
Hugging Face Transformers
Sentence Transformers
FAISS
NumPy
Python
Result

Successfully developed and tested REST APIs for contract upload, legal clause prediction, and semantic search.

## Week 3 - Day 7: API Documentation & Project Information
Objective

Enhanced the API by adding project information and interactive API documentation.

Features Added
Project information endpoint
Interactive Swagger UI documentation
API version information
Health monitoring
Upload API
Prediction API
Semantic Search API
New Endpoint
GET /info

Returns project metadata.

Example Response

{
    "project": "AI-Powered Contract Intelligence",
    "version": "1.1.0",
    "embedding_model": "all-MiniLM-L6-v2",
    "classifier": "RoBERTa",
    "vector_database": "FAISS",
    "status": "Production Ready"
}
API Documentation

Swagger UI

http://127.0.0.1:8000/docs

OpenAPI Specification

http://127.0.0.1:8000/openapi.json
Available APIs
GET /
GET /health
GET /version
GET /info
POST /upload
POST /predict
POST /search
Technologies Used
FastAPI
Swagger UI (OpenAPI)
FAISS
Hugging Face Transformers
Sentence Transformers
Python