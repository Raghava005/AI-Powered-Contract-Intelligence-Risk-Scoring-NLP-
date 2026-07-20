"""Shared paths and constants for the contract intelligence pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"

# --- OCR ingestion (Week 1, Day 3-5) ---
RAW_DOCS_DIR = RAW_DIR / "documents"  # source PDFs/DOCXs dropped in by users
OCR_TEXT_DIR = PROCESSED_DIR / "ocr_text"  # extracted .txt output, one per doc
OCR_MANIFEST_JSONL = PROCESSED_DIR / "ocr_manifest.jsonl"
# A page with fewer extractable characters than this is treated as scanned
# (no embedded text layer) and routed through Tesseract OCR instead.
OCR_MIN_CHARS_PER_PAGE = 20
OCR_DPI = 300

# --- Baseline NER (Week 1, Day 6-7) ---
# CUAD clause categories that map cleanly onto basic entity types, used to
# derive weakly-labeled training/eval data from CUAD's expert annotations.
CUAD_TO_NER_LABEL = {
    "Parties": "ORG",
    "Agreement Date": "DATE",
    "Effective Date": "DATE",
    "Expiration Date": "DATE",
}
SPACY_BASELINE_MODEL = "en_core_web_sm"
NER_TRAIN_DOCBIN = PROCESSED_DIR / "ner_train.spacy"
NER_LABELS_OF_INTEREST = ("ORG", "DATE", "MONEY")
NER_MODEL_DIR = PROJECT_ROOT / "models" / "ner_baseline"
NER_DEV_SPLIT = 0.1
# CUAD contracts are long, un-truncated documents, and spaCy's transition-based
# NER trainer scales with document length - a single epoch over the full
# training set takes ~8-9 min on a typical laptop CPU. 5 epochs is enough to
# validate the pipeline end-to-end for a Week 1 baseline; Week 2's transformer
# fine-tuning is where real convergence work happens.
NER_N_ITER = 5

# CUAD is distributed as a zip (contains CUADv1.json at its root, plus the
# source contract PDFs/TXTs) rather than as a standalone raw JSON file.
CUAD_ZIP_URL = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data.zip"
CUAD_ZIP_INNER_JSON_NAME = "CUADv1.json"
CUAD_RAW_ZIP = RAW_DIR / "cuad_data.zip"
CUAD_RAW_JSON = RAW_DIR / "CUAD_v1.json"
CUAD_CONTRACTS_JSONL = PROCESSED_DIR / "cuad_contracts.jsonl"
CUAD_FLATTENED_JSONL = PROCESSED_DIR / "cuad_clauses.jsonl"
CUAD_TOKENIZED_DIR = PROCESSED_DIR / "cuad_tokenized"

# Base transformer checkpoint used for tokenization (Week 1) and fine-tuning (Week 2).
TOKENIZER_MODEL_NAME = "roberta-base"
MAX_SEQ_LENGTH = 512
DOC_STRIDE = 128

# The 41 clause categories annotated in CUAD v1.
CUAD_CATEGORIES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "Competitive Restriction Exception",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
]

assert len(CUAD_CATEGORIES) == 41
