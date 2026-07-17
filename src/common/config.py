"""Shared paths and constants for the contract intelligence pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"

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
