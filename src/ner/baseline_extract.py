"""Extract basic entities (orgs, dates, money) from contract text.

Two sources are combined:
- A trained baseline model (train_baseline_ner.py) for ORG/DATE, learned
  from CUAD's Parties/Date clause annotations.
- A regex-based MONEY matcher, since CUAD doesn't annotate clean monetary
  spans directly (dollar amounts live inside longer clauses like "Cap On
  Liability"). Rule-based extraction is a reasonable, explainable baseline
  for a well-structured pattern like currency amounts.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import spacy

from src.common.config import NER_MODEL_DIR, SPACY_BASELINE_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MONEY_PATTERN = re.compile(
    r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|thousand))?"
    r"|\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:USD|dollars)\b",
    re.IGNORECASE,
)


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int


def extract_money(text: str) -> list[Entity]:
    return [Entity(m.group(), "MONEY", m.start(), m.end()) for m in MONEY_PATTERN.finditer(text)]


def load_ner_model(model_dir: Path = NER_MODEL_DIR):
    """Prefer our CUAD-trained baseline; fall back to spaCy's pretrained pipeline."""
    if model_dir.exists():
        logger.info("Loading trained baseline NER model from %s", model_dir)
        return spacy.load(model_dir)

    logger.warning(
        "%s not found (run `python -m src.ner.train_baseline_ner` first). "
        "Falling back to pretrained %s.",
        model_dir,
        SPACY_BASELINE_MODEL,
    )
    return spacy.load(SPACY_BASELINE_MODEL)


def extract_entities(text: str, nlp=None) -> list[Entity]:
    nlp = nlp or load_ner_model()
    doc = nlp(text)

    entities = [
        Entity(ent.text, ent.label_, ent.start_char, ent.end_char)
        for ent in doc.ents
        if ent.label_ in ("ORG", "DATE")
    ]
    entities.extend(extract_money(text))
    return sorted(entities, key=lambda e: e.start)


if __name__ == "__main__":
    import sys

    sample_text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (
            "This Agreement is entered into as of January 5, 2019, by and between "
            "Acme Corp and Globex LLC. Total fees shall not exceed $2,500,000."
        )
    )
    for entity in extract_entities(sample_text):
        print(f"{entity.label:6s} [{entity.start}:{entity.end}] {entity.text!r}")
