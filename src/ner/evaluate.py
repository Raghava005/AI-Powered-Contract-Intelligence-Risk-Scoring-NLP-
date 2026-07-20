"""Evaluate a spaCy NER model's precision/recall/F1 against gold examples."""

import json
import logging

import spacy
from spacy.scorer import Scorer
from spacy.tokens import DocBin
from spacy.training import Example

from src.common.config import NER_MODEL_DIR, NER_TRAIN_DOCBIN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(nlp, examples: list[Example]) -> dict:
    """Wraps spacy.scorer.Scorer; returns overall + per-label P/R/F1."""
    scorer = Scorer(nlp)
    scores = scorer.score(examples)
    return {
        "precision": scores.get("ents_p", 0.0),
        "recall": scores.get("ents_r", 0.0),
        "f1": scores.get("ents_f", 0.0),
        "per_type": scores.get("ents_per_type", {}),
    }


def evaluate_from_disk(model_dir=NER_MODEL_DIR, docbin_path=NER_TRAIN_DOCBIN) -> dict:
    if not model_dir.exists():
        raise FileNotFoundError(
            f"{model_dir} not found. Run `python -m src.ner.train_baseline_ner` first."
        )
    if not docbin_path.exists():
        raise FileNotFoundError(f"{docbin_path} not found.")

    nlp = spacy.load(model_dir)
    doc_bin = DocBin().from_disk(docbin_path)
    gold_docs = list(doc_bin.get_docs(nlp.vocab))
    examples = [Example(nlp(doc.text), doc) for doc in gold_docs]

    metrics = evaluate_model(nlp, examples)
    logger.info("Baseline NER eval: P=%.3f R=%.3f F1=%.3f", metrics["precision"], metrics["recall"], metrics["f1"])
    logger.info("Per-type: %s", json.dumps(metrics["per_type"], indent=2))
    return metrics


if __name__ == "__main__":
    evaluate_from_disk()
