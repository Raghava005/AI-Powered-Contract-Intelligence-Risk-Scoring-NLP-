"""Build a spaCy training set from CUAD's expert clause annotations.

CUAD doesn't ship generic NER labels, but several of its 41 clause
categories are, in effect, single-entity spans: "Parties" spans are
organization names, and "Agreement/Effective/Expiration Date" spans are
dates. CUAD_TO_NER_LABEL (src/common/config.py) maps those categories onto
ORG/DATE. This script re-uses that weak supervision to build a labeled
spaCy DocBin, which `train_baseline_ner.py` fine-tunes a baseline model on.

Run after src/data/preprocess.py has produced cuad_contracts.jsonl and
cuad_clauses.jsonl.
"""

import json
import logging
from collections import Counter, defaultdict

import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

from src.common.config import (
    CUAD_CONTRACTS_JSONL,
    CUAD_FLATTENED_JSONL,
    CUAD_TO_NER_LABEL,
    NER_TRAIN_DOCBIN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_training_data(
    contracts_path=CUAD_CONTRACTS_JSONL,
    clauses_path=CUAD_FLATTENED_JSONL,
    out_path=NER_TRAIN_DOCBIN,
    label_map=CUAD_TO_NER_LABEL,
) -> tuple[int, Counter]:
    if not contracts_path.exists() or not clauses_path.exists():
        raise FileNotFoundError(
            "Preprocessed CUAD files not found. Run `python -m src.data.preprocess` first."
        )

    contracts = {c["contract_id"]: c["context"] for c in _load_jsonl(contracts_path)}
    clauses = _load_jsonl(clauses_path)

    spans_by_contract = defaultdict(list)
    for clause in clauses:
        label = label_map.get(clause["category"])
        if label is None:
            continue
        for answer in clause["answers"]:
            spans_by_contract[clause["contract_id"]].append(
                (answer["start"], answer["end"], label)
            )

    nlp = spacy.blank("en")
    doc_bin = DocBin()
    label_counts: Counter = Counter()
    n_docs = 0

    for contract_id, spans in spans_by_contract.items():
        context = contracts.get(contract_id)
        if context is None:
            logger.warning("No contract text for %s, skipping", contract_id)
            continue

        doc = nlp.make_doc(context)
        candidate_spans = []
        for start, end, label in spans:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                logger.warning(
                    "Could not align span (%d, %d) in %s to token boundaries, skipping",
                    start,
                    end,
                    contract_id,
                )
                continue
            candidate_spans.append(span)

        entities = filter_spans(candidate_spans)
        if not entities:
            continue

        doc.ents = entities
        doc_bin.add(doc)
        n_docs += 1
        label_counts.update(e.label_ for e in entities)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc_bin.to_disk(out_path)
    logger.info("Wrote %d labeled docs (%s) -> %s", n_docs, dict(label_counts), out_path)
    return n_docs, label_counts


if __name__ == "__main__":
    build_training_data()
