"""Train a baseline spaCy NER model on the weakly-labeled CUAD data.

This is intentionally a from-scratch statistical NER model (blank English
pipeline + an "ner" pipe), trained directly via spaCy's Python training
API rather than the `spacy train` CLI, so the whole Day 6-7 pipeline stays
self-contained in this repo. It's a baseline: Week 2 fine-tunes a
transformer for the full 41-category clause classification task.
"""

import logging
import random

import spacy
from spacy.tokens import DocBin
from spacy.training import Example

from src.common.config import NER_DEV_SPLIT, NER_MODEL_DIR, NER_N_ITER, NER_TRAIN_DOCBIN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_examples(nlp, docbin_path):
    doc_bin = DocBin().from_disk(docbin_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    examples = [Example(nlp.make_doc(doc.text), doc) for doc in docs]
    return examples


def train(
    docbin_path=NER_TRAIN_DOCBIN,
    out_dir=NER_MODEL_DIR,
    n_iter: int = NER_N_ITER,
    dev_split: float = NER_DEV_SPLIT,
    seed: int = 13,
):
    if not docbin_path.exists():
        raise FileNotFoundError(
            f"{docbin_path} not found. Run `python -m src.ner.build_training_data` first."
        )

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    examples = _load_examples(nlp, docbin_path)
    if not examples:
        raise ValueError(f"No training examples found in {docbin_path}")

    for example in examples:
        for ent in example.reference.ents:
            ner.add_label(ent.label_)

    rng = random.Random(seed)
    rng.shuffle(examples)
    n_dev = max(1, int(len(examples) * dev_split)) if len(examples) > 1 else 0
    dev_examples = examples[:n_dev]
    train_examples = examples[n_dev:] or examples  # fall back to reusing data on tiny sets

    logger.info(
        "Training baseline NER on %d examples (%d dev) with labels=%s",
        len(train_examples),
        len(dev_examples),
        ner.labels,
    )

    optimizer = nlp.initialize(lambda: train_examples)
    for epoch in range(n_iter):
        rng.shuffle(train_examples)
        losses = {}
        batches = spacy.util.minibatch(train_examples, size=8)
        for batch in batches:
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
        logger.info("epoch %d/%d - losses=%s", epoch + 1, n_iter, losses)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(out_dir)
    logger.info("Saved trained baseline NER model -> %s", out_dir)

    return nlp, dev_examples


if __name__ == "__main__":
    train()
