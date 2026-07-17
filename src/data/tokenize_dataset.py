"""Tokenize the flattened CUAD clauses into a transformer-ready dataset.

Produces sliding-window (question, context) features with token-level
start/end answer positions, in the same style as the standard HF SQuAD
preprocessing recipe. Unanswerable clauses (is_impossible) and windows that
don't contain the answer span get start/end positions pointing at the CLS
token, matching how RoBERTa-style extractive QA models are trained.

Saved via `datasets.Dataset.save_to_disk` so Week 2's fine-tuning script can
`datasets.load_from_disk(CUAD_TOKENIZED_DIR)` directly.
"""

import json
import logging

from datasets import Dataset
from transformers import AutoTokenizer

from src.common.config import (
    CUAD_CONTRACTS_JSONL,
    CUAD_FLATTENED_JSONL,
    CUAD_TOKENIZED_DIR,
    DOC_STRIDE,
    MAX_SEQ_LENGTH,
    TOKENIZER_MODEL_NAME,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_examples(contracts_path=CUAD_CONTRACTS_JSONL, clauses_path=CUAD_FLATTENED_JSONL):
    """Join clause records back onto their contract text (SQuAD-style examples)."""
    contracts = {c["contract_id"]: c["context"] for c in _load_jsonl(contracts_path)}
    clauses = _load_jsonl(clauses_path)

    examples = []
    for clause in clauses:
        context = contracts.get(clause["contract_id"])
        if context is None:
            logger.warning("No contract text found for %s, skipping", clause["contract_id"])
            continue
        examples.append(
            {
                "contract_id": clause["contract_id"],
                "category": clause["category"],
                "question": clause["question"],
                "context": context,
                "answers": {
                    "text": [a["text"] for a in clause["answers"]],
                    "answer_start": [a["start"] for a in clause["answers"]],
                },
            }
        )
    return examples


def make_tokenize_fn(tokenizer):
    def tokenize_fn(batch):
        tokenized = tokenizer(
            batch["question"],
            batch["context"],
            truncation="only_second",
            max_length=MAX_SEQ_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answers = batch["answers"][sample_idx]
            sequence_ids = tokenized.sequence_ids(i)

            if len(answers["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find the token span of the context (sequence_ids == 1).
            ctx_start = sequence_ids.index(1)
            ctx_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

            if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
                # Answer isn't inside this window -> unanswerable in this chunk.
                start_positions.append(0)
                end_positions.append(0)
                continue

            token_start = ctx_start
            while token_start <= ctx_end and offsets[token_start][0] <= start_char:
                token_start += 1
            token_start -= 1

            token_end = ctx_end
            while token_end >= ctx_start and offsets[token_end][1] >= end_char:
                token_end -= 1
            token_end += 1

            start_positions.append(token_start)
            end_positions.append(token_end)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        tokenized["contract_id"] = [batch["contract_id"][sample_map[i]] for i in range(len(offset_mapping))]
        tokenized["category"] = [batch["category"][sample_map[i]] for i in range(len(offset_mapping))]
        return tokenized

    return tokenize_fn


def tokenize_dataset() -> Dataset:
    if not CUAD_CONTRACTS_JSONL.exists() or not CUAD_FLATTENED_JSONL.exists():
        raise FileNotFoundError(
            "Preprocessed CUAD files not found. Run `python -m src.data.preprocess` first."
        )

    logger.info("Loading tokenizer: %s", TOKENIZER_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)

    examples = build_examples()
    logger.info("Building dataset from %d clause examples", len(examples))
    raw_dataset = Dataset.from_list(examples)

    tokenized_dataset = raw_dataset.map(
        make_tokenize_fn(tokenizer),
        batched=True,
        remove_columns=raw_dataset.column_names,
    )

    CUAD_TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(str(CUAD_TOKENIZED_DIR))
    logger.info("Saved %d tokenized features -> %s", len(tokenized_dataset), CUAD_TOKENIZED_DIR)
    return tokenized_dataset


if __name__ == "__main__":
    tokenize_dataset()
