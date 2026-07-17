"""Flatten CUAD_v1.json into two training-ready JSONL files.

CUAD's raw format nests 41 clause-category questions inside a single
"paragraph" per contract. Keeping the full contract text duplicated 41x
(once per question) wastes disk and I/O, so this splits the data into:

- cuad_contracts.jsonl : one row per contract {contract_id, title, context}
- cuad_clauses.jsonl   : one row per (contract, category) pair
                         {contract_id, category, question, is_impossible, answers}

`answers` is a list of {text, start, end} character-offset spans into the
matching contract's `context`.
"""

import json
import logging
import re
from typing import Iterator

from src.common.config import (
    CUAD_CATEGORIES,
    CUAD_CONTRACTS_JSONL,
    CUAD_FLATTENED_JSONL,
    CUAD_RAW_JSON,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# CUAD questions are phrased like:
#   'Highlight the parts (if any) of this contract related to "Parties" that
#    should be reviewed by a lawyer. Details: ...'
# Pull the category name out of the quoted span so we don't have to trust
# question ordering to line up with CUAD_CATEGORIES.
_CATEGORY_PATTERN = re.compile(r'"([^"]+)"')

_CATEGORY_SET = set(CUAD_CATEGORIES)


def _extract_category(question: str) -> str:
    match = _CATEGORY_PATTERN.search(question)
    if not match:
        raise ValueError(f"Could not find a quoted category name in question: {question!r}")
    category = match.group(1)
    if category not in _CATEGORY_SET:
        logger.warning("Category %r not in the known CUAD_CATEGORIES list", category)
    return category


def iter_contracts(raw_json: dict) -> Iterator[dict]:
    for entry in raw_json["data"]:
        title = entry["title"]
        # CUAD stores exactly one paragraph per contract.
        paragraph = entry["paragraphs"][0]
        context = paragraph["context"]
        contract_id = title

        yield {
            "contract_id": contract_id,
            "title": title,
            "context": context,
        }

        for qa in paragraph["qas"]:
            answers = [
                {
                    "text": ans["text"],
                    "start": ans["answer_start"],
                    "end": ans["answer_start"] + len(ans["text"]),
                }
                for ans in qa.get("answers", [])
            ]
            yield {
                "contract_id": contract_id,
                "category": _extract_category(qa["question"]),
                "question": qa["question"],
                "is_impossible": qa.get("is_impossible", len(answers) == 0),
                "answers": answers,
                "_type": "clause",
            }


def preprocess(
    raw_json_path=CUAD_RAW_JSON,
    contracts_out=CUAD_CONTRACTS_JSONL,
    clauses_out=CUAD_FLATTENED_JSONL,
) -> tuple[int, int]:
    if not raw_json_path.exists():
        raise FileNotFoundError(
            f"{raw_json_path} not found. Run `python -m src.data.download_cuad` first "
            "(or place a manually-downloaded CUAD_v1.json there)."
        )

    contracts_out.parent.mkdir(parents=True, exist_ok=True)
    clauses_out.parent.mkdir(parents=True, exist_ok=True)

    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    n_contracts = 0
    n_clauses = 0

    with open(contracts_out, "w", encoding="utf-8") as contracts_f, open(
        clauses_out, "w", encoding="utf-8"
    ) as clauses_f:
        for record in iter_contracts(raw):
            if record.get("_type") == "clause":
                record = {k: v for k, v in record.items() if k != "_type"}
                clauses_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_clauses += 1
            else:
                contracts_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_contracts += 1

    logger.info(
        "Wrote %d contracts -> %s, %d clause records -> %s",
        n_contracts,
        contracts_out,
        n_clauses,
        clauses_out,
    )
    return n_contracts, n_clauses


if __name__ == "__main__":
    preprocess()
