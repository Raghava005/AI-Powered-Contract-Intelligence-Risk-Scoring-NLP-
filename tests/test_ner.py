import json

import pytest

spacy = pytest.importorskip("spacy")

from src.common.config import PROJECT_ROOT
from src.data.preprocess import preprocess
from src.ner.baseline_extract import extract_money
from src.ner.build_training_data import build_training_data

FIXTURE_PATH = PROJECT_ROOT / "data" / "sample" / "mini_cuad.json"


@pytest.fixture
def preprocessed(tmp_path):
    contracts_out = tmp_path / "contracts.jsonl"
    clauses_out = tmp_path / "clauses.jsonl"
    preprocess(raw_json_path=FIXTURE_PATH, contracts_out=contracts_out, clauses_out=clauses_out)
    return contracts_out, clauses_out


def test_extract_money_matches_dollar_amounts():
    text = "Total fees shall not exceed $2,500,000 per year, plus a $10 admin fee."
    matches = [e.text for e in extract_money(text)]
    assert "$2,500,000" in matches
    assert "$10" in matches


def test_extract_money_ignores_plain_numbers():
    text = "Section 4.2 references clause 90 of the agreement."
    assert extract_money(text) == []


def test_build_training_data_creates_docbin_with_expected_labels(preprocessed, tmp_path):
    contracts_out, clauses_out = preprocessed
    docbin_out = tmp_path / "ner_train.spacy"

    n_docs, label_counts = build_training_data(
        contracts_path=contracts_out,
        clauses_path=clauses_out,
        out_path=docbin_out,
    )

    assert n_docs == 1
    assert docbin_out.exists()
    # Fixture has 2 Parties spans + 1 Governing Law (not mapped) + 1 Termination
    # (not mapped) -> only the 2 "Parties" -> ORG spans should be labeled.
    assert label_counts["ORG"] == 2
    assert "DATE" not in label_counts  # fixture has no Agreement/Effective/Expiration Date qas

    from spacy.tokens import DocBin

    nlp = spacy.blank("en")
    docs = list(DocBin().from_disk(docbin_out).get_docs(nlp.vocab))
    assert len(docs) == 1
    ent_texts = {ent.text for ent in docs[0].ents}
    assert ent_texts == {"Acme Corp", "Globex LLC"}


def test_build_training_data_raises_on_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_training_data(
            contracts_path=tmp_path / "missing_contracts.jsonl",
            clauses_path=tmp_path / "missing_clauses.jsonl",
            out_path=tmp_path / "out.spacy",
        )
