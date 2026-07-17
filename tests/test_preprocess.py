import json

import pytest

from src.common.config import PROJECT_ROOT
from src.data.preprocess import _extract_category, iter_contracts, preprocess

FIXTURE_PATH = PROJECT_ROOT / "data" / "sample" / "mini_cuad.json"


@pytest.fixture
def raw_fixture():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_extract_category_reads_quoted_span():
    question = 'Highlight the parts (if any) of this contract related to "Governing Law" ...'
    assert _extract_category(question) == "Governing Law"


def test_extract_category_raises_without_quotes():
    with pytest.raises(ValueError):
        _extract_category("no quoted category here")


def test_iter_contracts_yields_one_contract_row(raw_fixture):
    rows = list(iter_contracts(raw_fixture))
    contract_rows = [r for r in rows if r.get("_type") != "clause"]
    assert len(contract_rows) == 1
    assert contract_rows[0]["contract_id"] == "ACME-SUPPLY-AGREEMENT"
    assert "Acme Corp" in contract_rows[0]["context"]


def test_iter_contracts_yields_one_clause_row_per_qa(raw_fixture):
    rows = list(iter_contracts(raw_fixture))
    clause_rows = [r for r in rows if r.get("_type") == "clause"]
    assert len(clause_rows) == 4  # matches the 4 qas in the fixture

    categories = {r["category"] for r in clause_rows}
    assert categories == {
        "Parties",
        "Governing Law",
        "Termination For Convenience",
        "Most Favored Nation",
    }


def test_iter_contracts_answer_offsets_are_correct(raw_fixture):
    rows = list(iter_contracts(raw_fixture))
    context = next(r["context"] for r in rows if r.get("_type") != "clause")
    parties = next(r for r in rows if r.get("category") == "Parties")

    for answer in parties["answers"]:
        assert context[answer["start"] : answer["end"]] == answer["text"]


def test_iter_contracts_marks_unanswered_clause_impossible(raw_fixture):
    rows = list(iter_contracts(raw_fixture))
    mfn = next(r for r in rows if r.get("category") == "Most Favored Nation")
    assert mfn["is_impossible"] is True
    assert mfn["answers"] == []


def test_preprocess_writes_expected_jsonl_files(tmp_path):
    contracts_out = tmp_path / "contracts.jsonl"
    clauses_out = tmp_path / "clauses.jsonl"

    n_contracts, n_clauses = preprocess(
        raw_json_path=FIXTURE_PATH,
        contracts_out=contracts_out,
        clauses_out=clauses_out,
    )

    assert n_contracts == 1
    assert n_clauses == 4
    assert contracts_out.exists()
    assert clauses_out.exists()

    with open(contracts_out, encoding="utf-8") as f:
        contract_lines = [json.loads(line) for line in f]
    with open(clauses_out, encoding="utf-8") as f:
        clause_lines = [json.loads(line) for line in f]

    assert len(contract_lines) == 1
    assert len(clause_lines) == 4
    assert all(c["contract_id"] == "ACME-SUPPLY-AGREEMENT" for c in clause_lines)


def test_preprocess_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess(raw_json_path=tmp_path / "does_not_exist.json")
