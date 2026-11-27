import importlib.util
from pathlib import Path

import pytest
import yaml

from regression_gate.dataset import DatasetError, find_leaks, load_dataset
from regression_gate.prompts import load_prompt

ROOT = Path(__file__).resolve().parent.parent


def case(**overrides):
    base = {
        "id": "case-001",
        "email": "I was charged twice this month.",
        "expected_category": "billing",
        "key_facts": ["charged twice"],
        "difficulty": "easy",
        "tags": [],
        "reviewed": True,
    }
    return {**base, **overrides}


def write_dataset(tmp_path, cases, **top):
    data = {"version": 1, "description": "Test set.", "cases": cases, **top}
    path = tmp_path / "golden.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_valid_dataset_loads_with_content_hash(tmp_path):
    ds = load_dataset(write_dataset(tmp_path, [case()]))
    assert ds.cases[0].expected_category == "billing"
    assert len(ds.content_hash) == 12


def test_content_hash_changes_when_file_changes(tmp_path):
    first = load_dataset(write_dataset(tmp_path, [case()])).content_hash
    second = load_dataset(write_dataset(tmp_path, [case(key_facts=["double charge"])])).content_hash
    assert first != second


@pytest.mark.parametrize("overrides", [{"reviewed": False}, {"notes": "Added later."}, {"difficulty": "medium"}])
def test_content_hash_ignores_fields_that_dont_affect_scores(tmp_path, overrides):
    first = load_dataset(write_dataset(tmp_path, [case()])).content_hash
    second = load_dataset(write_dataset(tmp_path, [case(**overrides)])).content_hash
    assert first == second


def test_content_hash_cannot_be_written_by_hand(tmp_path):
    with pytest.raises(DatasetError, match="computed"):
        load_dataset(write_dataset(tmp_path, [case()], content_hash="abc"))


def test_label_casing_is_strict_for_human_data(tmp_path):
    with pytest.raises(DatasetError, match="expected_category"):
        load_dataset(write_dataset(tmp_path, [case(expected_category="Billing")]))


@pytest.mark.parametrize("bad_id", ["billing-001", "case-1", "case_001"])
def test_id_format_is_enforced(tmp_path, bad_id):
    with pytest.raises(DatasetError, match="id"):
        load_dataset(write_dataset(tmp_path, [case(id=bad_id)]))


def test_duplicate_ids_are_rejected(tmp_path):
    cases = [case(), case(email="Different email.")]
    with pytest.raises(DatasetError, match="duplicate id"):
        load_dataset(write_dataset(tmp_path, cases))


def test_duplicate_emails_are_rejected_even_if_reformatted(tmp_path):
    cases = [case(), case(id="case-002", email="  I was CHARGED twice\nthis month. ")]
    with pytest.raises(DatasetError, match="same email"):
        load_dataset(write_dataset(tmp_path, cases))


@pytest.mark.parametrize("facts", [[], ["a", "b", "c", "d"], ["  "]])
def test_key_facts_must_be_one_to_three_non_blank(tmp_path, facts):
    with pytest.raises(DatasetError, match="key_facts"):
        load_dataset(write_dataset(tmp_path, [case(key_facts=facts)]))


def test_unknown_tag_is_rejected(tmp_path):
    with pytest.raises(DatasetError, match="tags"):
        load_dataset(write_dataset(tmp_path, [case(tags=["sarcastic"], notes="x")]))


@pytest.mark.parametrize("overrides", [{"difficulty": "hard"}, {"tags": ["typos"]}])
def test_tricky_cases_need_notes(tmp_path, overrides):
    with pytest.raises(DatasetError, match="notes"):
        load_dataset(write_dataset(tmp_path, [case(**overrides)]))


def test_unknown_field_is_rejected(tmp_path):
    with pytest.raises(DatasetError, match="expected_difficulty"):
        load_dataset(write_dataset(tmp_path, [case(expected_difficulty="easy")]))


def test_find_leaks_matches_reformatted_copies(tmp_path):
    ds = load_dataset(write_dataset(tmp_path, [case()]))
    leaks = find_leaks(ds, {"i was charged   twice this month.": "prompts/v2.yaml few-shot"})
    assert leaks == ["case-001 duplicates an email from prompts/v2.yaml few-shot"]


# --- Checks on the real files: these run in CI on every change ---


def _smoke_emails() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("smoke_test", ROOT / "scripts" / "smoke_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {email: "scripts/smoke_test.py (LLM-written)" for _, email in module.SMOKE_EMAILS}


def _few_shot_emails() -> dict[str, str]:
    return {
        example.email: f"{path.relative_to(ROOT)} few-shot"
        for path in sorted((ROOT / "prompts").glob("v*.yaml"))
        for example in load_prompt(path).few_shot
    }


def test_real_golden_dataset_is_valid():
    load_dataset(ROOT / "data" / "golden.yaml")


def test_real_golden_dataset_has_no_leaked_emails():
    dataset = load_dataset(ROOT / "data" / "golden.yaml")
    assert find_leaks(dataset, {**_smoke_emails(), **_few_shot_emails()}) == []
