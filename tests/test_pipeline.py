"""Tests for the judge, runner, metrics, storage and settings, with a fake OpenAI client."""

import asyncio
import re
from pathlib import Path

import pytest

from helpers import SETTINGS, RoutingClient, case_result, fake_response, judge_json, run_record
from regression_gate import llm
from regression_gate.classifier import ErrorType
from regression_gate.dataset import load_dataset
from regression_gate.judge import judge_summary
from regression_gate.prompts import load_prompt, prompt_path
from regression_gate.runner import run_eval
from regression_gate.settings import load_settings
from regression_gate.storage import Store

ROOT = Path(__file__).resolve().parent.parent
GOOD = '{"category": "billing", "summary": "Charged twice and wants a refund."}'


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    async def instant(_seconds):
        pass

    monkeypatch.setattr(llm.asyncio, "sleep", instant)


def judge(covered, client_text=None, **kw):
    client = RoutingClient(classify=None, judge=lambda _: fake_response(client_text or judge_json(covered), model="gpt-4.1"))
    return asyncio.run(judge_summary("email", ("a", "b"), "summary", model="gpt-4.1", temperature=0, client=client, **kw))


# --- judge


def test_judge_scores_fraction_of_facts_covered():
    out = judge([True, False])
    assert out.score == 0.5 and out.covered == [True, False] and out.error is None


def test_judge_rejects_wrong_fact_numbers():
    out = judge(None, client_text=judge_json([True]))
    assert out.score is None and "fact numbers" in out.error


def test_judge_invalid_output_is_an_error_not_a_score():
    out = judge(None, client_text="not json")
    assert out.score is None and out.error


# --- runner


def _client(classify_text=GOOD):
    def judge_all_covered(request):
        n_facts = len(re.findall(r"^\d+\. ", request["input"], flags=re.M))
        return fake_response(judge_json([True] * n_facts), model="gpt-4.1-2025-04-14")

    return RoutingClient(classify=lambda _: fake_response(classify_text), judge=judge_all_covered)


def _run(client, store, limit=3):
    dataset = load_dataset(ROOT / "data/golden.yaml")
    dataset = dataset.model_copy(update={"cases": dataset.cases[:limit]})
    path = prompt_path("v1", ROOT / "prompts")
    return asyncio.run(run_eval(load_prompt(path), path, dataset, SETTINGS, client, store))


def test_run_scores_every_case_and_records_metadata(tmp_path):
    record = _run(_client(), Store(tmp_path / "db.sqlite"))
    assert len(record.cases) == 3
    assert record.metrics.accuracy == 1.0  # the first 3 golden cases are billing
    assert record.metrics.summary_score == 1.0
    assert record.prompt_version == "v1" and record.judge_version == "j1"
    assert record.metrics.served_models == ["gpt-4.1-mini-2025-04-14"]


def test_judge_cache_reuses_verdicts_for_identical_summaries(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    _run(_client(), store)
    client = _client()
    record = _run(client, store)
    assert all(c.judge.cached for c in record.cases)
    assert not any(call["text"]["format"]["name"] == "summary_verdict" for call in client.calls)
    assert record.metrics.judge_cost_usd == 0


def test_api_errors_are_excluded_from_accuracy(tmp_path):
    import httpx2 as httpx
    from openai import APITimeoutError

    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    client = RoutingClient(classify=lambda _: APITimeoutError(request), judge=lambda _: None)
    record = _run(client, Store(tmp_path / "db.sqlite"))
    assert record.metrics.evaluated == 0
    assert record.metrics.accuracy is None
    assert record.metrics.infra_error_rate == 1.0


def test_quality_failure_counts_as_wrong_with_zero_summary(tmp_path):
    record = _run(_client(classify_text="not json"), Store(tmp_path / "db.sqlite"))
    assert record.metrics.accuracy == 0.0
    assert record.metrics.summary_score == 0.0
    assert record.metrics.invalid_output_rate == 1.0


def test_store_round_trips_runs(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    record = _run(_client(), store)
    store.save_run(record)
    assert store.load_run(record.run_id) == record
    assert [r["run_id"] for r in store.list_runs("v1")] == [record.run_id]


# --- metrics


def test_metrics_group_by_category_and_tag():
    cases = [
        case_result("case-001", "billing", "billing", tags=["typos"]),
        case_result("case-002", "billing", "general", tags=["typos"]),
        case_result("case-003", "technical", "technical"),
        case_result("case-004", "technical", error=ErrorType.API_ERROR),
    ]
    m = run_record(cases).metrics
    assert m.accuracy == pytest.approx(2 / 3)
    assert (m.by_category["billing"].correct, m.by_category["billing"].total) == (1, 2)
    assert m.by_category["technical"].total == 1  # the API error is excluded
    assert (m.by_tag["typos"].correct, m.by_tag["typos"].total) == (1, 2)
    assert m.infra_error_rate == 0.25


# --- settings


def test_price_lookup_uses_longest_prefix():
    assert SETTINGS.price_for("gpt-4.1-mini-2025-04-14").input == 0.40
    assert SETTINGS.price_for("gpt-4.1-2025-04-14").input == 2.00
    assert SETTINGS.price_for("unknown-model") is None


def test_unknown_settings_field_is_rejected(tmp_path):
    text = (ROOT / "gate.yaml").read_text() + "\nconcurency: 4\n"
    path = tmp_path / "gate.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="concurency"):
        load_settings(path)
