"""Builders for fake API responses and synthetic run records."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from regression_gate.classifier import ClassificationOutcome, ErrorType
from regression_gate.judge import JudgeOutcome
from regression_gate.records import CaseResult, RunRecord
from regression_gate.settings import load_settings

SETTINGS = load_settings()


def fake_response(text="", *, status="completed", refusal=None, reason=None, model="gpt-4.1-mini-2025-04-14"):
    content = [SimpleNamespace(type="refusal", refusal=refusal)] if refusal else [SimpleNamespace(type="output_text", text=text)]
    return SimpleNamespace(
        output_text=text,
        output=[SimpleNamespace(type="message", content=content)],
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        model=model,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


class RoutingClient:
    """Fake client that answers classifier and judge requests differently, keyed by
    the structured-output schema name."""

    def __init__(self, classify, judge):
        self.classify, self.judge = classify, judge
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        handler = self.judge if kwargs["text"]["format"]["name"] == "summary_verdict" else self.classify
        result = handler(kwargs)
        if isinstance(result, Exception):
            raise result
        return result


def judge_json(covered: list[bool], unsupported=False) -> str:
    return json.dumps({
        "reasoning": "checked",
        "facts": [{"index": i, "covered": c} for i, c in enumerate(covered, start=1)],
        "unsupported_claim": unsupported,
    })


def case_result(case_id, expected="billing", predicted="billing", *, score=1.0, tags=(), error=None, reviewed=True):
    if error is ErrorType.API_ERROR:
        outcome = _outcome(None, error)
        return _case(case_id, expected, tags, reviewed, outcome, None, None, None)
    if error is not None:
        return _case(case_id, expected, tags, reviewed, _outcome(None, error), False, None, 0.0)
    outcome = _outcome(predicted, None)
    judge = JudgeOutcome(score=score, covered=[score == 1.0], unsupported_claim=False, reasoning="ok", error=None,
                         served_model="gpt-4.1-2025-04-14", input_tokens=300, output_tokens=80)
    return _case(case_id, expected, tags, reviewed, outcome, predicted == expected, judge, score)


def run_record(cases, *, run_id="run", prompt="v1", dataset_hash="hash1", judge_version="j1",
               served="gpt-4.1-mini-2025-04-14", created_at=None) -> RunRecord:
    from regression_gate.metrics import compute_metrics

    metrics = compute_metrics(cases, SETTINGS)
    if served != "gpt-4.1-mini-2025-04-14":
        metrics = metrics.model_copy(update={"served_models": [served]})
    return RunRecord(
        run_id=run_id, created_at=created_at or datetime(2026, 11, 1, tzinfo=UTC), prompt_version=prompt,
        prompt_sha="abc", model="gpt-4.1-mini", temperature=0, dataset_version=1, dataset_hash=dataset_hash,
        judge_model="gpt-4.1", judge_version=judge_version, git_sha="deadbeef", duration_s=1.0,
        cases=cases, metrics=metrics,
    )


def days(n: int) -> datetime:
    return datetime(2026, 11, 1, tzinfo=UTC) + timedelta(days=n)


def _outcome(predicted, error) -> ClassificationOutcome:
    result = {"category": predicted, "summary": "A summary."} if predicted else None
    return ClassificationOutcome.model_validate({
        "result": result, "error_type": error, "error_message": str(error) if error else None,
        "raw_output": None, "category_normalized": False, "requested_model": "gpt-4.1-mini",
        "served_model": None if error is ErrorType.API_ERROR else "gpt-4.1-mini-2025-04-14",
        "latency_ms": None if error else 800.0, "input_tokens": None if error else 280,
        "output_tokens": None if error else 30, "attempts": 1,
    })


def _case(case_id, expected, tags, reviewed, outcome, correct, judge, score) -> CaseResult:
    return CaseResult(case_id=case_id, email=f"Email for {case_id} <b>bold</b>", expected_category=expected,
                      key_facts=["fact"], difficulty="easy", tags=list(tags), reviewed=reviewed, outcome=outcome,
                      category_correct=correct, judge=judge, summary_score=score)
