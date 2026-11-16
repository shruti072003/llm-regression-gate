import asyncio
from datetime import date
from types import SimpleNamespace

import httpx2 as httpx  # the HTTP library openai depends on
import pytest
from openai import APITimeoutError, AuthenticationError, RateLimitError

from regression_gate import classifier
from regression_gate.classifier import ErrorType, build_input, classify
from regression_gate.prompts import PromptConfig

CONFIG = PromptConfig(
    version_id="v1",
    created_at=date(2026, 11, 15),
    description="Test.",
    model="gpt-4.1-mini",
    temperature=0,
    system_prompt="Classify.",
    few_shot=[{"email": "Charged twice.", "output": {"category": "billing", "summary": "Duplicate charge."}}],
)

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")


def fake_response(text="", *, status="completed", refusal=None, reason=None):
    content = [SimpleNamespace(type="refusal", refusal=refusal)] if refusal else [SimpleNamespace(type="output_text", text=text)]
    return SimpleNamespace(
        output_text=text,
        output=[SimpleNamespace(type="message", content=content)],
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        model="gpt-4.1-mini-2025-04-14",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


class FakeClient:
    """Returns (or raises) the given items in order, one per API call."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    async def instant(_seconds):
        pass

    monkeypatch.setattr(classifier.asyncio, "sleep", instant)


def run(client):
    return asyncio.run(classify("My card was charged twice.", CONFIG, client))


def test_success_records_result_and_evidence():
    out = run(FakeClient(fake_response('{"category": "billing", "summary": "Charged twice."}')))
    assert out.error_type is None
    assert out.result.category == "billing"
    assert out.served_model == "gpt-4.1-mini-2025-04-14"
    assert (out.input_tokens, out.output_tokens, out.attempts) == (100, 20, 1)
    assert out.category_normalized is False


def test_category_casing_is_normalized_and_flagged():
    out = run(FakeClient(fake_response('{"category": "Billing", "summary": "Charged twice."}')))
    assert out.result.category == "billing"
    assert out.category_normalized is True


def test_invalid_json_is_an_invalid_output_failure_with_raw_text_kept():
    out = run(FakeClient(fake_response("Sure! Here is the category: billing")))
    assert out.error_type is ErrorType.INVALID_OUTPUT
    assert out.raw_output == "Sure! Here is the category: billing"
    assert out.input_tokens == 100
    assert not out.is_infra_error


def test_unknown_category_is_an_invalid_output_failure():
    out = run(FakeClient(fake_response('{"category": "payments", "summary": "Charged twice."}')))
    assert out.error_type is ErrorType.INVALID_OUTPUT


def test_refusal_is_recorded():
    out = run(FakeClient(fake_response(refusal="I can't help with that.")))
    assert out.error_type is ErrorType.REFUSAL
    assert out.error_message == "I can't help with that."


def test_truncated_response_is_recorded():
    out = run(FakeClient(fake_response('{"category": "bill', status="incomplete", reason="max_output_tokens")))
    assert out.error_type is ErrorType.TRUNCATED
    assert "max_output_tokens" in out.error_message


def test_transient_error_is_retried_then_succeeds():
    rate_limited = RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None)
    client = FakeClient(rate_limited, APITimeoutError(REQUEST), fake_response('{"category": "billing", "summary": "Charged twice."}'))
    out = run(client)
    assert out.error_type is None
    assert out.attempts == 3


def test_persistent_transient_error_becomes_infra_error():
    client = FakeClient(*[APITimeoutError(REQUEST) for _ in range(classifier.MAX_ATTEMPTS)])
    out = run(client)
    assert out.error_type is ErrorType.API_ERROR
    assert out.is_infra_error
    assert out.attempts == classifier.MAX_ATTEMPTS


def test_non_retryable_error_fails_immediately():
    auth = AuthenticationError("bad key", response=httpx.Response(401, request=REQUEST), body=None)
    client = FakeClient(auth)
    out = run(client)
    assert out.error_type is ErrorType.API_ERROR
    assert out.attempts == 1


def test_request_uses_config_and_structured_output():
    client = FakeClient(fake_response('{"category": "billing", "summary": "Charged twice."}'))
    run(client)
    call = client.calls[0]
    assert call["model"] == "gpt-4.1-mini"
    assert call["temperature"] == 0
    assert call["instructions"] == "Classify."
    fmt = call["text"]["format"]
    assert fmt["strict"] is True
    assert fmt["schema"]["additionalProperties"] is False


def test_few_shot_examples_become_user_assistant_pairs():
    messages = build_input("New email.", CONFIG)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == '{"category":"billing","summary":"Duplicate charge."}'
    assert messages[-1]["content"] == "New email."
