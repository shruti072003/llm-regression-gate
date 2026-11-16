"""The LLM feature under test: classify a support email into a category and summary."""

import asyncio
import json
import random
import time
from enum import StrEnum

from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from regression_gate.prompts import PromptConfig
from regression_gate.schemas import EmailClassification

TIMEOUT_SECONDS = 30
MAX_OUTPUT_TOKENS = 200
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 1.0

# Transient failures worth retrying. APITimeoutError is a subclass of APIConnectionError.
RETRYABLE_ERRORS = (APIConnectionError, RateLimitError, InternalServerError)


class ErrorType(StrEnum):
    # Quality failures: the model produced a bad answer. Count against accuracy.
    INVALID_OUTPUT = "invalid_output"
    REFUSAL = "refusal"
    TRUNCATED = "truncated"
    # Infrastructure failure: no answer to judge. Reported separately, never as a regression.
    API_ERROR = "api_error"


class ClassificationOutcome(BaseModel):
    """Everything one classify() call produced, success or failure."""

    result: EmailClassification | None
    error_type: ErrorType | None
    error_message: str | None
    raw_output: str | None
    # True when the model's category needed lowercasing or stripping to validate.
    category_normalized: bool
    requested_model: str
    # The snapshot that actually answered. Changes here, with no prompt change,
    # mean the provider updated the model behind the alias.
    served_model: str | None
    latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    attempts: int

    @property
    def is_infra_error(self) -> bool:
        return self.error_type is ErrorType.API_ERROR


def make_client() -> AsyncOpenAI:
    # SDK retries are off so this module controls them and can count attempts.
    return AsyncOpenAI(timeout=TIMEOUT_SECONDS, max_retries=0)


def build_input(email: str, config: PromptConfig) -> list[dict]:
    messages = []
    for example in config.few_shot:
        messages.append({"role": "user", "content": example.email})
        messages.append({"role": "assistant", "content": example.output.model_dump_json()})
    messages.append({"role": "user", "content": email})
    return messages


# Length limits are left out of the API schema on purpose. Enforced during
# generation, maxLength cut summaries off mid-sentence (and forced odd tokens),
# yet the result still passed validation. Checked locally instead, so a
# too-long summary becomes a visible invalid_output failure.
API_SCHEMA_EXCLUDED_KEYS = {"minLength", "maxLength"}


def response_format() -> dict:
    """Structured-output schema built from EmailClassification. The API enforces
    structure (keys, category enum); the local validator also enforces length."""
    schema = _without_keys(EmailClassification.model_json_schema(), API_SCHEMA_EXCLUDED_KEYS)
    schema["additionalProperties"] = False
    return {"type": "json_schema", "name": "email_classification", "schema": schema, "strict": True}


def _without_keys(node, keys: set[str]):
    if isinstance(node, dict):
        return {k: _without_keys(v, keys) for k, v in node.items() if k not in keys}
    if isinstance(node, list):
        return [_without_keys(v, keys) for v in node]
    return node


async def classify(email: str, config: PromptConfig, client: AsyncOpenAI) -> ClassificationOutcome:
    common = {"requested_model": config.model, "category_normalized": False}

    attempts = 0
    while True:
        attempts += 1
        started = time.perf_counter()
        try:
            response = await client.responses.create(
                model=config.model,
                instructions=config.system_prompt,
                input=build_input(email, config),
                temperature=config.temperature,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                text={"format": response_format()},
            )
            break
        except RETRYABLE_ERRORS as e:
            if attempts >= MAX_ATTEMPTS:
                return _failure(ErrorType.API_ERROR, f"{type(e).__name__} after {attempts} attempts: {e}", attempts, **common)
            # Exponential backoff with jitter, so parallel requests don't retry in lockstep.
            await asyncio.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempts - 1) * random.uniform(0.5, 1.5))
        except APIError as e:
            # Not transient (bad request, auth, unknown model): retrying won't help.
            return _failure(ErrorType.API_ERROR, f"{type(e).__name__}: {e}", attempts, **common)

    latency_ms = (time.perf_counter() - started) * 1000
    usage = response.usage
    evidence = {
        **common,
        "raw_output": response.output_text or None,
        "served_model": response.model,
        "latency_ms": latency_ms,
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
        "attempts": attempts,
    }

    refusal = _find_refusal(response)
    if refusal is not None:
        return _failure(ErrorType.REFUSAL, refusal, **evidence)

    if response.status == "incomplete":
        reason = getattr(response.incomplete_details, "reason", None) or "unknown"
        return _failure(ErrorType.TRUNCATED, f"response incomplete: {reason}", **evidence)

    try:
        data = json.loads(response.output_text)
        result = EmailClassification.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        return _failure(ErrorType.INVALID_OUTPUT, str(e), **evidence)

    evidence["category_normalized"] = data.get("category") != result.category.value
    return ClassificationOutcome(result=result, error_type=None, error_message=None, **evidence)


def _find_refusal(response) -> str | None:
    for item in response.output:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return part.refusal
    return None


def _failure(error_type: ErrorType, message: str, attempts: int, **fields) -> ClassificationOutcome:
    defaults = {"raw_output": None, "served_model": None, "latency_ms": None, "input_tokens": None, "output_tokens": None}
    return ClassificationOutcome(
        result=None,
        error_type=error_type,
        error_message=message,
        attempts=attempts,
        **{**defaults, **fields},
    )
