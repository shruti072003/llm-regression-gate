"""The LLM feature under test: classify a support email into a category and summary."""

import json
from enum import StrEnum

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from regression_gate.llm import MAX_ATTEMPTS, create_with_retries, find_refusal, make_client, strict_json_format, usage_tokens
from regression_gate.prompts import PromptConfig
from regression_gate.schemas import EmailClassification

__all__ = ["MAX_ATTEMPTS", "ClassificationOutcome", "ErrorType", "build_input", "classify", "make_client", "response_format"]

MAX_OUTPUT_TOKENS = 200


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


def build_input(email: str, config: PromptConfig) -> list[dict]:
    messages = []
    for example in config.few_shot:
        messages.append({"role": "user", "content": example.email})
        messages.append({"role": "assistant", "content": example.output.model_dump_json()})
    messages.append({"role": "user", "content": email})
    return messages


def response_format() -> dict:
    return strict_json_format(EmailClassification, "email_classification")


async def classify(email: str, config: PromptConfig, client: AsyncOpenAI) -> ClassificationOutcome:
    common = {"requested_model": config.model, "category_normalized": False}

    call = await create_with_retries(
        client,
        model=config.model,
        instructions=config.system_prompt,
        input=build_input(email, config),
        temperature=config.temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        text={"format": response_format()},
    )
    if call.response is None:
        return _failure(ErrorType.API_ERROR, call.error, call.attempts, **common)

    response = call.response
    input_tokens, output_tokens = usage_tokens(response)
    evidence = {
        **common,
        "raw_output": response.output_text or None,
        "served_model": response.model,
        "latency_ms": call.latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "attempts": call.attempts,
    }

    refusal = find_refusal(response)
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


def _failure(error_type: ErrorType, message: str, attempts: int, **fields) -> ClassificationOutcome:
    defaults = {"raw_output": None, "served_model": None, "latency_ms": None, "input_tokens": None, "output_tokens": None}
    return ClassificationOutcome(
        result=None,
        error_type=error_type,
        error_message=message,
        attempts=attempts,
        **{**defaults, **fields},
    )
