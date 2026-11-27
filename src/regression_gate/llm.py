"""Shared OpenAI plumbing for the classifier and the judge: retries and structured outputs."""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIError, AsyncOpenAI, InternalServerError, RateLimitError
from pydantic import BaseModel

TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 1.0

# Transient failures worth retrying. APITimeoutError is a subclass of APIConnectionError.
RETRYABLE_ERRORS = (APIConnectionError, RateLimitError, InternalServerError)

# Length limits are left out of API schemas on purpose. Enforced during generation,
# maxLength cut summaries off mid-sentence (and forced odd tokens), yet the result
# still passed validation. They are checked locally instead, where a violation is visible.
API_SCHEMA_EXCLUDED_KEYS = {"minLength", "maxLength"}


def make_client() -> AsyncOpenAI:
    # SDK retries are off so create_with_retries controls them and can count attempts.
    return AsyncOpenAI(timeout=TIMEOUT_SECONDS, max_retries=0)


@dataclass
class ApiCall:
    response: Any | None
    error: str | None
    attempts: int
    # Duration of the successful attempt only; retry waits are excluded.
    latency_ms: float | None


async def create_with_retries(client: AsyncOpenAI, **request) -> ApiCall:
    attempts = 0
    while True:
        attempts += 1
        started = time.perf_counter()
        try:
            response = await client.responses.create(**request)
            return ApiCall(response, None, attempts, (time.perf_counter() - started) * 1000)
        except RETRYABLE_ERRORS as e:
            if attempts >= MAX_ATTEMPTS:
                return ApiCall(None, f"{type(e).__name__} after {attempts} attempts: {e}", attempts, None)
            # Exponential backoff with jitter, so parallel requests don't retry in lockstep.
            await asyncio.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempts - 1) * random.uniform(0.5, 1.5))
        except APIError as e:
            # Not transient (bad request, auth, unknown model): retrying won't help.
            return ApiCall(None, f"{type(e).__name__}: {e}", attempts, None)


def strict_json_format(model: type[BaseModel], name: str) -> dict:
    """Structured-output format for `model`. The API enforces structure (keys, enums);
    local validation also enforces length."""
    schema = _without_keys(model.model_json_schema(), API_SCHEMA_EXCLUDED_KEYS)
    _forbid_extra_keys(schema)
    return {"type": "json_schema", "name": name, "schema": schema, "strict": True}


def find_refusal(response) -> str | None:
    for item in response.output:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return part.refusal
    return None


def usage_tokens(response) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    return (usage.input_tokens, usage.output_tokens) if usage else (None, None)


def _without_keys(node, keys: set[str]):
    if isinstance(node, dict):
        return {k: _without_keys(v, keys) for k, v in node.items() if k not in keys}
    if isinstance(node, list):
        return [_without_keys(v, keys) for v in node]
    return node


def _forbid_extra_keys(node) -> None:
    # Strict mode requires additionalProperties: false on every object, including nested ones.
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
        for value in node.values():
            _forbid_extra_keys(value)
    elif isinstance(node, list):
        for value in node:
            _forbid_extra_keys(value)
