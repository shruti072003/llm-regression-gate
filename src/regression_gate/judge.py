"""LLM-as-judge: does a summary cover the case's key facts without inventing anything?"""

import hashlib
import json

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from regression_gate.llm import create_with_retries, find_refusal, strict_json_format, usage_tokens

# Bump whenever the judge prompt or output format changes: scores from different
# judge versions are not comparable, and runs record which version scored them.
JUDGE_VERSION = "j1"
MAX_OUTPUT_TOKENS = 400

JUDGE_INSTRUCTIONS = """\
You grade a one-sentence summary of a customer support email.

You get the email, a numbered list of key facts, and the summary.

1. For each key fact, decide whether the summary states it, or something with the same meaning.
   Different wording is fine. A fact that is only implied, or stated less specifically, is not covered.
2. Decide whether the summary claims anything the email does not support (an invented detail,
   a wrong number, the wrong product area).

Write your reasoning first, then the verdicts. Return one entry per key fact, using its number."""


class FactCheck(BaseModel):
    index: int
    covered: bool


class JudgeVerdict(BaseModel):
    # Reasoning comes first so the model reasons before it commits to verdicts.
    reasoning: str = Field(min_length=1)
    facts: list[FactCheck]
    unsupported_claim: bool


class JudgeOutcome(BaseModel):
    """Score for one summary. score is the fraction of key facts covered."""

    score: float | None
    covered: list[bool] | None
    unsupported_claim: bool | None
    reasoning: str | None
    error: str | None
    served_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cached: bool = False


def cache_key(model: str, email: str, key_facts: tuple[str, ...], summary: str) -> str:
    payload = json.dumps([JUDGE_VERSION, model, email, list(key_facts), summary])
    return hashlib.sha256(payload.encode()).hexdigest()


def build_input(email: str, key_facts: tuple[str, ...], summary: str) -> str:
    facts = "\n".join(f"{i}. {fact}" for i, fact in enumerate(key_facts, start=1))
    return f"<email>\n{email.strip()}\n</email>\n\n<key_facts>\n{facts}\n</key_facts>\n\n<summary>\n{summary}\n</summary>"


async def judge_summary(
    email: str, key_facts: tuple[str, ...], summary: str, *, model: str, temperature: float, client: AsyncOpenAI
) -> JudgeOutcome:
    call = await create_with_retries(
        client,
        model=model,
        instructions=JUDGE_INSTRUCTIONS,
        input=build_input(email, key_facts, summary),
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        text={"format": strict_json_format(JudgeVerdict, "summary_verdict")},
    )
    empty = {"score": None, "covered": None, "unsupported_claim": None, "reasoning": None}
    if call.response is None:
        return JudgeOutcome(**empty, error=call.error, served_model=None, input_tokens=None, output_tokens=None)

    response = call.response
    input_tokens, output_tokens = usage_tokens(response)
    meta = {"served_model": response.model, "input_tokens": input_tokens, "output_tokens": output_tokens}

    refusal = find_refusal(response)
    if refusal is not None or response.status == "incomplete":
        return JudgeOutcome(**empty, error=f"judge gave no verdict: {refusal or response.status}", **meta)

    try:
        verdict = JudgeVerdict.model_validate_json(response.output_text)
    except ValidationError as e:
        return JudgeOutcome(**empty, error=f"invalid judge output: {e}", **meta)

    by_index = {check.index: check.covered for check in verdict.facts}
    if sorted(by_index) != list(range(1, len(key_facts) + 1)):
        return JudgeOutcome(**empty, error=f"judge returned fact numbers {sorted(by_index)} for {len(key_facts)} facts", **meta)

    covered = [by_index[i] for i in range(1, len(key_facts) + 1)]
    return JudgeOutcome(
        score=sum(covered) / len(covered),
        covered=covered,
        unsupported_claim=verdict.unsupported_claim,
        reasoning=verdict.reasoning,
        error=None,
        **meta,
    )
