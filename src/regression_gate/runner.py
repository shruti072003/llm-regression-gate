"""Run a prompt version against the golden dataset and score every case."""

import asyncio
import hashlib
import os
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI

from regression_gate.classifier import classify
from regression_gate.dataset import GoldenCase, GoldenDataset
from regression_gate.judge import JUDGE_VERSION, cache_key, judge_summary
from regression_gate.metrics import compute_metrics
from regression_gate.prompts import PromptConfig
from regression_gate.records import CaseResult, RunRecord
from regression_gate.settings import GateSettings
from regression_gate.storage import Store


async def run_eval(
    config: PromptConfig,
    prompt_path: Path,
    dataset: GoldenDataset,
    settings: GateSettings,
    client: AsyncOpenAI,
    store: Store,
    progress=None,
) -> RunRecord:
    started = time.perf_counter()
    created_at = datetime.now(UTC)
    # Bounds parallel requests: fast enough for CI, gentle enough to avoid rate limits.
    semaphore = asyncio.Semaphore(settings.concurrency)

    async def evaluate(case: GoldenCase) -> CaseResult:
        async with semaphore:
            result = await _evaluate_case(case, config, settings, client, store)
        if progress:
            progress(result)
        return result

    results = await asyncio.gather(*(evaluate(case) for case in dataset.cases))

    return RunRecord(
        run_id=f"{created_at:%Y%m%dT%H%M%SZ}-{config.version_id}-{uuid.uuid4().hex[:6]}",
        created_at=created_at,
        prompt_version=config.version_id,
        prompt_sha=hashlib.sha256(Path(prompt_path).read_bytes()).hexdigest()[:12],
        model=config.model,
        temperature=config.temperature,
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        judge_model=settings.judge.model,
        judge_version=JUDGE_VERSION,
        git_sha=_git_sha(),
        duration_s=time.perf_counter() - started,
        cases=list(results),
        metrics=compute_metrics(list(results), settings),
    )


async def _evaluate_case(
    case: GoldenCase, config: PromptConfig, settings: GateSettings, client: AsyncOpenAI, store: Store
) -> CaseResult:
    outcome = await classify(case.email, config, client)
    base = {
        "case_id": case.id,
        "email": case.email,
        "expected_category": case.expected_category.value,
        "key_facts": list(case.key_facts),
        "difficulty": case.difficulty.value,
        "tags": [t.value for t in case.tags],
        "reviewed": case.reviewed,
        "outcome": outcome,
    }

    if outcome.is_infra_error:
        return CaseResult(**base, category_correct=None, judge=None, summary_score=None)
    if outcome.result is None:
        # A quality failure (invalid output, refusal, truncation): wrong, with no usable summary.
        return CaseResult(**base, category_correct=False, judge=None, summary_score=0.0)

    category_correct = outcome.result.category == case.expected_category
    summary = outcome.result.summary
    key = cache_key(settings.judge.model, case.email, case.key_facts, summary)
    verdict = store.get_judge(key)
    if verdict is not None:
        verdict = verdict.model_copy(update={"cached": True})
    else:
        verdict = await judge_summary(
            case.email, case.key_facts, summary,
            model=settings.judge.model, temperature=settings.judge.temperature, client=client,
        )
        if verdict.error is None:
            store.put_judge(key, verdict)

    return CaseResult(**base, category_correct=category_correct, judge=verdict, summary_score=verdict.score)


def _git_sha() -> str | None:
    # The checkout's own commit first: in CI the baseline runs from a separate checkout of
    # main, where GITHUB_SHA would wrongly point at the PR.
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = os.environ.get("GITHUB_SHA")
        return sha[:12] if sha else None
