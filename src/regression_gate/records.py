"""Data recorded by an eval run: one result per golden case, plus run-level metrics."""

from datetime import datetime

from pydantic import BaseModel

from regression_gate.classifier import ClassificationOutcome
from regression_gate.judge import JudgeOutcome


class CaseResult(BaseModel):
    case_id: str
    email: str
    expected_category: str
    key_facts: list[str]
    difficulty: str
    tags: list[str]
    reviewed: bool
    outcome: ClassificationOutcome
    # None when the case couldn't be evaluated (API error): excluded from accuracy.
    category_correct: bool | None
    judge: JudgeOutcome | None
    # Fraction of key facts covered. 0 when the model gave no usable answer (a quality
    # failure); None when it couldn't be scored (API or judge error).
    summary_score: float | None

    @property
    def evaluated(self) -> bool:
        return self.category_correct is not None

    @property
    def predicted_category(self) -> str | None:
        return self.outcome.result.category.value if self.outcome.result else None


class GroupAccuracy(BaseModel):
    correct: int
    total: int

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None


class RunMetrics(BaseModel):
    cases: int
    evaluated: int
    accuracy: float | None
    summary_score: float | None
    by_category: dict[str, GroupAccuracy]
    by_difficulty: dict[str, GroupAccuracy]
    by_tag: dict[str, GroupAccuracy]
    infra_error_rate: float
    invalid_output_rate: float
    refusal_rate: float
    truncated_rate: float
    normalized_rate: float
    unsupported_claim_rate: float | None
    judge_error_rate: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    input_tokens: int
    output_tokens: int
    classifier_cost_usd: float | None
    judge_cost_usd: float | None
    served_models: list[str]


class RunRecord(BaseModel):
    run_id: str
    created_at: datetime
    prompt_version: str
    prompt_sha: str
    model: str
    temperature: float
    dataset_version: int
    dataset_hash: str
    judge_model: str
    judge_version: str
    git_sha: str | None
    duration_s: float
    cases: list[CaseResult]
    metrics: RunMetrics
