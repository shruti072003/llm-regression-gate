"""Aggregate per-case results into run metrics."""

from collections import defaultdict
from statistics import mean, quantiles

from regression_gate.classifier import ErrorType
from regression_gate.records import CaseResult, GroupAccuracy, RunMetrics
from regression_gate.settings import GateSettings


def compute_metrics(results: list[CaseResult], settings: GateSettings) -> RunMetrics:
    n = len(results)
    evaluated = [r for r in results if r.evaluated]
    outcomes = [r.outcome for r in results]

    def rate(error_type: ErrorType) -> float:
        return sum(o.error_type is error_type for o in outcomes) / n if n else 0.0

    def group(key) -> dict[str, GroupAccuracy]:
        groups: dict[str, list[bool]] = defaultdict(list)
        for r in evaluated:
            for name in key(r):
                groups[name].append(r.category_correct)
        return {name: GroupAccuracy(correct=sum(v), total=len(v)) for name, v in sorted(groups.items())}

    scores = [r.summary_score for r in results if r.summary_score is not None]
    judged = [r.judge for r in results if r.judge and r.judge.error is None]
    latencies = sorted(o.latency_ms for o in outcomes if o.latency_ms is not None)

    classifier_costs = [settings.cost(o.served_model, o.input_tokens, o.output_tokens) for o in outcomes if o.served_model]
    # Cached verdicts cost nothing in this run.
    judge_costs = [
        settings.cost(r.judge.served_model, r.judge.input_tokens, r.judge.output_tokens)
        for r in results
        if r.judge and r.judge.served_model and not r.judge.cached
    ]

    return RunMetrics(
        cases=n,
        evaluated=len(evaluated),
        accuracy=sum(r.category_correct for r in evaluated) / len(evaluated) if evaluated else None,
        summary_score=mean(scores) if scores else None,
        by_category=group(lambda r: [r.expected_category]),
        by_difficulty=group(lambda r: [r.difficulty]),
        by_tag=group(lambda r: r.tags),
        infra_error_rate=rate(ErrorType.API_ERROR),
        invalid_output_rate=rate(ErrorType.INVALID_OUTPUT),
        refusal_rate=rate(ErrorType.REFUSAL),
        truncated_rate=rate(ErrorType.TRUNCATED),
        normalized_rate=sum(o.category_normalized for o in outcomes) / n if n else 0.0,
        unsupported_claim_rate=sum(j.unsupported_claim for j in judged) / len(judged) if judged else None,
        judge_error_rate=sum(1 for r in results if r.judge and r.judge.error) / n if n else 0.0,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        input_tokens=sum(o.input_tokens or 0 for o in outcomes),
        output_tokens=sum(o.output_tokens or 0 for o in outcomes),
        classifier_cost_usd=_total(classifier_costs),
        judge_cost_usd=_total(judge_costs) if judge_costs else 0.0,
        served_models=sorted({o.served_model for o in outcomes if o.served_model}),
    )


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[pct - 1]


def _total(costs: list[float | None]) -> float | None:
    # Unknown if any served model had no configured price, rather than a misleading partial sum.
    return None if not costs or any(c is None for c in costs) else sum(costs)
