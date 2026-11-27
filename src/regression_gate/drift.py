"""Run history and slow-drift detection.

Per-run comparisons catch sudden regressions. Drift catches the slow kind: a moving average
that sinks over many runs without any single run crossing a threshold, for example when the
provider updates the model behind an alias.
"""

from datetime import datetime
from pathlib import Path
from statistics import mean

from pydantic import BaseModel

from regression_gate.records import RunRecord
from regression_gate.settings import DriftSettings

HISTORY_PATH = Path("results/history.jsonl")


class RunSummary(BaseModel):
    run_id: str
    created_at: datetime
    prompt_version: str
    dataset_version: int
    dataset_hash: str
    judge_version: str
    git_sha: str | None
    accuracy: float | None
    summary_score: float | None
    infra_error_rate: float
    served_models: list[str]
    cost_usd: float | None


class DriftReport(BaseModel):
    status: str  # "ok", "drift", or "insufficient_history"
    message: str
    comparable_runs: int
    recent_accuracy: float | None = None
    previous_accuracy: float | None = None
    recent_summary: float | None = None
    previous_summary: float | None = None
    served_model_changes: list[str] = []


def summarize(record: RunRecord) -> RunSummary:
    m = record.metrics
    costs = [c for c in (m.classifier_cost_usd, m.judge_cost_usd) if c is not None]
    return RunSummary(
        run_id=record.run_id,
        created_at=record.created_at,
        prompt_version=record.prompt_version,
        dataset_version=record.dataset_version,
        dataset_hash=record.dataset_hash,
        judge_version=record.judge_version,
        git_sha=record.git_sha,
        accuracy=m.accuracy,
        summary_score=m.summary_score,
        infra_error_rate=m.infra_error_rate,
        served_models=m.served_models,
        cost_usd=sum(costs) if costs else None,
    )


def load_history(path: Path = HISTORY_PATH) -> list[RunSummary]:
    path = Path(path)
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return sorted((RunSummary.model_validate_json(line) for line in lines), key=lambda s: s.created_at)


def append_history(summary: RunSummary, path: Path = HISTORY_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(summary.model_dump_json() + "\n")


def detect_drift(history: list[RunSummary], settings: DriftSettings, max_infra_error_rate: float) -> DriftReport:
    if not history:
        return DriftReport(status="insufficient_history", message="no runs in history", comparable_runs=0)

    latest = history[-1]
    # Only runs measured the same way are comparable: same prompt, dataset and judge.
    # Runs dominated by API errors are skipped, so an outage doesn't look like drift.
    comparable = [
        s for s in history
        if (s.prompt_version, s.dataset_hash, s.judge_version) == (latest.prompt_version, latest.dataset_hash, latest.judge_version)
        and s.infra_error_rate <= max_infra_error_rate
    ]
    model_changes = [
        f"{prev.served_models} -> {cur.served_models} at {cur.run_id}"
        for prev, cur in zip(comparable, comparable[1:])
        if prev.served_models != cur.served_models
    ]

    w = settings.window
    if len(comparable) < 2 * w:
        return DriftReport(
            status="insufficient_history",
            message=f"{len(comparable)} comparable runs; drift needs {2 * w} ({w} recent vs {w} before)",
            comparable_runs=len(comparable),
            served_model_changes=model_changes,
        )

    recent, previous = comparable[-w:], comparable[-2 * w : -w]
    rec_acc, prev_acc = _avg(s.accuracy for s in recent), _avg(s.accuracy for s in previous)
    rec_sum, prev_sum = _avg(s.summary_score for s in recent), _avg(s.summary_score for s in previous)

    problems = []
    if rec_acc is not None and prev_acc is not None and prev_acc - rec_acc > settings.accuracy_warn_drop:
        problems.append(f"accuracy {w}-run average fell {100 * (prev_acc - rec_acc):.1f} points ({prev_acc:.1%} -> {rec_acc:.1%})")
    if rec_sum is not None and prev_sum is not None and prev_sum - rec_sum > settings.summary_warn_drop:
        problems.append(f"summary score {w}-run average fell {100 * (prev_sum - rec_sum):.1f} points ({prev_sum:.1%} -> {rec_sum:.1%})")

    return DriftReport(
        status="drift" if problems else "ok",
        message="; ".join(problems) if problems else f"no drift across the last {2 * w} comparable runs",
        comparable_runs=len(comparable),
        recent_accuracy=rec_acc,
        previous_accuracy=prev_acc,
        recent_summary=rec_sum,
        previous_summary=prev_sum,
        served_model_changes=model_changes,
    )


def _avg(values) -> float | None:
    values = [v for v in values if v is not None]
    return mean(values) if values else None
