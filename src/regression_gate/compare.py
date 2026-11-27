"""Compare a candidate run against a baseline run and decide PASS / WARN / FAIL / INVALID."""

from enum import StrEnum
from math import comb

from pydantic import BaseModel

from regression_gate.records import CaseResult, RunRecord
from regression_gate.settings import Thresholds

# A per-case summary score drop at least this large (e.g. losing 1 of 2 key facts) is listed.
SUMMARY_DROP_LISTED = 0.5
EPSILON = 1e-9


class Status(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    # Too many API errors to judge quality: neither pass nor fail, rerun.
    INVALID = "invalid"


class CaseDiff(BaseModel):
    case_id: str
    email: str
    expected_category: str
    tags: list[str]
    reviewed: bool
    baseline_category: str | None
    candidate_category: str | None
    baseline_summary: str | None
    candidate_summary: str | None
    baseline_score: float | None
    candidate_score: float | None
    baseline_error: str | None
    candidate_error: str | None


class GroupDelta(BaseModel):
    baseline: float | None
    candidate: float | None
    delta: float | None
    total: int


class Comparison(BaseModel):
    status: Status
    reasons: list[str]
    notes: list[str]
    baseline_run_id: str
    candidate_run_id: str
    baseline_prompt: str
    candidate_prompt: str
    shared_cases: int
    paired_cases: int
    added_case_ids: list[str]
    removed_case_ids: list[str]
    # Accuracy and summary score on cases evaluated in both runs (a paired comparison).
    baseline_accuracy: float | None
    candidate_accuracy: float | None
    accuracy_delta: float | None
    baseline_summary_score: float | None
    candidate_summary_score: float | None
    summary_delta: float | None
    by_category: dict[str, GroupDelta]
    by_tag: dict[str, GroupDelta]
    regressions: list[CaseDiff]
    improvements: list[CaseDiff]
    summary_regressions: list[CaseDiff]
    # McNemar exact test on the flipped cases: could this many flips be noise?
    mcnemar_p: float | None
    baseline_infra_error_rate: float
    candidate_infra_error_rate: float
    unreviewed_cases: int


def compare_runs(baseline: RunRecord, candidate: RunRecord, thresholds: Thresholds) -> Comparison:
    base_cases = {c.case_id: c for c in baseline.cases}
    cand_cases = {c.case_id: c for c in candidate.cases}
    shared_ids = sorted(base_cases.keys() & cand_cases.keys())
    paired = [(base_cases[i], cand_cases[i]) for i in shared_ids if base_cases[i].evaluated and cand_cases[i].evaluated]

    regressions = [_diff(b, c) for b, c in paired if b.category_correct and not c.category_correct]
    improvements = [_diff(b, c) for b, c in paired if not b.category_correct and c.category_correct]

    judges_match = (baseline.judge_model, baseline.judge_version) == (candidate.judge_model, candidate.judge_version)
    scored = [(b, c) for b, c in paired if b.summary_score is not None and c.summary_score is not None]
    summary_regressions = [
        _diff(b, c) for b, c in scored if b.summary_score - c.summary_score >= SUMMARY_DROP_LISTED - EPSILON
    ] if judges_match else []

    base_acc = _mean([b.category_correct for b, _ in paired])
    cand_acc = _mean([c.category_correct for _, c in paired])
    base_sum = _mean([b.summary_score for b, _ in scored])
    cand_sum = _mean([c.summary_score for _, c in scored])

    by_category = _group_deltas(paired, lambda r: [r.expected_category])
    by_tag = _group_deltas(paired, lambda r: r.tags)

    comparison = Comparison(
        status=Status.PASS,
        reasons=[],
        notes=_notes(baseline, candidate, shared_ids, judges_match),
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_prompt=baseline.prompt_version,
        candidate_prompt=candidate.prompt_version,
        shared_cases=len(shared_ids),
        paired_cases=len(paired),
        added_case_ids=sorted(cand_cases.keys() - base_cases.keys()),
        removed_case_ids=sorted(base_cases.keys() - cand_cases.keys()),
        baseline_accuracy=base_acc,
        candidate_accuracy=cand_acc,
        accuracy_delta=_delta(base_acc, cand_acc),
        baseline_summary_score=base_sum if judges_match else None,
        candidate_summary_score=cand_sum if judges_match else None,
        summary_delta=_delta(base_sum, cand_sum) if judges_match else None,
        by_category=by_category,
        by_tag=by_tag,
        regressions=regressions,
        improvements=improvements,
        summary_regressions=summary_regressions,
        mcnemar_p=mcnemar_exact(len(regressions), len(improvements)),
        baseline_infra_error_rate=baseline.metrics.infra_error_rate,
        candidate_infra_error_rate=candidate.metrics.infra_error_rate,
        unreviewed_cases=sum(1 for i in shared_ids if not cand_cases[i].reviewed),
    )
    status, reasons = decide(comparison, thresholds)
    return comparison.model_copy(update={"status": status, "reasons": reasons})


def decide(c: Comparison, t: Thresholds) -> tuple[Status, list[str]]:
    worst_infra = max(c.baseline_infra_error_rate, c.candidate_infra_error_rate)
    if worst_infra > t.max_infra_error_rate + EPSILON:
        return Status.INVALID, [
            f"API error rate {worst_infra:.0%} exceeds {t.max_infra_error_rate:.0%}: results not trustworthy, rerun"
        ]
    if c.paired_cases == 0:
        return Status.INVALID, ["no cases were evaluated in both runs"]

    critical, warnings = [], []

    def check(label: str, drop: float | None, warn: float, crit: float) -> None:
        if drop is None:
            return
        message = f"{label} dropped {drop * 100:.1f} points"
        if drop > crit + EPSILON:
            critical.append(f"{message} (critical above {crit * 100:.0f})")
        elif drop > warn + EPSILON:
            warnings.append(f"{message} (warning above {warn * 100:.0f})")

    check("accuracy", _neg(c.accuracy_delta), t.accuracy_warn_drop, t.accuracy_critical_drop)
    for name, group in c.by_category.items():
        check(f"{name} accuracy", _neg(group.delta), t.category_warn_drop, t.category_critical_drop)
    check("summary score", _neg(c.summary_delta), t.summary_warn_drop, t.summary_critical_drop)

    if critical:
        return Status.FAIL, critical + warnings
    if warnings:
        return Status.WARN, warnings
    return Status.PASS, []


def mcnemar_exact(regressions: int, improvements: int) -> float | None:
    """Two-sided exact McNemar test: if the change made no difference, each flipped case
    is equally likely to flip either way. Small p means the imbalance is unlikely to be noise."""
    n = regressions + improvements
    if n == 0:
        return None
    k = min(regressions, improvements)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def _notes(baseline: RunRecord, candidate: RunRecord, shared_ids: list[str], judges_match: bool) -> list[str]:
    notes = []
    if baseline.dataset_hash != candidate.dataset_hash:
        notes.append(
            f"golden dataset changed (v{baseline.dataset_version} {baseline.dataset_hash} -> "
            f"v{candidate.dataset_version} {candidate.dataset_hash}); compared on the {len(shared_ids)} shared cases"
        )
    if not judges_match:
        notes.append(
            f"judge changed ({baseline.judge_model}/{baseline.judge_version} -> "
            f"{candidate.judge_model}/{candidate.judge_version}); summary scores not compared"
        )
    if baseline.metrics.served_models != candidate.metrics.served_models:
        notes.append(f"served model changed: {baseline.metrics.served_models} -> {candidate.metrics.served_models}")
    return notes


def _diff(b: CaseResult, c: CaseResult) -> CaseDiff:
    return CaseDiff(
        case_id=c.case_id,
        email=c.email,
        expected_category=c.expected_category,
        tags=c.tags,
        reviewed=c.reviewed,
        baseline_category=b.predicted_category,
        candidate_category=c.predicted_category,
        baseline_summary=b.outcome.result.summary if b.outcome.result else None,
        candidate_summary=c.outcome.result.summary if c.outcome.result else None,
        baseline_score=b.summary_score,
        candidate_score=c.summary_score,
        baseline_error=b.outcome.error_message,
        candidate_error=c.outcome.error_message,
    )


def _group_deltas(paired: list[tuple[CaseResult, CaseResult]], key) -> dict[str, GroupDelta]:
    groups: dict[str, list[tuple[bool, bool]]] = {}
    for b, c in paired:
        for name in key(c):
            groups.setdefault(name, []).append((b.category_correct, c.category_correct))
    out = {}
    for name in sorted(groups):
        pairs = groups[name]
        base, cand = _mean([p[0] for p in pairs]), _mean([p[1] for p in pairs])
        out[name] = GroupDelta(baseline=base, candidate=cand, delta=_delta(base, cand), total=len(pairs))
    return out


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _delta(base: float | None, cand: float | None) -> float | None:
    return None if base is None or cand is None else cand - base


def _neg(delta: float | None) -> float | None:
    return None if delta is None else -delta
