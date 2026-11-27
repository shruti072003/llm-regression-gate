"""Tests for the comparison decision, drift detection, reports and the CLI exit codes."""

import pytest

from helpers import SETTINGS, case_result, days, run_record
from regression_gate.classifier import ErrorType
from regression_gate.cli import main
from regression_gate.compare import Status, compare_runs, mcnemar_exact
from regression_gate.drift import RunSummary, append_history, detect_drift, load_history, summarize
from regression_gate.notify import comparison_message, send
from regression_gate.report import render_html, render_markdown

T = SETTINGS.thresholds


def cases(n_wrong=0, n=60, category="billing", **kw):
    """n cases, the first n_wrong predicted wrong."""
    return [
        case_result(f"case-{i:03d}", category, "general" if i <= n_wrong else category, **kw)
        for i in range(1, n + 1)
    ]


def compare(base_cases, cand_cases, **kw):
    return compare_runs(run_record(base_cases, run_id="base"), run_record(cand_cases, run_id="cand", **kw), T)


# --- decision


def test_identical_runs_pass():
    c = compare(cases(), cases())
    assert c.status is Status.PASS and c.regressions == [] and c.mcnemar_p is None


def test_one_flip_is_within_noise_and_passes():
    c = compare(cases(), cases(n_wrong=1))
    assert c.status is Status.PASS
    assert [d.case_id for d in c.regressions] == ["case-001"]


def test_drop_above_warn_threshold_warns():
    c = compare(cases(), cases(n_wrong=3, n=60))  # 5 points: above 3 (warn), below 8 (critical)
    assert c.status is Status.WARN


def test_drop_above_critical_threshold_fails():
    c = compare(cases(), cases(n_wrong=6))  # 10 points
    assert c.status is Status.FAIL
    assert any("accuracy dropped 10.0 points" in r for r in c.reasons)


def test_category_drop_can_fail_even_when_overall_drop_is_small():
    base = cases(n=45, category="billing") + cases(n=15, category="general")
    base = [c.model_copy(update={"case_id": f"case-{i:03d}"}) for i, c in enumerate(base, start=1)]
    cand = [c.model_copy() for c in base]
    for i in range(45, 49):  # 4 general cases wrong: general drops 26.7 points, overall 6.7
        cand[i] = case_result(cand[i].case_id, "general", "billing")
    c = compare(base, cand)
    assert c.status is Status.FAIL
    assert any(r.startswith("general accuracy dropped") for r in c.reasons)


def test_improvements_are_listed():
    c = compare(cases(n_wrong=2), cases())
    assert len(c.improvements) == 2 and c.status is Status.PASS


def test_too_many_api_errors_make_the_run_invalid():
    cand = cases(n=54) + [case_result(f"case-{i:03d}", error=ErrorType.API_ERROR) for i in range(55, 61)]
    c = compare(cases(), cand)  # 10% errors is allowed; 7 would not be
    assert c.status is Status.PASS
    cand = cases(n=53) + [case_result(f"case-{i:03d}", error=ErrorType.API_ERROR) for i in range(54, 61)]
    assert compare(cases(), cand).status is Status.INVALID


def test_api_errors_are_not_regressions():
    cand = cases(n=59) + [case_result("case-060", error=ErrorType.API_ERROR)]
    c = compare(cases(), cand)
    assert c.regressions == [] and c.paired_cases == 59


def test_summary_score_drop_fails_and_lists_cases():
    c = compare(cases(), cases(score=0.5))
    assert c.status is Status.FAIL
    assert len(c.summary_regressions) == 60


def test_changed_judge_disables_summary_comparison():
    c = compare(cases(), cases(score=0.5), judge_version="j2")
    assert c.summary_delta is None and c.status is Status.PASS
    assert any("judge changed" in n for n in c.notes)


def test_dataset_change_compares_shared_cases_only():
    c = compare(cases(n=60), cases(n=62), dataset_hash="hash2")
    assert c.shared_cases == 60 and c.added_case_ids == ["case-061", "case-062"]
    assert any("dataset changed" in n for n in c.notes)


def test_served_model_change_is_noted():
    c = compare(cases(), cases(), served="gpt-4.1-mini-2026-01-01")
    assert any("served model changed" in n for n in c.notes)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) is None
    assert mcnemar_exact(1, 0) == 1.0
    assert mcnemar_exact(6, 0) == pytest.approx(2 / 64)
    assert mcnemar_exact(3, 3) == 1.0


# --- drift


def summary(i, accuracy, prompt="v1", infra=0.0, served="m1"):
    return RunSummary(run_id=f"r{i}", created_at=days(i), prompt_version=prompt, dataset_version=1, dataset_hash="h",
                      judge_version="j1", git_sha=None, accuracy=accuracy, summary_score=0.9,
                      infra_error_rate=infra, served_models=[served], cost_usd=0.05)


def test_slow_drift_is_detected_without_any_single_big_drop():
    # Each run drops about 1 point, never 3 at once, but the 7-run average falls about 7 points.
    history = [summary(i, 0.97 - 0.01 * i) for i in range(14)]
    report = detect_drift(history, SETTINGS.drift, T.max_infra_error_rate)
    assert report.status == "drift"


def test_stable_history_has_no_drift():
    history = [summary(i, 0.95 + (0.01 if i % 2 else -0.01)) for i in range(14)]
    assert detect_drift(history, SETTINGS.drift, T.max_infra_error_rate).status == "ok"


def test_drift_needs_two_windows_of_comparable_runs():
    history = [summary(i, 0.9) for i in range(10)] + [summary(10 + i, 0.5, prompt="v2") for i in range(4)]
    report = detect_drift(history, SETTINGS.drift, T.max_infra_error_rate)
    assert report.status == "insufficient_history" and report.comparable_runs == 4


def test_outage_runs_are_ignored_for_drift():
    history = [summary(i, 0.95) for i in range(14)] + [summary(20, 0.2, infra=0.5)]
    assert detect_drift(history, SETTINGS.drift, T.max_infra_error_rate).status == "ok"


def test_served_model_changes_are_reported():
    history = [summary(i, 0.95, served="m1" if i < 3 else "m2") for i in range(5)]
    assert detect_drift(history, SETTINGS.drift, T.max_infra_error_rate).served_model_changes


def test_history_round_trip(tmp_path):
    path = tmp_path / "history.jsonl"
    append_history(summarize(run_record(cases())), path)
    append_history(summarize(run_record(cases(n_wrong=6), run_id="later", created_at=days(1))), path)
    loaded = load_history(path)
    assert [s.run_id for s in loaded] == ["run", "later"] and loaded[1].accuracy == 0.9


# --- reports and notifications


def test_html_report_escapes_untrusted_text_and_shows_regressions():
    base, cand = run_record(cases(), run_id="base"), run_record(cases(n_wrong=6), run_id="cand", created_at=days(1))
    c = compare_runs(base, cand, T)
    html = render_html(c, base, cand, [summarize(base), summarize(cand)])
    assert "<b>bold</b>" not in html and "&lt;b&gt;bold&lt;/b&gt;" in html
    assert "FAIL" in html and "case-006" in html and "<svg" in html


def test_markdown_comment_has_marker_and_status():
    base, cand = run_record(cases(), run_id="base"), run_record(cases(n_wrong=1), run_id="cand")
    md = render_markdown(compare_runs(base, cand, T), base, cand, "https://example.com/run")
    assert md.startswith("<!-- regression-gate-report -->")
    assert "PASS" in md and "case-001" in md and "https://example.com/run" in md


def test_slack_is_skipped_without_webhook(monkeypatch, capsys):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    c = compare(cases(), cases(n_wrong=6))
    message = comparison_message(c, None)
    assert "FAIL" in message["text"] and "6 regressions" in message["text"]
    assert send(message) is False
    assert "not sent" in capsys.readouterr().out


# --- CLI exit codes


@pytest.mark.parametrize("n_wrong, code", [(0, 0), (3, 0), (6, 1)])
def test_compare_exit_codes(tmp_path, n_wrong, code):
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    base.write_text(run_record(cases(), run_id="base").model_dump_json())
    cand.write_text(run_record(cases(n_wrong=n_wrong), run_id="cand").model_dump_json())
    report, comment = tmp_path / "r.html", tmp_path / "c.md"
    args = ["--db", str(tmp_path / "db.sqlite"), "compare", str(base), str(cand), "--report", str(report), "--comment", str(comment)]
    assert main(args) == code
    assert report.exists() and comment.exists()


def test_invalid_exit_code(tmp_path):
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    base.write_text(run_record(cases(), run_id="base").model_dump_json())
    errors = [case_result(f"case-{i:03d}", error=ErrorType.API_ERROR) for i in range(1, 61)]
    cand.write_text(run_record(errors, run_id="cand").model_dump_json())
    assert main(["--db", str(tmp_path / "db.sqlite"), "compare", str(base), str(cand)]) == 2
