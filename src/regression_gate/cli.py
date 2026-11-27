"""Command-line entry point: `uv run gate <command>`.

Exit codes for `compare` and `check`: 0 = PASS or WARN, 1 = FAIL, 2 = INVALID.
CI treats anything non-zero as a failed check, which blocks the merge.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from regression_gate.compare import Comparison, Status, compare_runs
from regression_gate.dataset import load_dataset
from regression_gate.drift import HISTORY_PATH, append_history, detect_drift, load_history, summarize
from regression_gate.llm import make_client
from regression_gate.notify import comparison_message, drift_message, send
from regression_gate.prompts import load_prompt, prompt_path
from regression_gate.records import RunRecord
from regression_gate.report import headline, render_html, render_markdown
from regression_gate.runner import run_eval
from regression_gate.settings import SETTINGS_PATH, load_settings
from regression_gate.storage import DB_PATH, Store

EXIT_CODES = {Status.PASS: 0, Status.WARN: 0, Status.FAIL: 1, Status.INVALID: 2}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="gate", description="LLM regression gate")
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="evaluate a prompt version against the golden dataset")
    p.add_argument("--prompt", help="prompt version, default: active_prompt in gate.yaml")
    p.add_argument("--out", type=Path, help="also write the full run record as JSON")
    p.add_argument("--limit", type=int, help="only the first N cases (quick local checks)")

    p = sub.add_parser("compare", help="compare two runs and decide PASS / WARN / FAIL / INVALID")
    p.add_argument("baseline", help="run JSON file or run id in the database")
    p.add_argument("candidate", help="run JSON file or run id in the database")
    _add_output_args(p)

    p = sub.add_parser("check", help="run two prompt versions now and compare them")
    p.add_argument("--baseline-prompt", required=True)
    p.add_argument("--candidate-prompt", required=True)
    p.add_argument("--limit", type=int)
    _add_output_args(p)

    p = sub.add_parser("record", help="append a run's summary to the history file")
    p.add_argument("run", help="run JSON file or run id in the database")
    p.add_argument("--history", type=Path, default=HISTORY_PATH)

    p = sub.add_parser("drift", help="check the history for slow drift (exit 1 if drifting)")
    p.add_argument("--history", type=Path, default=HISTORY_PATH)
    p.add_argument("--slack", action="store_true")
    p.add_argument("--report-url")

    p = sub.add_parser("runs", help="list stored runs")
    p.add_argument("--prompt")

    args = parser.parse_args(argv)
    settings = load_settings(args.settings)
    store = Store(args.db)

    if args.command == "run":
        record = asyncio.run(_run(args.prompt or settings.active_prompt, settings, store, args.limit))
        if args.out:
            _write(args.out, record.model_dump_json(indent=1))
        return 0

    if args.command == "compare":
        baseline, candidate = _load_run(args.baseline, store), _load_run(args.candidate, store)
        return _compare_and_report(baseline, candidate, settings, store, args)

    if args.command == "check":
        baseline = asyncio.run(_run(args.baseline_prompt, settings, store, args.limit))
        candidate = asyncio.run(_run(args.candidate_prompt, settings, store, args.limit))
        return _compare_and_report(baseline, candidate, settings, store, args)

    if args.command == "record":
        append_history(summarize(_load_run(args.run, store)), args.history)
        print(f"appended to {args.history}")
        return 0

    if args.command == "drift":
        report = detect_drift(load_history(args.history), settings.drift, settings.thresholds.max_infra_error_rate)
        print(f"drift: {report.status}: {report.message}")
        for change in report.served_model_changes:
            print(f"served model changed: {change}")
        if report.status == "drift" and args.slack:
            send(drift_message(report, args.report_url))
        return 1 if report.status == "drift" else 0

    if args.command == "runs":
        for r in store.list_runs(args.prompt):
            acc = "n/a" if r["accuracy"] is None else f"{r['accuracy']:.1%}"
            summ = "n/a" if r["summary_score"] is None else f"{r['summary_score']:.1%}"
            print(f"{r['run_id']}  dataset v{r['dataset_version']}  accuracy {acc}  summary {summ}")
        return 0
    return 2


def _add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--report", type=Path, help="write the HTML diff report here")
    p.add_argument("--comment", type=Path, help="write the Markdown PR comment here")
    p.add_argument("--json", type=Path, help="write the comparison as JSON here")
    p.add_argument("--history", type=Path, help="history file for the trend chart (default: runs in the database)")
    p.add_argument("--report-url", help="link to the full report, used in the comment and Slack")
    p.add_argument("--slack", action="store_true", help="send the result to Slack (needs SLACK_WEBHOOK_URL)")


async def _run(version: str, settings, store: Store, limit: int | None) -> RunRecord:
    path = prompt_path(version)
    config = load_prompt(path)
    dataset = load_dataset(settings.dataset)
    if limit:
        dataset = dataset.model_copy(update={"cases": dataset.cases[:limit]})

    total, done = len(dataset.cases), 0

    interactive = sys.stderr.isatty()

    def progress(_result) -> None:
        nonlocal done
        done += 1
        # Overwrite one line in a terminal; in CI logs, print every 10 cases instead.
        if interactive:
            print(f"\r{version}: {done}/{total} cases", end="", file=sys.stderr, flush=True)
        elif done % 10 == 0 or done == total:
            print(f"{version}: {done}/{total} cases", file=sys.stderr, flush=True)

    record = await run_eval(config, path, dataset, settings, make_client(), store, progress)
    if interactive:
        print(file=sys.stderr)
    store.save_run(record)
    _print_run(record)
    return record


def _compare_and_report(baseline: RunRecord, candidate: RunRecord, settings, store: Store, args) -> int:
    comparison = compare_runs(baseline, candidate, settings.thresholds)
    history = _trend_history(args.history, store, [baseline, candidate])

    _print_comparison(comparison)
    if args.report:
        _write(args.report, render_html(comparison, baseline, candidate, history))
    if args.comment:
        _write(args.comment, render_markdown(comparison, baseline, candidate, args.report_url))
    if args.json:
        _write(args.json, comparison.model_dump_json(indent=1))
    if args.slack:
        send(comparison_message(comparison, args.report_url))
    return EXIT_CODES[comparison.status]


def _trend_history(path: Path | None, store: Store, current: list[RunRecord]):
    if path:
        history = load_history(path)
    else:
        history = [summarize(store.load_run(r["run_id"])) for r in store.list_runs()]
    known = {h.run_id for h in history}
    history += [summarize(r) for r in current if r.run_id not in known]
    return sorted(history, key=lambda s: s.created_at)[-20:]


def _load_run(ref: str, store: Store) -> RunRecord:
    path = Path(ref)
    if path.suffix == ".json" and path.exists():
        return RunRecord.model_validate_json(path.read_text())
    return store.load_run(ref)


def _print_run(r: RunRecord) -> None:
    m = r.metrics
    pct = lambda v: "n/a" if v is None else f"{v:.1%}"  # noqa: E731
    cost = sum(c for c in (m.classifier_cost_usd, m.judge_cost_usd) if c is not None)
    print(
        f"{r.run_id}: accuracy {pct(m.accuracy)} ({m.evaluated}/{m.cases} evaluated), summary {pct(m.summary_score)}, "
        f"API errors {pct(m.infra_error_rate)}, invalid {pct(m.invalid_output_rate)}, "
        f"p95 {m.latency_p95_ms or 0:.0f} ms, cost ${cost:.4f}, served {', '.join(m.served_models)}"
    )
    for name, g in m.by_category.items():
        print(f"  {name:10} {g.correct}/{g.total}")


def _print_comparison(c: Comparison) -> None:
    print(f"\n{c.status.value.upper()}: {c.baseline_prompt} -> {c.candidate_prompt}: {headline(c)}")
    for reason in c.reasons:
        print(f"  - {reason}")
    for note in c.notes:
        print(f"  note: {note}")
    for d in c.regressions:
        print(f"  regressed {d.case_id}: expected {d.expected_category}, {d.baseline_category} -> {d.candidate_category or d.candidate_error}")
    for d in c.improvements:
        print(f"  improved  {d.case_id}: expected {d.expected_category}, {d.baseline_category or d.baseline_error} -> {d.candidate_category}")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
