"""SQLite store for eval runs and cached judge verdicts."""

import json
import sqlite3
from pathlib import Path

from regression_gate.judge import JudgeOutcome
from regression_gate.records import RunRecord

DB_PATH = Path("results/gate.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    dataset_version INTEGER NOT NULL,
    dataset_hash TEXT NOT NULL,
    git_sha TEXT,
    accuracy REAL,
    summary_score REAL,
    record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_prompt ON runs (prompt_version, created_at);
CREATE TABLE IF NOT EXISTS judge_cache (
    key TEXT PRIMARY KEY,
    outcome_json TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: Path = DB_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def save_run(self, record: RunRecord) -> None:
        m = record.metrics
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.run_id,
                    record.created_at.isoformat(),
                    record.prompt_version,
                    record.model,
                    record.dataset_version,
                    record.dataset_hash,
                    record.git_sha,
                    m.accuracy,
                    m.summary_score,
                    record.model_dump_json(),
                ),
            )

    def load_run(self, run_id: str) -> RunRecord:
        row = self.conn.execute("SELECT record_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no run {run_id!r}")
        return RunRecord.model_validate_json(row[0])

    def list_runs(self, prompt_version: str | None = None) -> list[dict]:
        query = "SELECT run_id, created_at, prompt_version, dataset_version, accuracy, summary_score FROM runs"
        params: tuple = ()
        if prompt_version:
            query += " WHERE prompt_version = ?"
            params = (prompt_version,)
        rows = self.conn.execute(query + " ORDER BY created_at", params).fetchall()
        keys = ["run_id", "created_at", "prompt_version", "dataset_version", "accuracy", "summary_score"]
        return [dict(zip(keys, row)) for row in rows]

    # Judge verdicts are cached by (judge version, model, email, facts, summary). An identical
    # summary always gets the identical verdict, so judge noise can't create a fake regression
    # between two runs that produced the same summary.
    def get_judge(self, key: str) -> JudgeOutcome | None:
        row = self.conn.execute("SELECT outcome_json FROM judge_cache WHERE key = ?", (key,)).fetchone()
        return JudgeOutcome.model_validate(json.loads(row[0])) if row else None

    def put_judge(self, key: str, outcome: JudgeOutcome) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO judge_cache VALUES (?, ?)", (key, outcome.model_dump_json())
            )
