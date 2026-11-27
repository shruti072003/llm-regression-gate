# LLM regression gate

[![tests](https://github.com/shruti072003/llm-regression-gate/actions/workflows/tests.yml/badge.svg)](https://github.com/shruti072003/llm-regression-gate/actions/workflows/tests.yml)

CI for model behavior. Every pull request that touches a prompt, the classifier code, the golden dataset or the gate's settings is evaluated against a hand-labeled golden dataset, compared case by case with `main`, and blocked if quality regresses. The result is posted as a PR comment with a full HTML diff report. A nightly run of the production prompt watches for slow drift, such as the provider changing the model behind an alias.

The feature under test is a customer-support email classifier: `gpt-4.1-mini` returns a category (`billing`, `technical`, `account`, `general`) and a one-sentence summary.

## How it works

```
PR changes prompts/, src/, data/ or gate.yaml
        │
        ├── baseline:  main's code + prompt + dataset ──┐
        └── candidate: the PR's code + prompt + dataset ─┤  same job, same served model, shared judge cache
                                                         ▼
        each of the 60 golden cases:
          classify (gpt-4.1-mini, structured outputs)
            → category correct?                 exact match, the case's pass/fail
            → summary covers the key facts?     gpt-4.1 judge, cached by summary
            → latency, tokens, cost, served model snapshot
                                                         ▼
        paired comparison on cases evaluated in both runs
          regressions, improvements, per-category and per-tag deltas, McNemar on flips
                                                         ▼
        PASS / WARN / FAIL / INVALID ──► PR comment · HTML report artifact · Slack (optional) · check fails on FAIL/INVALID
```

## Results on the current golden set

| Prompt | Change | Accuracy | Summary score | Gate vs previous |
|---|---|---:|---:|---|
| v1 | zero-shot baseline | 95.0% | 96.4% | — |
| v2 | + labeling rules | 100% | 99.2% | PASS, 3 improvements |
| v3 | + 4 few-shot examples | 100% | 98.3% | PASS, no gain, +33% cost per call |
| v4 | 6-word summaries + keyword shortcut | 98.3% | 71.1% | **FAIL**: summary score −28.1 points |

v4 is the interesting one. Its accuracy drop (−1.7 points, one case) is within noise, so a gate that only checked categories would have let it ship. It's caught because summaries are scored separately, against key facts, by a stronger judge model.

Example reports: [v1 → v2](docs/examples/v1-vs-v2.html) and [v2 → v4](docs/examples/v2-vs-v4.html) (download and open in a browser).

**Caveat:** the 60 golden cases were drafted by an LLM and then human-reviewed against the written labeling policy. v2 reaching 100% is a ceiling: the rules it encodes were written alongside the dataset. See [Limitations](#limitations). The example reports were generated before the review, so they still list the cases as unreviewed.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
echo 'OPENAI_API_KEY=sk-...' > .env      # never committed
uv run pytest                            # 87 tests, no API calls
```

A full run of 60 cases costs about $0.05–0.10 (classifier plus judge) at the prices in `gate.yaml`.

## Everyday commands

```sh
uv run gate run                                   # evaluate the active prompt (gate.yaml)
uv run gate run --prompt v3 --out results/v3.json # a specific version, also saved as JSON
uv run gate run --limit 5                         # quick check on the first 5 cases
uv run gate check --baseline-prompt v2 --candidate-prompt v3 --report results/report.html
uv run gate compare results/a.json results/b.json --report results/report.html
uv run gate runs                                  # runs stored in results/gate.db
uv run python scripts/smoke_test.py --version v3  # 6 emails x 5 runs: how stable is the output?
uv run python scripts/dataset_stats.py            # balance of the golden set
```

`compare` and `check` exit with 0 for PASS or WARN, 1 for FAIL, 2 for INVALID.

With Docker:

```sh
docker build -t llm-regression-gate .
docker run --rm -e OPENAI_API_KEY llm-regression-gate run --prompt v2
docker run --rm -e OPENAI_API_KEY -v "$PWD/my-gate.yaml:/app/custom.yaml" -e GATE_SETTINGS=custom.yaml llm-regression-gate run
```

## Shipping a prompt change

1. Copy the current prompt to a new version: `prompts/v5.yaml`, with `version_id: v5`. The loader rejects a file whose `version_id` doesn't match its filename, which catches copy-and-forget.
2. Edit the prompt, and say what changed and why in `description`.
3. Set `active_prompt: v5` in `gate.yaml`.
4. Open a PR. The eval gate runs v4 (main) against v5 (your branch) and comments with the result.

Never edit a prompt file after it's merged. Old results refer to it by version and hash; a new idea is a new version. A model change is a prompt change too: `model` and `temperature` live in the prompt file, so switching models goes through the same gate.

## Adding golden cases

Cases live in [data/golden.yaml](data/golden.yaml); the format and rules are in the comments at the top of the file. In short:

- Write the email as a customer would. Don't use the category name in it.
- Label it using the labeling policy in [DECISIONS.md](DECISIONS.md#prompt-files-prompts). If no rule covers it, write the rule there first.
- 1–3 `key_facts`: what a correct one-sentence summary must mention (the problem and the request, not dates or order numbers).
- `difficulty`, edge-case `tags`, and `notes` for hard or tagged cases.
- Use the next unused `id`. IDs are never reused or renumbered.
- Bump `version`. Each run also records a content hash, so a forgotten bump is still detectable.

The best new cases come from real failures: when the classifier gets something wrong in production, add that email. Tests check that no golden email duplicates a few-shot example or the LLM-written smoke-test emails (that would be data leakage).

## Adjusting thresholds

All thresholds are in [gate.yaml](gate.yaml), and changing them triggers the eval like any other PR.

| Setting | Default | Meaning |
|---|---|---|
| `accuracy_warn_drop` / `accuracy_critical_drop` | 0.03 / 0.08 | overall accuracy drop that warns / fails |
| `category_warn_drop` / `category_critical_drop` | 0.10 / 0.20 | the same per category (15 cases each) |
| `summary_warn_drop` / `summary_critical_drop` | 0.05 / 0.10 | mean key-fact coverage drop |
| `max_infra_error_rate` | 0.10 | above this share of API errors, the result is INVALID |
| `drift.window` | 7 | slow drift compares the last 7 comparable runs with the 7 before |

With 60 cases one flip is 1.7 points. A single case flipped between two identical v1 runs (a boundary case at temperature 0), so thresholds tighter than a couple of flips would make the gate flaky.

## CI

| Workflow | When | What |
|---|---|---|
| [tests.yml](.github/workflows/tests.yml) | every push and PR | unit tests, Docker build and smoke test; no API calls |
| [eval-gate.yml](.github/workflows/eval-gate.yml) | every PR (passes in seconds if nothing model-affecting changed) | baseline vs candidate, PR comment, report artifact, fails on FAIL/INVALID |
| [nightly.yml](.github/workflows/nightly.yml) | daily 06:17 UTC, or manually | runs the production prompt, appends to `results/history.jsonl` on the [`eval-history`](../../tree/eval-history) branch, checks slow drift |

Secrets: `OPENAI_API_KEY` (required), `SLACK_WEBHOOK_URL` (optional; without it alerts are printed in the job log).

`main` is protected: a PR can merge only when `gate`, `pytest` and `docker` pass and the branch is up to date with `main`. Up to date matters here because the gate's baseline is main at the time it ran; if main's prompt changed since, the comparison is stale.

## Architecture decisions

Every decision, with the alternatives and the reasoning, is in [DECISIONS.md](DECISIONS.md). The ones that shape the system most:

- **Structured outputs, validated after generation.** The API enforces the schema (keys, category enum) but not length. Enforcing `maxLength` while generating cut summaries off mid-sentence, and the broken summaries still passed validation.
- **Category and summary are scored separately.** Category is binary and stable; summary quality is graded by an LLM and noisier. v4 shows why both are needed.
- **Key-fact checklists, not "rate similarity 1–5".** The judge says which fact is missing, and scores are more consistent.
- **Judge verdicts are cached by summary text.** An identical summary always gets an identical verdict, so judge noise can't create a regression.
- **The baseline is re-run in the same CI job** instead of reusing a stored result. Both runs see the same served model snapshot and dataset, so only the PR's change differs.
- **API errors are not regressions.** They're excluded from quality metrics; too many make the result INVALID.
- **Slow drift is tracked separately from per-run checks.** A model that loses a point a week never fails a single comparison.
- **Everything that can change results is recorded:** prompt hash, served model snapshot, dataset hash, judge version and git commit.

## Repository layout

```
prompts/v*.yaml          versioned prompts (model and temperature included)
data/golden.yaml         golden dataset
gate.yaml                thresholds, judge, pricing, active prompt
src/regression_gate/
  schemas.py             output contract (category enum, summary)
  prompts.py             prompt loader (strict, versioned)
  dataset.py             golden dataset loader and validation
  llm.py                 OpenAI calls: retries, structured outputs
  classifier.py          the feature under test
  judge.py               LLM-as-judge for summaries
  runner.py metrics.py   run the dataset, aggregate metrics
  compare.py             PASS/WARN/FAIL/INVALID decision
  drift.py               history and slow-drift detection
  report.py notify.py    HTML report, PR comment, Slack
  storage.py             SQLite runs and judge cache
  cli.py                 the `gate` command
scripts/                 smoke test, dataset stats
results/                local run database and reports (not committed)
```

## Limitations

- **The golden set is LLM-drafted.** Every label was human-reviewed against the policy, but the emails were written by a model, which can make them cleaner than real customer email.
- **It's small.** 60 cases means 1.7 points per case; fine for catching big regressions, not for telling 97% from 98%.
- **v2 is partly fitted to the dataset.** The labeling rules were written while drafting the cases. A held-out split that prompt iterations never look at would give an honest estimate.
- **One run per prompt.** Boundary cases can flip between identical runs. Running each case several times and voting would reduce noise at a proportional cost.
- **The judge is unvalidated against human ratings.** It passed four hand-made checks; agreement with human scores on a sample would be the next step.
- **Pricing in `gate.yaml` is from launch** and should be checked against OpenAI's current pricing.
