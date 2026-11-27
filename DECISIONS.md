# Decisions

One entry per design decision: what was chosen, what else was considered, and why.

## Setup

**uv for environments and dependencies** (instead of pip + venv or Poetry)
One tool manages the Python version, the venv, dependencies and the lockfile. `uv add` records each dependency in `pyproject.toml` and pins exact versions in `uv.lock`, so CI, Docker and my laptop install identical packages. That matters here because an SDK upgrade can change model behavior.

**Python 3.12** (instead of 3.14, which was installed)
Meets the 3.11+ requirement and has wider library support than the newest release. Pinned in `.python-version`.

**OpenAI as the provider**
The BASWE guide uses OpenAI. The API call sits behind a single classifier function, so switching providers later means changing one module.

**src layout, with the module named `regression_gate`** (project name `llm-regression-gate`)
The src layout forces tests to import the installed package, the same way CI and Docker will, instead of whatever is in the current folder. `uv_build` derives `llm_regression_gate` from the project name by default, so `module-name` is set explicitly under `[tool.uv.build-backend]`. The project name is what gets installed; the module name is what gets imported.

## Output schema (`schemas.py`)

**Category is an Enum, not a plain string**
The eval needs a closed set of labels. With a plain `str`, a made-up label like "payments" would be accepted and silently scored. With an Enum, it fails validation and is counted as a failure.

**`StrEnum` instead of `(str, Enum)`**
Both validate the same way, but `str(Category.BILLING)` gives `"billing"` with `StrEnum` and `"Category.BILLING"` with `(str, Enum)`. Categories will be written to SQLite, reports and Slack, so the plain value is what I want.

**Summary must be 3–200 characters after stripping whitespace, checked locally only** *(revised in Step 1.5; was 100)*
The minimum rejects empty or whitespace-only output (`str_strip_whitespace=True`). The maximum is a guard against essays, not a style rule; conciseness is judged by the LLM judge in Step 4. Raised from 100 because natural one-sentence summaries ran 73–109 characters in the smoke test. See "Length limits are not sent to the API" below for why it's local-only.

**Category casing and whitespace are normalized before validation** (instead of strict matching)
The model returning "Billing" for a billing email understood it correctly; only the formatting is off. Strict validation would count that as a wrong answer, mixing "wrong category" with "wrong format" in one number. Because LLM output varies even at temperature 0, it could also make CI fail at random. A `mode="before"` validator lowercases and strips the value, so accuracy measures understanding.
Trade-off: normalization hides formatting drift. To keep that signal, the classifier (Step 1.4) should record whether the raw value had to be normalized and report a separate format-compliance rate. Structured outputs in 1.4 should also prevent the problem at the API level; Pydantic stays as defense in depth.

## Prompt files (`prompts/`)

**Model and temperature live in the prompt file, next to the prompt text**
The output depends on the prompt, the model and the parameters together, so they are versioned as one unit. Consequence: a model change is a prompt change. Swapping models means creating a new prompt version, and it goes through the same regression gate as a wording change. Nobody can swap the model in code without it being tested.

**Versions are `v1`, `v2`, … (not semantic versioning like `1.0.0`)**
Semver promises compatibility levels ("minor won't break you") that a prompt change can't guarantee; one changed word can move accuracy by 10%. The version only records order, and the eval results say how big a change was. Also avoids a YAML trap: unquoted `1.0` loads as a float.

**`version_id` is in the file and must match the filename**
The file describes itself even if moved, and the loader raises an error if `v2.yaml` says `version_id: v1`. That catches the copy-paste bug where every result gets stored under the wrong version.

**`created_at` is a plain date, `temperature` is 0**
A date is enough because versions are ordered by number, not time. Temperature 0 minimizes variation between runs so that differences point to the prompt, not randomness (Step 1.5 tests how deterministic it actually is).

**v1 is zero-shot (`few_shot: []`)**
v1 is the baseline. v2 adds examples, which makes it a controlled experiment where only one thing changes, and the gate measures exactly what few-shot gained. If v1 already scores very high, that's a finding too: examples may not be worth their token cost.
Rule for later: few-shot emails must never appear in the golden dataset (data leakage inflates accuracy).

**Labeling policy (the prompt and the golden dataset must follow the same rules)**
- Login and password problems are **account** (the customer's access or identity). An error page shown during login is **technical** (the product is broken).
- *Chosen by Claude as defaults; confirm or change:* plan upgrades and downgrades are **billing**; closing an account is **account**; an email covering several topics gets the category of the main thing the customer needs resolved.
- **Paid but not delivered is technical** *(Step 1.5)*: "I upgraded to Pro and paid, but the features aren't showing." The payment went through; the product didn't deliver, so the thing to resolve is technical. v1 says billing in 5/5 runs, a consistent failure for v2 to fix. Rule followed: never change a label because the model disagrees; decide what's right, then fix the prompt. Prompt versions are never edited after commit, so this goes into v2, not v1.
- *Rules added while drafting the golden set (Step 2), each with a contrasting pair of cases:*
  - **Cancelling a paid subscription but keeping the account is billing; closing the account is account** (case-010 vs case-042).
  - **Pre-purchase questions about prices, plans or discounts are general** (sales inquiries); billing is money on an existing subscription (case-060, case-051 vs case-014). v1's prompt lists "prices" under billing, so v1 is expected to miss these.
  - **A failed payment that locks features is billing,** not "paid but not delivered", because nothing was paid (case-009 vs case-022).
  - **Credentials rejected is account; an error page during login is technical** (case-037 vs case-021).
  - **Product notification emails not arriving is technical; password-reset or verification emails are account** (case-025 vs case-036, case-038).
  - **Content shared inside the product not showing up is technical;** account means access to your own account, identity, workspace roles and ownership (case-030 vs case-035, case-041).
  - **Design feedback where nothing is broken is general** (case-056).
  - **Workspace-wide security settings such as requiring 2FA are account,** even when phrased as a how-to question (case-044).

## Prompt loader (`prompts.py`)

**Unknown fields are rejected (`extra="forbid"`)** (instead of Pydantic's default of ignoring them)
A typo like `temprature: 0.7` would otherwise be dropped silently, and I'd believe I tested at 0.7 when I didn't. Cost: every new field must be added to `PromptConfig` before it can be used, which is the point.

**No defaults: every field is required**
The prompt file is the complete record of what ran. With defaults in code, the real config would be split between YAML and Python, and changing a default would quietly change how old prompt versions behave, breaking historical comparisons. Together with `extra="forbid"`, a typo produces two errors (one missing field, one unknown field), so the cause is obvious.

**Few-shot expected outputs use the same `EmailClassification` schema as model answers**
One source of truth for the output format. An example can never teach the model a format my own validator would reject; a bad example fails when the file loads, not as unexplained bad output later.

**Defaults chosen by Claude:**
- Any invalid file raises `PromptConfigError` with the file path in the message. In CI a run must fail loudly, not continue with a half-loaded config.
- `temperature` must be 0–2 (OpenAI's range), and `version_id` must look like `v<number>`.
- The config is frozen after loading, and `few_shot` is a tuple, because `frozen` only blocks reassigning fields and a list could still be appended to. Results stay tied to exactly the file they came from.
- The loader takes a path; `prompt_path("v1")` resolves a version to `prompts/v1.yaml`.
- pytest is a dev dependency. Each decision above has a test, so it's enforced in CI and not just documented.

## Classifier (`classifier.py`)

**Structured outputs: the API enforces the `EmailClassification` schema** (instead of JSON mode or prompt instructions only)
An eval should test what I'd actually ship, and in production I'd use structured outputs. Generation is constrained to the schema, so invalid JSON and made-up categories become nearly impossible, and parse failures don't add noise to accuracy. Pydantic still validates locally, as defense in depth.
Implementation detail: `responses.parse()` raises an exception on invalid output and discards the response, losing the raw text and token counts. So I call `responses.create()` with a schema built from `EmailClassification.model_json_schema()` and validate it myself. Both come from one source (`EmailClassification`), but deliberately differ in one way: see the next entry.

**Length limits are not sent to the API; they're checked locally** *(Step 1.5)*
Found in the smoke test: with `maxLength=100` in the structured-output schema, the API enforced the limit while generating. Summaries were cut mid-sentence ("…despite using the correct"), one ended in a Chinese word the model was forced into, and all of them still passed validation. That's an invisible failure. Now the API schema keeps structure (keys, category enum) and drops `minLength`/`maxLength`, and the local validator checks length, so a too-long summary is a visible `invalid_output`.
Side effect: the squeezed email went from 4 different summaries in 5 runs to 1. The constraint was causing part of the variation I'd have blamed on non-determinism.
Lesson: a constraint enforced during generation changes what the model writes, not just what gets accepted. Validate after generation when you need to see the violation.

**`classify()` always returns a `ClassificationOutcome`, never raises** (instead of exceptions or `None`)
One bad case can't crash an 80-case run, and every failure keeps its evidence for the diff report: raw output, error type and message, latency, tokens, attempts, served model.

**API errors are infrastructure errors, kept out of quality metrics**
`ErrorType` separates quality failures (`invalid_output`, `refusal`, `truncated`: the model gave a bad answer) from `api_error` (no answer to judge). An OpenAI outage during CI must not look like "accuracy dropped 15%" and block a good PR. If too many cases hit `api_error`, the run will be marked invalid instead of passing or failing (Step 5).

**Defaults chosen by Claude:**
- **Async from the start,** because the Step 3 runner sends requests concurrently.
- **Own retry loop, SDK retries off:** up to 4 attempts for connection errors, timeouts, 429s and 5xx, with exponential backoff and jitter so parallel requests don't retry in lockstep. Non-transient errors (401 auth, 400 bad request) fail immediately, since retrying can't help. My own loop can record `attempts`, and `latency_ms` measures only the successful attempt, so retry waits don't pollute the latency metric.
- **30-second timeout** (the SDK default read timeout is 600 s, so one stuck call could hold CI for 10 minutes).
- **`max_output_tokens=200`** (a one-sentence answer is about 30 tokens). The cap limits cost; hitting it is recorded as `truncated`.
- **Record `served_model`,** the snapshot that answered (`gpt-4.1-mini-2025-04-14`), not only the alias requested. If scores change with no prompt change, this shows whether the provider updated the model behind the alias.
- **Record `category_normalized`** by comparing the raw category to the validated one. That's the format-compliance signal from the casing decision. With structured outputs it should almost always be false; if it isn't, that's worth investigating.
- **The client is passed in,** not created inside `classify()`. One client is shared across the run, and tests substitute a fake one, so they run without network or cost.

## Smoke test findings (Step 1.5, v1, 6 emails × 5 runs, after the length fix)

**Temperature 0 is not fully deterministic** (prediction before measuring: "mostly identical, some variation", which was correct)
- Category: identical across all 5 runs for 6/6 emails. A category flip in CI is a real signal.
- Summary: word-for-word identical for 4/6 emails; the rest differ in wording ("on Windows 11 version 4.2" vs. "on version 4.2 with Windows 11"). Exact-match comparison of summaries would report false regressions every run, so summaries must be scored on meaning (LLM judge, Step 4).
- The emails in `scripts/smoke_test.py` are LLM-written and must never go into the golden dataset.

**Changing classifier code changes outputs, just like changing a prompt**
The length fix changed v1's results without touching `prompts/v1.yaml`. So the CI gate (Step 8) must run when the classifier or schema changes, not only when `prompts/` changes.

## Golden dataset (`data/golden.yaml`, `dataset.py`)

**The 60 cases were drafted by Claude, then human-reviewed** *(revised twice; the original decision was that I write every case)*
Original reasoning still holds: the dataset is the ground truth, LLM-written cases test the model on its own kind of writing, and label errors go unnoticed. I changed course to finish the pipeline end to end, so provenance is tracked: every case has a required `reviewed` flag, new cases start as `false`, and reports show how many are unreviewed. On 2026-11-27 I reviewed all 60 labels against the labeling policy, changed none, and set them to `reviewed: true`. In an interview: "LLM-drafted, human-reviewed against a written labeling policy."

**YAML** (the guide says JSON; I first chose JSON and switched)
I'll hand-write about 60 multi-line emails and edit them repeatedly. In JSON every line break is `\n`, every quote is escaped, and comments are impossible, so malformed cases are easy to create and hard to spot. YAML lets me write emails as plain text and explain tricky labels in comments. The guide's reason for JSON, "git-friendly", applies equally to YAML. The loader validates every field, which covers YAML's type surprises. Lesson: "the guide says so" isn't a reason; the guide's reason has to apply to my situation.

**Expected summary = 1–3 key facts, not one ideal sentence** (differs from the guide)
The smoke test showed the same correct meaning can be worded many ways. A judge checking a short checklist ("charged twice", "wants a refund") gives more consistent scores than one rating similarity to a reference sentence, and facts are quicker to write.

**Balanced: about 15 cases per category** (not a realistic traffic mix)
With a realistic mix, `general` would get about 6 cases, and one flip would move its accuracy by 17 points. Balanced gives every category enough cases for its own accuracy to mean something. I have no real traffic to match anyway.

**Difficulty level plus edge-case tags**
`difficulty` (easy/medium/hard) plus a closed set of tags (`ambiguous`, `policy_boundary`, `multi_topic`, `very_short`, `typos`, `mixed_language`, `sarcasm`). The report can say "sarcasm accuracy dropped from 4/4 to 1/4" instead of only "hard cases dropped".

**Defaults chosen by Claude:**
- **Strict validation for my data, lenient for model output.** `expected_category: Billing` in my file is a typo and fails; the same value from the model is normalized and counted. Different sources call for different strictness.
- **Neutral, permanent IDs** (`case-001`). An ID like `billing-001` would be misleading after a relabel. IDs are never renumbered or reused, since Step 5 compares runs case by case.
- **One file with a `version` number,** plus a `content_hash` computed at load time (and rejected if written by hand). Bumping the version is for humans; the hash catches a forgotten bump.
- **The hash covers only what affects scores** (IDs, emails, labels, key facts) *(revised when marking cases reviewed)*. It originally hashed the whole file, so flipping 60 `reviewed` flags would have flagged "dataset changed" in every comparison and reset drift history, with no change to scoring. Notes, tags, difficulty, review flags and comments can now change freely. (The first nightly history entry still has an old whole-file hash, so drift starts counting from the next run.)
- **`notes` required only for hard or tagged cases.** Easy cases explain themselves; tricky ones need the reasoning written down.
- **Duplicates rejected after normalizing** case and whitespace, so a reformatted copy still counts as a duplicate.
- **Leakage is checked by tests on the real files,** not inside the loader. Tests compare the golden emails against every prompt's few-shot examples and the LLM-written smoke emails, so a leak fails CI. Keeping it out of the loader avoids making the dataset loader depend on prompts and scripts.

*Key facts revised after the first v1 run (Step 3):* five facts were details a one-sentence summary may reasonably leave out ("since Monday", "$480", "200 people", "password login still works", "after failed logins"). That breaks the written rule (facts are the problem and the request), so I fixed them before freezing the baseline. Done after seeing model output, which is exactly when to be careful: each change was checked against the rule, not against what the model wrote. Facts the model missed that do matter (the refund request in case-026, "invoice shows full price" in case-014) stayed.

## Runner, scoring and storage (Steps 3–4)

**Settings in `gate.yaml`:** thresholds, judge model, concurrency and pricing live in one validated file (unknown keys rejected, like prompts). Changing it changes what the gate tolerates, so it's reviewed like code and triggers the CI eval.

**Concurrency with a semaphore (8 requests at a time):** a 60-case run takes about 15 s instead of a minute, without hitting rate limits. Combined with jittered retries from Step 1.4.

**A case "passes" on category alone; summaries are a separate score**
Category correctness is binary and stable (no flips in 30 smoke runs on clear emails). Summary quality is graded and judged by an LLM, so it's noisier. Mixing them into one pass/fail would let judge noise flip cases. Each gets its own metric and thresholds.

**LLM judge: `gpt-4.1`, temperature 0, checks each key fact plus "unsupported claim"**
- Stronger than the classifier (`gpt-4.1-mini`), so it can catch what the classifier gets wrong. `gpt-5` models were rejected: they're reasoning models with no temperature setting and spent the whole 16-token test budget thinking.
- Checklist scoring (fraction of key facts covered) rather than "rate similarity 1–5": more consistent, and the reasoning says which fact was missed.
- The output schema puts `reasoning` before the verdicts, so the model reasons before committing.
- Verified on four hand-made summaries (good, missing fact, invented detail, truncated): all scored correctly.
- `JUDGE_VERSION` (`j1`) is recorded on every run. Scores from different judge versions aren't comparable, and the comparison refuses to compare them.

**Judge verdicts are cached** by (judge version, model, email, facts, summary)
An identical summary always gets the identical verdict, so judge noise can't create a fake summary regression between two runs that wrote the same summary. It also halves the cost of a rerun. In CI both runs share one database, so they share the cache.

**Quality failures score 0, API errors are excluded:** an invalid output or refusal is a wrong category and a 0 summary. An API error has no answer to judge, so it's left out of both metrics and counted in `infra_error_rate`.

**Storage: SQLite (`results/gate.db`, not committed) plus JSON run files.** SQLite is queryable locally; JSON files move between CI steps. Run summaries for drift go to `results/history.jsonl`, the one results file that is committed.

**Every run records what could change its results:** prompt version and file hash, model and served snapshot, dataset version and content hash, judge model and version, git commit.

**Shared OpenAI plumbing (`llm.py`):** retries and structured-output schemas are used by both the classifier and the judge, so they live in one module.

## Comparison (Step 5, `compare.py`)

**Paired comparison on cases evaluated in both runs:** each case is compared with itself. Cases added to or removed from the dataset are listed but don't count; API-error cases are excluded.

**Status rules** (thresholds in `gate.yaml`):
- **INVALID** if either run has more than 10% API errors: results aren't trustworthy, rerun. This is a failed check, so an outage can neither pass a bad prompt nor be mistaken for a regression.
- **FAIL** if accuracy drops more than 8 points, any category more than 20, or the summary score more than 10.
- **WARN** at 3 / 10 / 5 points. Warnings don't block merging.
- **Per-category thresholds** catch a category collapsing while overall accuracy barely moves (4 wrong general cases = −26.7 for general, only −6.7 overall).
- With 60 cases one flip is 1.7 points, below the warning level: one flip is noise, as the runs below show.

**McNemar exact test on flipped cases (informational):** if a change made no difference, each flipped case is equally likely to flip either way. A small p-value means the imbalance is unlikely to be noise. It's shown, not used to gate, because with this few flips it rarely reaches significance; the thresholds are the rule.

**Changes that make results incomparable are surfaced:** different dataset hash (compare shared cases only), different judge version (summary scores not compared), different served model snapshot (noted, because it can explain a change with no prompt change).

## Reports and alerts (Steps 6–7)

**HTML report** (`report.py`, Jinja2 with auto-escaping because emails and model output are untrusted text): status with icon and reasons, scorecard, trend chart, regressed / improved / summary-drop cases with old and new output side by side, per-category and per-tag tables, run details. The chart follows the dataviz palette: validated blue/orange pair, direct labels plus a table view because orange is under 3:1 contrast on white, hover tooltips, a dark mode.

**PR comment in Markdown** with a hidden marker (`<!-- regression-gate-report -->`), so CI updates one comment instead of adding a new one per push.

**Slack is optional:** alerts go out only when `SLACK_WEBHOOK_URL` is set; otherwise the message is printed. Nothing else depends on Slack.

**Slow drift is tracked separately from per-run regressions**
Per-run checks compare two runs; a model updated behind an alias can lose 1 point a week and never cross the 3-point warning. Drift compares the average of the last 7 comparable runs with the 7 before (same prompt, dataset and judge; outage runs skipped) and warns above 3 points. It also reports served-model snapshot changes, the most likely cause. Fed by a nightly CI run.

## Results (Step 10 demo, on the LLM-drafted golden set)

| Prompt | Change | Accuracy | Summary score | Gate vs previous |
|---|---|---:|---:|---|
| v1 | zero-shot baseline | 95.0% | 96.4% | — |
| v2 | + labeling rules | 100% | 99.2% | PASS, 3 improvements (case-022, 051, 060) |
| v3 | + 4 few-shot examples | 100% | 98.3% | PASS, no gain, 33% higher cost per run |
| v4 | 6-word summaries + keyword shortcut | 98.3% | 71.1% | **FAIL**: summary score −28.1 |

- **v1's misses were exactly the predicted ones:** the two labeling rules its prompt didn't have.
- **v1 flipped `case-051` between two identical runs** (general, then billing) at temperature 0. Boundary cases are close to 50/50 for the model. This is why one flip isn't a regression.
- **Few-shot examples weren't worth their tokens** here, so v2 stays the candidate for production rather than v3.
- **v4 would have shipped if the gate only checked accuracy** (−1.7 points, within noise). It was caught because summaries are scored separately.
- **Caveat:** v2 reaches 100%, which is a ceiling. The golden set was drafted together with the rules v2 encodes, so v2 is partly fitted to it. Next steps: harder cases from real failures, and a held-out split that prompt iterations never look at. (All labels have since been human-reviewed, with no changes.)

## CI, packaging and docs (Steps 8–9)

**Three workflows:** unit tests and a Docker build on every push and PR (no secrets, no cost); the eval gate on PRs that touch `prompts/`, `src/`, `data/`, `gate.yaml` or dependencies; and a nightly drift run. The BASWE guide triggers only on `prompts/`, but Step 1.5 showed that code changes alter outputs too, and so do dataset, settings and dependency changes.

**The baseline is re-run in the PR job,** from a second checkout of main, instead of reusing a stored result. Both runs hit the same served model snapshot at the same time and share one judge cache, so a provider-side change or judge noise can't appear as the PR's regression. Costs one extra run (about $0.05–0.10 per PR).

**One sticky PR comment,** found by its hidden marker and updated on each push, plus the full report and run JSON as an artifact and the summary in the job summary.

**The check fails on FAIL and INVALID, passes on WARN.** Warnings are visible in the comment but don't block; INVALID blocks because nothing was verified, and a rerun fixes it.

**Slack only on WARN, FAIL, INVALID or drift.** A message on every green PR is noise people learn to ignore.

**Nightly run commits a one-line summary to `results/history.jsonl`** (the only committed results file) and caches the SQLite judge cache between nights, so unchanged summaries aren't re-judged. A detected drift fails the job, which also notifies the repository owner by email even without Slack.

**Docker image** with dependencies installed in a separate layer from code, so editing prompts doesn't reinstall packages. Configured through environment variables (`OPENAI_API_KEY`, `SLACK_WEBHOOK_URL`, `GATE_SETTINGS` for custom thresholds). Built and smoke-tested in CI rather than locally.

**Example reports are committed under `docs/examples/`** so the README can show output without running anything.

## Verified on GitHub (Step 10)

- **PR #1, v1 → v2: PASS.** Accuracy 91.7% → 100%, 5 improvements, 0 regressions, McNemar p = 0.062.
- **PR #2, v2 → v4 (stacked on #1): FAIL,** summary score −28.1 points; the check fails on the PR.
- **v1 scored 91.7% in CI but 95.0% and 96.7% in two local runs** (same prompt, same dataset): 3–5 misses depending on the run, all on boundary cases. Run-to-run variation is larger than a single run suggests, which confirms two decisions: re-running the baseline in the same job rather than reusing a stored score, and thresholds wider than one or two flips. Running each case several times and voting is the next improvement if this noise starts to matter.

## Branch protection (after the demo)

**`main` requires `gate`, `pytest` and `docker` to pass, and the PR branch to be up to date.** Without it, a FAIL was visible but didn't block anything. "Up to date" matters more than usual: the gate's baseline is main at the time it ran, so if main's prompt changed since, the comparison is stale and must rerun. Admins can still push directly (`enforce_admins` off), since this is a one-person repo.

**The gate now runs on every PR, and passes in seconds when nothing model-affecting changed.** A required check filtered by `paths:` would never start on a docs-only PR, which would block it forever. The job lists the PR's files and skips the eval (no API cost) unless prompts, code, data, settings or dependencies changed.

**Nightly history moved to an `eval-history` branch.** A protected branch rejects the bot's pushes, and a daily data commit on main was noise anyway. The nightly job checks out main for code and the data branch for history.
