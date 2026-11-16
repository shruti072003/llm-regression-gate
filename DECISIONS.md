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
