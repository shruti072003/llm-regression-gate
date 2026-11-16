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

**Summary must be 3–100 characters after stripping whitespace**
The feature asks for a one-sentence summary. The minimum rejects empty or whitespace-only output (`str_strip_whitespace=True`). The maximum (about 15–20 words) rejects essays. *Revisit 100 once I see real model output.*

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
