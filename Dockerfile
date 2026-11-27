# The eval runner, golden dataset, prompts and reporting in one image.
# Configure with environment variables: OPENAI_API_KEY (required), SLACK_WEBHOOK_URL (optional),
# GATE_SETTINGS (path to a gate.yaml with your own thresholds; mount it into the container).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 PYTHONUNBUFFERED=1

# Dependencies first, so editing prompts or code doesn't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

COPY gate.yaml ./
COPY prompts ./prompts
COPY data ./data

ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "gate"]
CMD ["--help"]
