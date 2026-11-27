"""Slack alerts through an incoming webhook. Optional: without SLACK_WEBHOOK_URL the
message is printed instead, so every other part of the gate works without Slack."""

import json
import os
import urllib.request

from regression_gate.compare import Comparison
from regression_gate.drift import DriftReport
from regression_gate.report import STATUS_LABEL, headline


def comparison_message(c: Comparison, report_url: str | None) -> dict:
    icon, label = STATUS_LABEL[c.status]
    title = f"{icon} LLM regression gate {label}: {c.baseline_prompt} → {c.candidate_prompt}"
    lines = [headline(c), *[f"• {r}" for r in c.reasons]]
    if report_url:
        lines.append(f"<{report_url}|Full diff report>")
    return _blocks(title, "\n".join(lines))


def drift_message(d: DriftReport, report_url: str | None) -> dict:
    lines = [d.message, *[f"• served model changed: {m}" for m in d.served_model_changes]]
    if report_url:
        lines.append(f"<{report_url}|Run details>")
    return _blocks("⚠️ LLM regression gate: slow drift detected", "\n".join(lines))


def send(message: dict, webhook_url: str | None = None) -> bool:
    """Post to Slack. Returns True if sent, False if no webhook is configured."""
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        print("SLACK_WEBHOOK_URL not set; alert not sent:\n" + message["text"])
        return False
    request = urllib.request.Request(
        url, data=json.dumps(message).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()
    return True


def _blocks(title: str, body: str) -> dict:
    # `text` is the fallback shown in notifications; blocks are the formatted message.
    return {
        "text": f"{title}\n{body}",
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": title[:150]}},
            {"type": "section", "text": {"type": "mrkdwn", "text": body[:3000]}},
        ],
    }
