"""Render the comparison as a self-contained HTML diff report and a Markdown PR comment."""

from datetime import UTC, datetime

from jinja2 import Environment

from regression_gate.compare import Comparison, Status
from regression_gate.drift import DriftReport, RunSummary
from regression_gate.records import RunRecord

STATUS_LABEL = {
    Status.PASS: ("✅", "PASS"),
    Status.WARN: ("⚠️", "WARN"),
    Status.FAIL: ("❌", "FAIL"),
    Status.INVALID: ("⛔", "INVALID"),
}

_env = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)
_env.filters["pct"] = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"
_env.filters["pts"] = lambda v: "n/a" if v is None else f"{v * 100:+.1f} pts"
_env.filters["usd"] = lambda v: "n/a" if v is None else f"${v:.4f}"
_env.filters["ms"] = lambda v: "n/a" if v is None else f"{v:,.0f} ms"


def trend_chart(history: list[RunSummary], width: int = 720, height: int = 240) -> dict | None:
    """Geometry for a two-series line chart of accuracy and summary score over runs.
    Both are fractions, so they share one 0-100% axis."""
    if len(history) < 2:
        return None
    left, right, top, bottom = 44, 170, 16, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(history)

    def x(i: int) -> float:
        return left + plot_w * i / (n - 1)

    def y(v: float) -> float:
        return top + plot_h * (1 - v)

    series = []
    for key, name, css in (("accuracy", "Accuracy", "s1"), ("summary_score", "Summary score", "s2")):
        points = [(x(i), y(getattr(s, key)), getattr(s, key)) for i, s in enumerate(history) if getattr(s, key) is not None]
        if not points:
            continue
        series.append({
            "name": name,
            "css": css,
            "path": " ".join(f"{'M' if j == 0 else 'L'}{px:.1f},{py:.1f}" for j, (px, py, _) in enumerate(points)),
            "points": [{"x": px, "y": py} for px, py, _ in points],
            "label_y": points[-1][1],
            "label_value": points[-1][2],
        })
    # Keep the two end labels from overlapping.
    if len(series) == 2 and abs(series[0]["label_y"] - series[1]["label_y"]) < 16:
        upper, lower = sorted(series, key=lambda s: s["label_y"])
        upper["label_y"] -= 8
        lower["label_y"] += 8

    return {
        "width": width,
        "height": height,
        "left": left,
        "plot_right": left + plot_w,
        "top": top,
        "bottom_y": top + plot_h,
        "grid": [{"y": y(v), "label": f"{v:.0%}"} for v in (0, 0.25, 0.5, 0.75, 1.0)],
        "series": series,
        "label_x": left + plot_w + 10,
        "x_label_y": top + plot_h + 18,
        "runs": [
            {
                "x": x(i),
                "run_id": s.run_id,
                "date": f"{s.created_at:%Y-%m-%d %H:%M}",
                "prompt": s.prompt_version,
                "accuracy": s.accuracy,
                "summary": s.summary_score,
            }
            for i, s in enumerate(history)
        ],
    }


def render_html(
    comparison: Comparison,
    baseline: RunRecord,
    candidate: RunRecord,
    history: list[RunSummary],
    drift: DriftReport | None = None,
) -> str:
    icon, label = STATUS_LABEL[comparison.status]
    return _env.from_string(HTML_TEMPLATE).render(
        c=comparison,
        base=baseline,
        cand=candidate,
        icon=icon,
        label=label,
        chart=trend_chart(history),
        history=history,
        drift=drift,
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )


def render_markdown(comparison: Comparison, baseline: RunRecord, candidate: RunRecord, report_url: str | None = None) -> str:
    icon, label = STATUS_LABEL[comparison.status]
    return _env.from_string(MARKDOWN_TEMPLATE).render(
        c=comparison, base=baseline, cand=candidate, icon=icon, label=label, report_url=report_url
    )


def headline(comparison: Comparison) -> str:
    """One line for Slack and logs, e.g. '3 regressions, accuracy 94.0% -> 89.0%'."""
    c = comparison
    parts = [f"{len(c.regressions)} regression{'s' if len(c.regressions) != 1 else ''}"]
    if c.improvements:
        parts.append(f"{len(c.improvements)} improvement{'s' if len(c.improvements) != 1 else ''}")
    if c.baseline_accuracy is not None and c.candidate_accuracy is not None:
        parts.append(f"accuracy {c.baseline_accuracy:.1%} -> {c.candidate_accuracy:.1%}")
    if c.summary_delta is not None:
        parts.append(f"summary {c.baseline_summary_score:.1%} -> {c.candidate_summary_score:.1%}")
    return ", ".join(parts)


MARKDOWN_TEMPLATE = """\
<!-- regression-gate-report -->
## {{ icon }} LLM regression gate: {{ label }}

**{{ c.baseline_prompt }} → {{ c.candidate_prompt }}** on {{ c.paired_cases }} golden cases ({{ cand.model }}, judge {{ cand.judge_model }})

| | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Accuracy | {{ c.baseline_accuracy|pct }} | {{ c.candidate_accuracy|pct }} | {{ c.accuracy_delta|pts }} |
| Summary score | {{ c.baseline_summary_score|pct }} | {{ c.candidate_summary_score|pct }} | {{ c.summary_delta|pts }} |
| API error rate | {{ c.baseline_infra_error_rate|pct }} | {{ c.candidate_infra_error_rate|pct }} | |
| Cost (classifier + judge) | {{ ((base.metrics.classifier_cost_usd or 0) + (base.metrics.judge_cost_usd or 0))|usd }} | {{ ((cand.metrics.classifier_cost_usd or 0) + (cand.metrics.judge_cost_usd or 0))|usd }} | |

**{{ c.regressions|length }} regressed · {{ c.improvements|length }} improved · {{ c.summary_regressions|length }} summary drops**{% if c.mcnemar_p is not none %} · McNemar p = {{ "%.3f"|format(c.mcnemar_p) }}{% endif %}

{% if c.reasons %}
{% for r in c.reasons %}
- {{ r }}
{% endfor %}

{% endif %}
{% if c.regressions %}
<details><summary>Regressed cases</summary>

| Case | Expected | Baseline | Candidate |
|---|---|---|---|
{% for d in c.regressions %}
| `{{ d.case_id }}` {{ d.tags|join(", ") }} | {{ d.expected_category }} | {{ d.baseline_category }} | {{ d.candidate_category or d.candidate_error }} |
{% endfor %}

</details>

{% endif %}
{% if c.improvements %}
<details><summary>Improved cases</summary>

| Case | Expected | Baseline | Candidate |
|---|---|---|---|
{% for d in c.improvements %}
| `{{ d.case_id }}` {{ d.tags|join(", ") }} | {{ d.expected_category }} | {{ d.baseline_category or d.baseline_error }} | {{ d.candidate_category }} |
{% endfor %}

</details>

{% endif %}
{% for n in c.notes %}
> ℹ️ {{ n }}
{% endfor %}
{% if c.unreviewed_cases %}
> ℹ️ {{ c.unreviewed_cases }} of {{ c.shared_cases }} golden cases are LLM-drafted and not yet human-reviewed.
{% endif %}

{% if report_url %}[Full diff report]({{ report_url }}) · {% endif %}baseline `{{ c.baseline_run_id }}` · candidate `{{ c.candidate_run_id }}`
"""


HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ label }}: {{ c.baseline_prompt }} → {{ c.candidate_prompt }} · LLM regression gate</title>
<style>
  :root {
    --page: #f9f9f9; --surface: #ffffff; --ink: #191919; --ink-2: #666666; --rule: #e5e5e5;
    --s1: #257de0; --s2: #f47230;
    --good: #0aac23; --warning: #fbb315; --serious: #f18a59; --critical: #dc3c3d;
    --good-ink: #006c10; --critical-ink: #b42324;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --page: #0b0b0b; --surface: #191919; --ink: #efefef; --ink-2: #9c9c9c; --rule: #383838;
      --s1: #338ced; --s2: #e76624; --good-ink: #5fd36f; --critical-ink: #f08080;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink);
         font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 32px 0 12px; }
  .muted { color: var(--ink-2); }
  .card { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 16px; }
  .badge { display: inline-flex; gap: 6px; align-items: center; padding: 4px 10px; border-radius: 999px;
           font-weight: 600; border: 2px solid; }
  .badge.pass { border-color: var(--good); } .badge.warn { border-color: var(--warning); }
  .badge.fail { border-color: var(--critical); } .badge.invalid { border-color: var(--serious); }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
  .tile .k { color: var(--ink-2); font-size: 12px; }
  .tile .v { font-size: 22px; font-weight: 600; }
  .tile .d { font-size: 12px; }
  .up { color: var(--good-ink); } .down { color: var(--critical-ink); }
  table { width: 100%; border-collapse: collapse; background: var(--surface); }
  table.diffs { table-layout: fixed; }
  table.diffs col.case { width: 14%; } table.diffs col.expected { width: 10%; }
  table.diffs col.output { width: 38%; }
  th, td { text-align: left; padding: 8px; border-bottom: 1px solid var(--rule); vertical-align: top; }
  th { color: var(--ink-2); font-weight: 600; font-size: 12px; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .pair > div { padding: 8px; border-radius: 6px; border: 1px solid var(--rule); }
  .tag { display: inline-block; font-size: 11px; padding: 0 6px; border: 1px solid var(--rule);
         border-radius: 4px; margin: 0 4px 2px 0; color: var(--ink-2); }
  details summary { cursor: pointer; color: var(--ink-2); margin-top: 8px; }
  pre.email { white-space: pre-wrap; margin: 6px 0 0; font: 12px/1.4 ui-monospace, monospace; color: var(--ink-2); }
  ul.reasons { margin: 8px 0 0; padding-left: 20px; }
  .chart { position: relative; }
  .chart svg { display: block; width: 100%; height: auto; }
  .chart .grid line { stroke: var(--rule); stroke-width: 1; }
  .chart .grid text, .chart .label { fill: var(--ink-2); font-size: 11px; }
  .chart .label.value { fill: var(--ink); font-weight: 600; }
  .chart path { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .chart .s1 { stroke: var(--s1); } .chart .s2 { stroke: var(--s2); }
  .chart circle { stroke: var(--surface); stroke-width: 2; }
  .chart circle.s1 { fill: var(--s1); } .chart circle.s2 { fill: var(--s2); }
  .chart .crosshair { stroke: var(--ink-2); stroke-width: 1; stroke-dasharray: 3 3; visibility: hidden; }
  .legend { display: flex; gap: 16px; margin-bottom: 8px; color: var(--ink-2); font-size: 12px; }
  .legend i { display: inline-block; width: 12px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
  .tooltip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--rule);
             border-radius: 6px; padding: 8px 10px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.12);
             visibility: hidden; white-space: nowrap; }
</style>
</head>
<body>
<main>
  <p class="muted">LLM regression gate · {{ generated }}</p>
  <h1><span class="badge {{ c.status.value }}">{{ icon }} {{ label }}</span>
      &nbsp;{{ c.baseline_prompt }} → {{ c.candidate_prompt }}</h1>
  <p class="muted">
    Baseline <code>{{ base.run_id }}</code> ({{ base.git_sha or "no git sha" }}) ·
    Candidate <code>{{ cand.run_id }}</code> ({{ cand.git_sha or "no git sha" }})<br>
    Classifier {{ cand.model }} (served {{ cand.metrics.served_models|join(", ") or "n/a" }}, temperature {{ cand.temperature }}) ·
    Judge {{ cand.judge_model }}/{{ cand.judge_version }} ·
    Dataset v{{ cand.dataset_version }} ({{ cand.dataset_hash }})
  </p>

  {% if c.reasons or c.notes or c.unreviewed_cases %}
  <div class="card">
    {% if c.reasons %}<strong>Why {{ label }}</strong>
    <ul class="reasons">{% for r in c.reasons %}<li>{{ r }}</li>{% endfor %}</ul>{% endif %}
    {% if c.notes or c.unreviewed_cases %}
    <ul class="reasons muted">
      {% for n in c.notes %}<li>{{ n }}</li>{% endfor %}
      {% if c.unreviewed_cases %}<li>{{ c.unreviewed_cases }} of {{ c.shared_cases }} golden cases are LLM-drafted and not yet human-reviewed.</li>{% endif %}
    </ul>
    {% endif %}
  </div>
  {% endif %}

  <h2>Scorecard</h2>
  <div class="tiles">
    {% macro tile(k, v, d=None, good=None) %}
    <div class="card tile"><div class="k">{{ k }}</div><div class="v">{{ v }}</div>
      {% if d is not none %}<div class="d {{ 'up' if good else ('down' if good is not none and not good else 'muted') }}">{{ d }}</div>{% endif %}</div>
    {% endmacro %}
    {{ tile("Accuracy", c.candidate_accuracy|pct, (c.accuracy_delta|pts) ~ " vs " ~ (c.baseline_accuracy|pct),
            none if not c.accuracy_delta else c.accuracy_delta > 0) }}
    {{ tile("Summary score", c.candidate_summary_score|pct, (c.summary_delta|pts) ~ " vs " ~ (c.baseline_summary_score|pct),
            none if not c.summary_delta else c.summary_delta > 0) }}
    {{ tile("Regressions", c.regressions|length, (c.improvements|length) ~ " improved") }}
    {{ tile("Flip significance", "n/a" if c.mcnemar_p is none else "p = %.3f"|format(c.mcnemar_p), "McNemar exact test") }}
    {{ tile("Latency p95", cand.metrics.latency_p95_ms|ms, "baseline " ~ (base.metrics.latency_p95_ms|ms)) }}
    {{ tile("Run cost", ((cand.metrics.classifier_cost_usd or 0) + (cand.metrics.judge_cost_usd or 0))|usd,
            (cand.metrics.input_tokens + cand.metrics.output_tokens) ~ " classifier tokens") }}
    {{ tile("API errors", c.candidate_infra_error_rate|pct, "baseline " ~ (c.baseline_infra_error_rate|pct)) }}
    {{ tile("Format fixes", cand.metrics.normalized_rate|pct, "invalid output " ~ (cand.metrics.invalid_output_rate|pct)) }}
  </div>

  {% if chart %}
  <h2>Trend over the last {{ history|length }} runs</h2>
  <div class="card chart" id="trend">
    <div class="legend">{% for s in chart.series %}<span><i style="background: var(--{{ s.css }})"></i>{{ s.name }}</span>{% endfor %}</div>
    <svg viewBox="0 0 {{ chart.width }} {{ chart.height }}" role="img" aria-label="Accuracy and summary score per run">
      <g class="grid">
        {% for g in chart.grid %}
        <line x1="{{ chart.left }}" x2="{{ chart.plot_right }}" y1="{{ g.y }}" y2="{{ g.y }}"></line>
        <text x="{{ chart.left - 8 }}" y="{{ g.y + 4 }}" text-anchor="end">{{ g.label }}</text>
        {% endfor %}
      </g>
      <line class="crosshair" y1="{{ chart.top }}" y2="{{ chart.bottom_y }}"></line>
      {% for s in chart.series %}
      <path class="{{ s.css }}" d="{{ s.path }}"></path>
      {% for p in s.points %}<circle class="{{ s.css }}" cx="{{ p.x }}" cy="{{ p.y }}" r="4"></circle>{% endfor %}
      <text class="label" x="{{ chart.label_x }}" y="{{ s.label_y + 4 }}">{{ s.name }} <tspan class="label value">{{ s.label_value|pct }}</tspan></text>
      {% endfor %}
      {% for r in chart.runs %}<text class="label" x="{{ r.x }}" y="{{ chart.x_label_y }}" text-anchor="middle">{{ r.prompt }}</text>{% endfor %}
      <rect class="hit" x="{{ chart.left - 10 }}" y="{{ chart.top }}" width="{{ chart.plot_right - chart.left + 20 }}"
            height="{{ chart.bottom_y - chart.top }}" fill="transparent"></rect>
    </svg>
    <div class="tooltip"></div>
    <details><summary>Show as table</summary>
      <table><tr><th>Run</th><th>Date</th><th>Prompt</th><th class="num">Accuracy</th><th class="num">Summary score</th></tr>
      {% for r in chart.runs %}<tr><td><code>{{ r.run_id }}</code></td><td>{{ r.date }}</td><td>{{ r.prompt }}</td>
        <td class="num">{{ r.accuracy|pct }}</td><td class="num">{{ r.summary|pct }}</td></tr>{% endfor %}
      </table>
    </details>
  </div>
  {% endif %}
  {% if drift %}<p class="muted">Slow drift ({{ drift.comparable_runs }} comparable runs): {{ drift.message }}</p>{% endif %}

  <h2>Regressed cases ({{ c.regressions|length }})</h2>
  {% macro diff_rows(diffs, empty) %}
  {% if not diffs %}<p class="muted">{{ empty }}</p>{% else %}
  <table class="diffs">
    <colgroup><col class="case"><col class="expected"><col class="output"><col class="output"></colgroup>
    <tr><th>Case</th><th>Expected</th><th>Baseline output</th><th>Candidate output</th></tr>
    {% for d in diffs %}
    <tr>
      <td><code>{{ d.case_id }}</code><br>{% for t in d.tags %}<span class="tag">{{ t }}</span>{% endfor %}
          {% if not d.reviewed %}<span class="tag">unreviewed</span>{% endif %}</td>
      <td>{{ d.expected_category }}</td>
      <td><strong>{{ d.baseline_category or "no answer" }}</strong>{% if d.baseline_score is not none %} · summary {{ d.baseline_score|pct }}{% endif %}<br>
          <span class="muted">{{ d.baseline_summary or d.baseline_error or "" }}</span></td>
      <td><strong>{{ d.candidate_category or "no answer" }}</strong>{% if d.candidate_score is not none %} · summary {{ d.candidate_score|pct }}{% endif %}<br>
          <span class="muted">{{ d.candidate_summary or d.candidate_error or "" }}</span></td>
    </tr>
    <tr><td colspan="4" style="padding-top:0"><details><summary>Email</summary><pre class="email">{{ d.email }}</pre></details></td></tr>
    {% endfor %}
  </table>
  {% endif %}
  {% endmacro %}
  {{ diff_rows(c.regressions, "No case flipped from correct to wrong.") }}

  <h2>Improved cases ({{ c.improvements|length }})</h2>
  {{ diff_rows(c.improvements, "No case flipped from wrong to correct.") }}

  <h2>Summary drops ({{ c.summary_regressions|length }})</h2>
  {{ diff_rows(c.summary_regressions, "No summary lost half or more of its key facts.") }}

  <h2>Accuracy by category</h2>
  {% macro group_table(groups, first) %}
  <table>
    <tr><th>{{ first }}</th><th class="num">Cases</th><th class="num">Baseline</th><th class="num">Candidate</th><th class="num">Change</th></tr>
    {% for name, g in groups.items() %}
    <tr><td>{{ name }}</td><td class="num">{{ g.total }}</td><td class="num">{{ g.baseline|pct }}</td><td class="num">{{ g.candidate|pct }}</td>
        <td class="num {{ 'up' if g.delta and g.delta > 0 else ('down' if g.delta and g.delta < 0 else 'muted') }}">{{ g.delta|pts }}</td></tr>
    {% endfor %}
  </table>
  {% endmacro %}
  {{ group_table(c.by_category, "Category") }}

  <h2>Accuracy by edge-case tag</h2>
  {{ group_table(c.by_tag, "Tag") }}

  <h2>Run details</h2>
  <table>
    <tr><th></th><th>Baseline</th><th>Candidate</th></tr>
    <tr><td>Prompt</td><td>{{ base.prompt_version }} ({{ base.prompt_sha }})</td><td>{{ cand.prompt_version }} ({{ cand.prompt_sha }})</td></tr>
    <tr><td>Served model</td><td>{{ base.metrics.served_models|join(", ") }}</td><td>{{ cand.metrics.served_models|join(", ") }}</td></tr>
    <tr><td>Dataset</td><td>v{{ base.dataset_version }} ({{ base.dataset_hash }})</td><td>v{{ cand.dataset_version }} ({{ cand.dataset_hash }})</td></tr>
    <tr><td>Judge</td><td>{{ base.judge_model }}/{{ base.judge_version }}</td><td>{{ cand.judge_model }}/{{ cand.judge_version }}</td></tr>
    <tr><td>Latency p50 / p95</td><td>{{ base.metrics.latency_p50_ms|ms }} / {{ base.metrics.latency_p95_ms|ms }}</td><td>{{ cand.metrics.latency_p50_ms|ms }} / {{ cand.metrics.latency_p95_ms|ms }}</td></tr>
    <tr><td>Tokens in / out</td><td>{{ base.metrics.input_tokens }} / {{ base.metrics.output_tokens }}</td><td>{{ cand.metrics.input_tokens }} / {{ cand.metrics.output_tokens }}</td></tr>
    <tr><td>Classifier / judge cost</td><td>{{ base.metrics.classifier_cost_usd|usd }} / {{ base.metrics.judge_cost_usd|usd }}</td><td>{{ cand.metrics.classifier_cost_usd|usd }} / {{ cand.metrics.judge_cost_usd|usd }}</td></tr>
    <tr><td>Unsupported claims</td><td>{{ base.metrics.unsupported_claim_rate|pct }}</td><td>{{ cand.metrics.unsupported_claim_rate|pct }}</td></tr>
    <tr><td>Duration</td><td>{{ "%.1f"|format(base.duration_s) }} s</td><td>{{ "%.1f"|format(cand.duration_s) }} s</td></tr>
  </table>
</main>
{% if chart %}
<script>
  (function () {
    const runs = {{ chart.runs|tojson }};
    const box = document.getElementById("trend");
    const svg = box.querySelector("svg"), tip = box.querySelector(".tooltip"), cross = svg.querySelector(".crosshair");
    const pct = v => v === null ? "n/a" : (v * 100).toFixed(1) + "%";
    function nearest(evt) {
      const pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
      const x = pt.matrixTransform(svg.getScreenCTM().inverse()).x;
      return runs.reduce((best, r) => Math.abs(r.x - x) < Math.abs(best.x - x) ? r : best, runs[0]);
    }
    svg.addEventListener("mousemove", evt => {
      const r = nearest(evt);
      cross.setAttribute("x1", r.x); cross.setAttribute("x2", r.x); cross.style.visibility = "visible";
      tip.textContent = "";
      [["", r.date + " · " + r.prompt], ["Accuracy ", pct(r.accuracy)], ["Summary score ", pct(r.summary)], ["", r.run_id]]
        .forEach(([k, v]) => { const d = document.createElement("div"); d.textContent = k + v; tip.appendChild(d); });
      const rect = box.getBoundingClientRect(), scale = svg.getBoundingClientRect().width / {{ chart.width }};
      tip.style.left = Math.min(r.x * scale + 24, rect.width - 220) + "px";
      tip.style.top = "40px"; tip.style.visibility = "visible";
    });
    svg.addEventListener("mouseleave", () => { tip.style.visibility = "hidden"; cross.style.visibility = "hidden"; });
  })();
</script>
{% endif %}
</body>
</html>
"""
