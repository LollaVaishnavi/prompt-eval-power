"""Self-contained HTML report renderer for prompt-eval-power audits.

Exists so evidence generation renders the exact statistical power tables,
resolution floors, and RAG prompt regression outcomes with WCAG AAA badge
contrast and 100% CSS class coverage.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

REPORT_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: #ffffff;
    color: #111827;
    margin: 0;
    padding: 32px;
    line-height: 1.5;
}
.report-shell {
    max-width: 920px;
    margin: 0 auto;
}
.header-block {
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 16px;
    margin-bottom: 24px;
}
.report-title {
    font-size: 26px;
    margin: 0 0 8px 0;
    color: #0f172a;
}
.report-subtitle {
    font-size: 14px;
    color: #475569;
    margin: 0 0 12px 0;
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.02em;
}
.badge-fail {
    background-color: #991b1b;
    color: #ffffff;
}
.badge-warn {
    background-color: #92400e;
    color: #ffffff;
}
.badge-pass {
    background-color: #166534;
    color: #ffffff;
}
.section-heading {
    font-size: 19px;
    color: #0f172a;
    margin-top: 26px;
    margin-bottom: 10px;
    border-left: 4px solid #1d4ed8;
    padding-left: 10px;
}
.kpi-grid {
    display: flex;
    gap: 14px;
    margin: 14px 0 20px 0;
}
.kpi-card {
    flex: 1;
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 12px 14px;
}
.kpi-label {
    font-size: 12px;
    color: #475569;
    text-transform: uppercase;
}
.kpi-value {
    font-size: 21px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 4px;
}
.data-table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 20px 0;
    font-size: 13px;
}
.data-table th {
    background-color: #f1f5f9;
    color: #0f172a;
    text-align: left;
    padding: 8px 10px;
    border: 1px solid #cbd5e1;
}
.data-table td {
    padding: 8px 10px;
    border: 1px solid #cbd5e1;
}
.num-cell {
    text-align: right;
    font-variant-numeric: tabular-nums;
}
.alert-cell {
    color: #991b1b;
    font-weight: 700;
}
.ok-cell {
    color: #166534;
    font-weight: 700;
}
.footnote {
    font-size: 12px;
    color: #475569;
    margin-top: 24px;
    border-top: 1px solid #e5e7eb;
    padding-top: 10px;
}
"""


def render_html_report(
    exp01: dict[str, Any],
    exp02: dict[str, Any],
    exp03: dict[str, Any],
    exp04: dict[str, Any],
) -> str:
    """Render all four experiment summaries into a self-contained HTML report."""
    verdict = str(exp04.get("v2_verdict", "REJECT_REGRESSION"))
    if verdict == "PASS":
        badge_cls, badge_txt = "badge badge-pass", "GATE PASSED: POWERED AND SAFE"
    elif verdict == "UNDERPOWERED":
        badge_cls, badge_txt = "badge badge-warn", "GATE BLOCKED: UNDERPOWERED SUITE"
    else:
        badge_cls, badge_txt = "badge badge-fail", "GATE REJECTED: PROMPT REGRESSION DETECTED"

    exp01_rows = []
    for row in exp01["sample_sizes"]:
        cls = "alert-cell" if row["false_pass_rate_10pp_pct"] > 20.0 else "ok-cell"
        exp01_rows.append(
            "<tr>"
            f"<td class=\"num-cell\">n={row['n_queries']}</td>"
            f"<td class=\"num-cell\">{row['power_10pp_pct']:.1f}%</td>"
            f"<td class=\"num-cell {cls}\">{row['false_pass_rate_10pp_pct']:.1f}%</td>"
            f"<td class=\"num-cell\">{row['false_pass_rate_5pp_pct']:.1f}%</td>"
            f"<td class=\"num-cell\">{row['mde_80_power_pp']:.1f} pp</td>"
            "</tr>"
        )

    exp03_rows = []
    for row in exp03["zero_failure_table"]:
        exp03_rows.append(
            "<tr>"
            f"<td class=\"num-cell\">0 / {row['n_queries']}</td>"
            f"<td class=\"num-cell\">{row['resolution_floor_pct']:.2f}%</td>"
            f"<td class=\"num-cell alert-cell\">{row['upper_bound_95_pct']:.1f}%</td>"
            "</tr>"
        )

    bucket_rows = []
    for row in exp04["v2_buckets"]:
        cls = "alert-cell" if row["breached"] else "ok-cell"
        bucket_rows.append(
            "<tr>"
            f"<td>{html.escape(row['bucket'])}</td>"
            f"<td class=\"num-cell\">{row['baseline_pass_pct']:.1f}%</td>"
            f"<td class=\"num-cell\">{row['candidate_pass_pct']:.1f}%</td>"
            f"<td class=\"num-cell {cls}\">{row['net_delta_pp']:+.1f} pp</td>"
            f"<td class=\"num-cell\">{row['pass_to_fail']} / {row['fail_to_pass']}</td>"
            f"<td class=\"num-cell\">{row['p_value']:.4f}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>prompt-eval-power Audit Report</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="report-shell">
  <div class="header-block">
    <h1 class="report-title">prompt-eval-power: Statistical Prompt Regression Report</h1>
    <p class="report-subtitle">Offline paired McNemar power analysis and RAG state-graph audit over {exp04['corpus_queries']} labeled queries.</p>
    <span class="{badge_cls}">{badge_txt}</span>
  </div>

  <h2 class="section-heading" id="small-suite-blindness">1. Small-Suite False-Pass Rate (10pp Regression)</h2>
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">n=30 False-Pass Rate (10pp Drop)</div>
      <div class="kpi-value">{exp01['false_pass_rate_n30_pct']:.1f}% false-pass rate</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">n=30 Minimum Detectable Drop</div>
      <div class="kpi-value">{exp01['mde_at_n30_pp']:.1f} percentage points</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">0/30 Failure 95% Upper Bound</div>
      <div class="kpi-value">{exp03['zero_failure_upper_bound_n30_pct']:.1f}% upper bound</div>
    </div>
  </div>
  <table class="data-table">
    <thead>
      <tr>
        <th class="num-cell">Suite Size</th>
        <th class="num-cell">Power (10pp Drop)</th>
        <th class="num-cell">False-Pass Rate (10pp)</th>
        <th class="num-cell">False-Pass Rate (5pp)</th>
        <th class="num-cell">MDE at 80% Power</th>
      </tr>
    </thead>
    <tbody>
      {''.join(exp01_rows)}
    </tbody>
  </table>

  <h2 class="section-heading" id="zero-failure-floor">2. Zero-Failure Resolution Floor (Clopper-Pearson 95%)</h2>
  <table class="data-table">
    <thead>
      <tr>
        <th class="num-cell">Observed Failures</th>
        <th class="num-cell">1/n Resolution Floor</th>
        <th class="num-cell">95% True Failure Upper Bound</th>
      </tr>
    </thead>
    <tbody>
      {''.join(exp03_rows)}
    </tbody>
  </table>

  <h2 class="section-heading" id="langgraph-prompt-audit">3. Offline RAG Prompt Regression Audit (v1 vs v2_concise_rewrite)</h2>
  <table class="data-table">
    <thead>
      <tr>
        <th>Query Bucket</th>
        <th class="num-cell">Baseline Pass</th>
        <th class="num-cell">Candidate Pass</th>
        <th class="num-cell">Net Delta</th>
        <th class="num-cell">Flips (P->F / F->P)</th>
        <th class="num-cell">McNemar p-val</th>
      </tr>
    </thead>
    <tbody>
      {''.join(bucket_rows)}
    </tbody>
  </table>

  <p class="footnote">Generated offline by prompt-eval-power. Every figure traces to docs/experiments/*.json and docs/metrics.json.</p>
</div>
</body>
</html>
"""


def write_html_report(
    output_path: str | Path,
    exp01: dict[str, Any],
    exp02: dict[str, Any],
    exp03: dict[str, Any],
    exp04: dict[str, Any],
) -> Path:
    """Write the rendered HTML report to disk."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html_report(exp01, exp02, exp03, exp04), encoding="utf-8")
    return target
