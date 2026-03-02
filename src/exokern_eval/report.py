"""HTML Report Card generator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from jinja2 import Environment, BaseLoader

from exokern_eval.evaluator import EvalResults


REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EXOKERN Policy Report Card</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d28; --border: #2a2d3a;
    --text: #e4e4e7; --muted: #9ca3af; --accent: #6366f1;
    --green: #22c55e; --red: #ef4444; --amber: #f59e0b;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.6; padding: 2rem; }
  .container { max-width: 900px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
  .subtitle { color: var(--muted); font-size: 0.875rem; margin-bottom: 2rem; }
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
  .card h2 { font-size: 1.1rem; margin-bottom: 1rem; color: var(--accent); }
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }
  .metric { text-align: center; }
  .metric .value { font-size: 2rem; font-weight: 700; }
  .metric .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .green { color: var(--green); }
  .amber { color: var(--amber); }
  .red { color: var(--red); }
  table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
  th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; font-size: 0.75rem; text-transform: uppercase; }
  .badge { display: inline-block; padding: 0.125rem 0.5rem; border-radius: 999px;
           font-size: 0.75rem; font-weight: 600; }
  .badge-green { background: rgba(34,197,94,0.15); color: var(--green); }
  .badge-amber { background: rgba(245,158,11,0.15); color: var(--amber); }
  .badge-red { background: rgba(239,68,68,0.15); color: var(--red); }
  .footer { text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 2rem; padding-top: 1rem;
            border-top: 1px solid var(--border); }
  .footer a { color: var(--accent); text-decoration: none; }
  .comparison { margin-top: 1rem; }
  .comparison .bar-container { display: flex; align-items: center; gap: 0.5rem; margin: 0.25rem 0; }
  .comparison .bar { height: 24px; border-radius: 4px; min-width: 2px; transition: width 0.3s; }
  .comparison .bar-label { font-size: 0.75rem; color: var(--muted); min-width: 80px; }
  .comparison .bar-value { font-size: 0.875rem; font-weight: 600; min-width: 60px; }
  .bar-yours { background: var(--accent); }
  .bar-baseline { background: var(--green); }
</style>
</head>
<body>
<div class="container">

<h1>EXOKERN Policy Report Card</h1>
<p class="subtitle">Generated {{ timestamp }} &mdash; {{ n_episodes }} episodes evaluated</p>

<div class="card">
  <h2>Summary</h2>
  <div class="metrics">
    <div class="metric">
      <div class="value {{ sr_color }}">{{ success_rate }}%</div>
      <div class="label">Success Rate</div>
    </div>
    <div class="metric">
      <div class="value">{{ avg_force }} N</div>
      <div class="label">Avg Force</div>
    </div>
    <div class="metric">
      <div class="value">{{ peak_force }} N</div>
      <div class="label">Peak Force</div>
    </div>
    <div class="metric">
      <div class="value">{{ avg_time }}s</div>
      <div class="label">Avg Time</div>
    </div>
  </div>
</div>

<div class="card">
  <h2>Details</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>Grade</th></tr>
    <tr>
      <td>Success Rate</td>
      <td>{{ success_rate }}% (95% CI: {{ ci_low }}% &ndash; {{ ci_high }}%)</td>
      <td><span class="badge badge-{{ sr_badge }}">{{ sr_grade }}</span></td>
    </tr>
    <tr>
      <td>Avg Contact Force</td>
      <td>{{ avg_force }} N</td>
      <td><span class="badge badge-{{ force_badge }}">{{ force_grade }}</span></td>
    </tr>
    <tr>
      <td>Peak Contact Force</td>
      <td>{{ peak_force }} N</td>
      <td>&mdash;</td>
    </tr>
    <tr>
      <td>Episodes</td>
      <td>{{ n_episodes }}</td>
      <td>&mdash;</td>
    </tr>
    <tr>
      <td>Condition</td>
      <td>{{ condition }}</td>
      <td>&mdash;</td>
    </tr>
  </table>
</div>

{% if baseline %}
<div class="card">
  <h2>Comparison with EXOKERN SOTA Models (or custom baseline)</h2>
  <table>
    <tr><th>Metric</th><th>Your Policy</th><th>Reference Baseline</th><th>Delta</th></tr>
    <tr>
      <td>Success Rate</td>
      <td>{{ success_rate }}%</td>
      <td>{{ baseline.success_rate }}%</td>
      <td class="{{ 'green' if sr_delta >= 0 else 'red' }}">{{ '+' if sr_delta >= 0 else '' }}{{ sr_delta }}%</td>
    </tr>
    <tr>
      <td>Avg Force</td>
      <td>{{ avg_force }} N</td>
      <td>{{ baseline.avg_force }} N</td>
      <td class="{{ 'green' if force_delta <= 0 else 'red' }}">{{ '+' if force_delta > 0 else '' }}{{ force_delta }} N</td>
    </tr>
  </table>
  <div class="comparison">
    <p style="font-size:0.75rem; color:var(--muted); margin-bottom:0.5rem;">Force comparison (lower is better)</p>
    <div class="bar-container">
      <div class="bar-label">You</div>
      <div class="bar bar-yours" style="width: {{ your_bar_pct }}%"></div>
      <div class="bar-value">{{ avg_force }} N</div>
    </div>
    <div class="bar-container">
      <div class="bar-label">Reference</div>
      <div class="bar bar-baseline" style="width: {{ baseline_bar_pct }}%"></div>
      <div class="bar-value">{{ baseline.avg_force }} N</div>
    </div>
  </div>
</div>
{% endif %}

<div class="card">
  <h2>Environment</h2>
  <table>
    <tr><td>Environment</td><td>{{ env_name }}</td></tr>
    <tr><td>Policy Architecture</td><td>Diffusion Policy (Temporal U-Net 1D)</td></tr>
    <tr><td>Observation Dim</td><td>{{ obs_dim }}</td></tr>
    <tr><td>Action Dim</td><td>{{ action_dim }}</td></tr>
  </table>
</div>

<div class="footer">
  Generated by <a href="https://github.com/Exokern/exokern_eval">exokern-eval</a> v{{ version }}
  &mdash; <a href="https://huggingface.co/EXOKERN">EXOKERN on HuggingFace</a>
</div>

</div>
</body>
</html>
"""


def _grade_success_rate(sr: float) -> tuple[str, str]:
    if sr >= 95:
        return "green", "Excellent"
    if sr >= 80:
        return "amber", "Good"
    return "red", "Needs Work"


def _grade_force(f: float) -> tuple[str, str]:
    if f <= 5.0:
        return "green", "Gentle"
    if f <= 10.0:
        return "amber", "Moderate"
    return "red", "High"


def generate_report(
    results: EvalResults,
    env_name: str,
    obs_dim: int,
    action_dim: int,
    baseline: Optional[dict[str, Any]] = None,
    output_path: Union[str, Path] = "report.html",
) -> Path:
    """Render an HTML report card and write it to disk."""
    from exokern_eval import __version__

    sr = results.success_rate
    ci = results.to_dict()["success_rate_ci95"]
    sr_badge, sr_grade = _grade_success_rate(sr)
    force_badge, force_grade = _grade_force(results.mean_force)

    ctx: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "version": __version__,
        "n_episodes": results.n_episodes,
        "success_rate": round(sr, 1),
        "avg_force": round(results.mean_force, 1),
        "peak_force": round(results.mean_peak_force, 1),
        "avg_time": round(results.mean_time, 1),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "sr_color": "green" if sr >= 95 else ("amber" if sr >= 80 else "red"),
        "sr_badge": sr_badge,
        "sr_grade": sr_grade,
        "force_badge": force_badge,
        "force_grade": force_grade,
        "condition": results.condition,
        "env_name": env_name,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "baseline": None,
    }

    if baseline:
        ctx["baseline"] = baseline
        ctx["sr_delta"] = round(sr - baseline["success_rate"], 1)
        ctx["force_delta"] = round(results.mean_force - baseline["avg_force"], 1)
        max_force = max(results.mean_force, baseline["avg_force"], 0.1)
        ctx["your_bar_pct"] = round(results.mean_force / max_force * 80, 1)
        ctx["baseline_bar_pct"] = round(baseline["avg_force"] / max_force * 80, 1)

    env = Environment(loader=BaseLoader())
    template = env.from_string(REPORT_TEMPLATE)
    html = template.render(**ctx)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


def generate_json_report(
    results: EvalResults,
    env_name: str,
    baseline: Optional[dict[str, Any]] = None,
    output_path: Union[str, Path] = "report.json",
) -> Path:
    """Write a machine-readable JSON report."""
    data = {
        "tool": "exokern-eval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": env_name,
        "results": results.to_dict(),
    }
    if baseline:
        data["baseline"] = baseline
        data["delta"] = {
            "success_rate": round(results.success_rate - baseline["success_rate"], 1),
            "avg_force": round(results.mean_force - baseline["avg_force"], 2),
        }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    return out
