import tempfile
from pathlib import Path

from exokern_eval.evaluator import EvalResults
from exokern_eval.report import generate_report, generate_json_report


def _make_results() -> EvalResults:
    r = EvalResults(condition="full_ft", n_episodes=10)
    r.successes = [True] * 9 + [False]
    r.avg_forces = [3.5] * 10
    r.max_forces = [8.0] * 10
    r.completion_times = [25.0] * 10
    return r


def test_html_report_generates():
    with tempfile.TemporaryDirectory() as d:
        out = generate_report(
            _make_results(), "TestEnv-v0", obs_dim=22, action_dim=7,
            output_path=Path(d) / "report.html",
        )
        assert out.exists()
        html = out.read_text()
        assert "90.0%" in html
        assert "EXOKERN" in html


def test_json_report_generates():
    with tempfile.TemporaryDirectory() as d:
        out = generate_json_report(
            _make_results(), "TestEnv-v0",
            output_path=Path(d) / "report.json",
        )
        assert out.exists()
        import json
        data = json.loads(out.read_text())
        assert data["results"]["success_rate_pct"] == 90.0
