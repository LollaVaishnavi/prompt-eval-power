"""Property tests for CLI exit codes, JSON output, and HTML report CSS integrity."""

from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from prompt_eval_power.cli import main
from prompt_eval_power.report import REPORT_CSS, render_html_report, write_html_report


def _sample_payloads() -> tuple[dict, dict, dict, dict]:
    exp01 = {
        "false_pass_rate_n30_pct": 75.2,
        "mde_at_n30_pp": 28.7,
        "sample_sizes": [
            {
                "n_queries": 30,
                "power_10pp_pct": 24.8,
                "false_pass_rate_10pp_pct": 75.2,
                "false_pass_rate_5pp_pct": 87.9,
                "mde_80_power_pp": 28.7,
            }
        ],
    }
    exp02 = {}
    exp03 = {
        "zero_failure_upper_bound_n30_pct": 9.5,
        "zero_failure_table": [
            {"n_queries": 30, "resolution_floor_pct": 3.33, "upper_bound_95_pct": 9.5}
        ],
    }
    exp04 = {
        "corpus_queries": 300,
        "v2_verdict": "REJECT_REGRESSION",
        "v2_buckets": [
            {
                "bucket": "negation_refusal",
                "baseline_pass_pct": 70.0,
                "candidate_pass_pct": 50.0,
                "net_delta_pp": -20.0,
                "pass_to_fail": 15,
                "fail_to_pass": 0,
                "p_value": 0.0001,
                "breached": True,
            }
        ],
    }
    return exp01, exp02, exp03, exp04


class TestCLIAndReport(unittest.TestCase):
    def test_cli_gate_returns_exit_2_on_v2_regression_at_n300(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main([
                "gate", "--baseline", "v1_baseline_grounded",
                "--candidate", "v2_concise_rewrite", "--sample-size", "300", "--json",
            ])
        self.assertEqual(code, 2)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["verdict"], "REJECT_REGRESSION")

    def test_cli_gate_returns_exit_3_when_suite_is_underpowered_at_n30(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main([
                "gate", "--baseline", "v1_baseline_grounded",
                "--candidate", "v2_concise_rewrite", "--sample-size", "30",
            ])
        self.assertEqual(code, 3)
        self.assertIn("UNDERPOWERED", buf.getvalue())

    def test_cli_gate_returns_exit_0_on_powered_safe_v3_prompt(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main([
                "gate", "--baseline", "v1_baseline_grounded",
                "--candidate", "v3_strict_citation", "--sample-size", "300",
            ])
        self.assertEqual(code, 0)
        self.assertIn("PASS", buf.getvalue())

    def test_cli_gate_loads_policy_json_and_returns_exit_1_on_invalid_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "policy.json"
            p.write_text(
                json.dumps({
                    "alpha": 0.05, "target_power": 0.80, "target_mde": 0.05,
                    "confidence_level": 0.95, "max_bucket_drop": 0.10,
                }),
                encoding="utf-8",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(["gate", "--policy", str(p), "--sample-size", "300"])
            self.assertEqual(code, 2)

        err = io.StringIO()
        with redirect_stderr(err):
            code_bad = main(["gate", "--alpha", "0.9"])
        self.assertEqual(code_bad, 1)
        self.assertIn("error:", err.getvalue())

    def test_cli_power_and_zero_bound_emit_text_and_json(self) -> None:
        for cmd in [
            ["power", "--sample-size", "30", "--json"],
            ["power", "--sample-size", "30"],
            ["zero-bound", "--failures", "0", "--sample-size", "30", "--json"],
            ["zero-bound", "--failures", "0", "--sample-size", "30"],
        ]:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(cmd)
            self.assertEqual(code, 0)
            self.assertGreater(len(buf.getvalue()), 10)

    def test_every_css_class_in_html_report_has_a_defined_stylesheet_rule(self) -> None:
        e1, e2, e3, e4 = _sample_payloads()
        for verdict in ("REJECT_REGRESSION", "UNDERPOWERED", "PASS"):
            e4_mod = {**e4, "v2_verdict": verdict}
            markup = render_html_report(e1, e2, e3, e4_mod)
            used: set[str] = set()
            for m in re.findall(r'class="([^"]+)"', markup):
                used.update(m.split())
            defined = set(re.findall(r"\.([a-zA-Z0-9_-]+)\s*\{", REPORT_CSS))
            self.assertEqual(used - defined, set())

    def test_write_html_report_creates_nested_file_with_all_three_headings(self) -> None:
        e1, e2, e3, e4 = _sample_payloads()
        with tempfile.TemporaryDirectory() as tmp:
            out = write_html_report(Path(tmp) / "out" / "report.html", e1, e2, e3, e4)
            text = out.read_text(encoding="utf-8")
        for hid in ("small-suite-blindness", "zero-failure-floor", "langgraph-prompt-audit"):
            self.assertIn(f'id="{hid}"', text)


if __name__ == "__main__":
    unittest.main()
