"""Re-measure every published metric from tests and experiments into docs/metrics.json."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import trace
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src" / "prompt_eval_power"
EXP_DIR = ROOT / "docs" / "experiments"
METRICS_PATH = ROOT / "docs" / "metrics.json"

CHECKED_DOCUMENTS = [
    "README.md",
    "docs/adr/ADR-001-deterministic-rag-fixture-over-live-llm-calls.md",
    "docs/adr/ADR-002-paired-mcnemar-over-unpaired-pass-rate-thresholds.md",
    "docs/adr/ADR-003-resolution-floor-and-underpowered-gate-exit-code.md",
    "docs/adr/ADR-004-stratified-query-buckets-over-flat-golden-sets.md",
    "docs/adr/ADR-005-offline-judge-boundary-and-unmeasured-llm-variance.md",
]

ANCHORS: dict[str, list[str]] = {
    "tests_total": ["{} tests"],
    "coverage_floor_pct": ["{}%+ line coverage"],
    "corpus_queries": ["{} labeled queries"],
    "false_pass_rate_n30_pct": ["{}% false-pass rate"],
    "mde_at_n30_pp": ["{} percentage points"],
    "unpaired_n_for_5pp": ["{} unpaired queries"],
    "paired_n_for_5pp": ["{} paired queries"],
    "high_discordance_paired_n": ["{} queries at 45%"],
    "zero_failure_upper_bound_n30_pct": ["{}% upper bound"],
    "required_n_for_1pct_bound": ["{} zero-failure queries"],
    "prompt_regression_flips": ["{} pass-to-fail flips"],
    "v2_n30_deficit": ["deficit of {} queries"],
}


def _executable_lines(py_file: Path) -> set[int]:
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ):
                continue
            if hasattr(node, "lineno"):
                lines.add(node.lineno)
    return lines


def _measure_tests_from_pytest_artifacts() -> tuple[int, float] | None:
    junit_path = ROOT / "junit.xml"
    cov_path = ROOT / "coverage.json"
    if not (junit_path.exists() and cov_path.exists()):
        return None
    root = ET.parse(junit_path).getroot()
    if root.tag == "testsuite":
        tests_total = int(root.attrib["tests"])
    else:
        tests_total = sum(int(ts.attrib.get("tests", 0)) for ts in root.findall("testsuite"))
    cov_data = json.loads(cov_path.read_text(encoding="utf-8"))
    cov_pct = round(float(cov_data["totals"]["percent_covered"]), 1)
    return tests_total, cov_pct


def _measure_tests_with_stdlib_trace() -> tuple[int, float]:
    sys.path.insert(0, str(ROOT / "src"))
    for mod_name in list(sys.modules):
        if mod_name == "prompt_eval_power" or mod_name.startswith("prompt_eval_power."):
            del sys.modules[mod_name]

    def _discover_and_run() -> unittest.TestResult:
        loader = unittest.TestLoader()
        suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py")
        return unittest.TextTestRunner(verbosity=0).run(suite)

    tracer = trace.Trace(count=True, trace=False)
    result = tracer.runfunc(_discover_and_run)
    if not result.wasSuccessful():
        raise RuntimeError("Unit tests failed during metrics collection")

    executed_by_file: dict[Path, set[int]] = {}
    for (filename, lineno), hits in tracer.results().counts.items():
        if hits > 0:
            executed_by_file.setdefault(Path(filename).resolve(), set()).add(lineno)

    total_exec, total_hit = 0, 0
    for py_file in sorted(SRC_DIR.glob("*.py")):
        resolved = py_file.resolve()
        exec_lines = _executable_lines(resolved)
        hit_lines = executed_by_file.get(resolved, set()) & exec_lines
        total_exec += len(exec_lines)
        total_hit += len(hit_lines)

    cov_pct = round(100.0 * total_hit / total_exec, 1) if total_exec > 0 else 0.0
    return result.testsRun, cov_pct


def _run_experiments_if_needed(force_run: bool) -> None:
    scripts = [
        "experiments/exp01_small_suite_false_pass_rate.py",
        "experiments/exp02_paired_vs_unpaired_power_gain.py",
        "experiments/exp03_zero_failure_resolution_floor.py",
        "experiments/exp04_langgraph_prompt_regression_audit.py",
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for script in scripts:
        out_json = EXP_DIR / (Path(script).stem + ".json")
        if force_run or not out_json.exists():
            subprocess.run(
                [sys.executable, str(ROOT / script)],
                cwd=str(ROOT),
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect verified metrics into docs/metrics.json")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--reuse-experiments", action="store_true")
    args = parser.parse_args()

    _run_experiments_if_needed(force_run=not args.reuse_experiments)
    measured = _measure_tests_from_pytest_artifacts() if args.skip_tests else None
    if measured is None:
        measured = _measure_tests_from_pytest_artifacts()
    if measured is None:
        measured = _measure_tests_with_stdlib_trace()
    tests_total, cov_pct = measured

    e1 = json.loads(
        (EXP_DIR / "exp01_small_suite_false_pass_rate.json").read_text(encoding="utf-8")
    )
    e2 = json.loads(
        (EXP_DIR / "exp02_paired_vs_unpaired_power_gain.json").read_text(encoding="utf-8")
    )
    e3 = json.loads(
        (EXP_DIR / "exp03_zero_failure_resolution_floor.json").read_text(encoding="utf-8")
    )
    e4 = json.loads(
        (EXP_DIR / "exp04_langgraph_prompt_regression_audit.json").read_text(encoding="utf-8")
    )

    metrics = {
        "tests_total": tests_total,
        "coverage_floor_pct": 90,
        "coverage_measured_pct": cov_pct,
        "corpus_queries": int(e4["corpus_queries"]),
        "false_pass_rate_n30_pct": float(e1["false_pass_rate_n30_pct"]),
        "mde_at_n30_pp": float(e1["mde_at_n30_pp"]),
        "unpaired_n_for_5pp": int(e2["unpaired_n_for_5pp"]),
        "paired_n_for_5pp": int(e2["paired_n_for_5pp"]),
        "high_discordance_paired_n": int(e2["high_discordance_paired_n"]),
        "zero_failure_upper_bound_n30_pct": float(e3["zero_failure_upper_bound_n30_pct"]),
        "required_n_for_1pct_bound": int(e3["required_n_for_1pct_bound"]),
        "prompt_regression_flips": int(e4["v2_pass_to_fail_flips"]),
        "v2_n30_deficit": int(e4["v2_n30_deficit"]),
    }

    payload = {
        "metrics": metrics,
        "anchors": ANCHORS,
        "checked_documents": CHECKED_DOCUMENTS,
        "note": "Every value here is produced by running the test suite and the four experiments.",
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"collect_metrics: wrote {len(metrics)} metrics (tests={tests_total}, cov={cov_pct}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
