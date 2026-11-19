"""Experiment 03: Zero-Failure Resolution Floor (A Measured Zero Is Not Zero).

Answers: When a prompt passes 100% of a golden set with zero failures, what is
the exact 95% Clopper-Pearson upper bound on its true failure rate, and how
many zero-failure queries are required to certify <1% failure?
"""

from __future__ import annotations

import json
from pathlib import Path

from prompt_eval_power.stats import required_n_for_failure_bound, summarize_zero_failure_bound

OUT_PATH = Path("docs/experiments/exp03_zero_failure_resolution_floor.json")
ZERO_SIZES = (10, 20, 30, 50, 100, 300, 500)


def run() -> dict:
    rows = []
    for n in ZERO_SIZES:
        zb = summarize_zero_failure_bound(failures=0, n_queries=n, confidence_level=0.95)
        rows.append(
            {
                "n_queries": n,
                "resolution_floor_pct": round(100.0 * zb.resolution_floor, 2),
                "upper_bound_95_pct": round(100.0 * zb.upper_bound_rate, 1),
            }
        )

    n30 = next(r for r in rows if r["n_queries"] == 30)
    req_1pct = required_n_for_failure_bound(0.01, confidence_level=0.95)
    req_5pct = required_n_for_failure_bound(0.05, confidence_level=0.95)

    payload = {
        "experiment": "exp03_zero_failure_resolution_floor",
        "confidence_level": 0.95,
        "zero_failure_upper_bound_n30_pct": n30["upper_bound_95_pct"],
        "required_n_for_1pct_bound": req_1pct,
        "required_n_for_5pct_bound": req_5pct,
        "zero_failure_table": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    data = run()
    print("=== Experiment 03: Zero-Failure Resolution Floor (95% Clopper-Pearson) ===")
    print(f"{'Zero Failures':>14} {'1/n Floor':>11} {'95% Upper Bound':>16}")
    for r in data["zero_failure_table"]:
        print(
            f"0 / {r['n_queries']:<10} {r['resolution_floor_pct']:>10.2f}% "
            f"{r['upper_bound_95_pct']:>15.1f}%"
        )
    print(
        f"0/30 leaves a {data['zero_failure_upper_bound_n30_pct']:.1f}% upper bound; "
        f"certifying <1% failure requires {data['required_n_for_1pct_bound']} zero-failure queries."
    )
