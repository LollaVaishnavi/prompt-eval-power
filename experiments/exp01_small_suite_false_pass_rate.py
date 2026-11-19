"""Experiment 01: Small Prompt Eval Suite False-Pass Rate and MDE.

Answers: How often does a typical 20-to-50 query prompt evaluation suite miss a
real 10-percentage-point or 5-percentage-point regression, and what is the true
minimum detectable effect at n=30?
"""

from __future__ import annotations

import json
from pathlib import Path

from prompt_eval_power.power import minimum_detectable_drop, unpaired_power

OUT_PATH = Path("docs/experiments/exp01_small_suite_false_pass_rate.json")
SAMPLE_SIZES = (20, 30, 50, 100, 200, 500)


def run() -> dict:
    p_base = 0.85
    alpha = 0.05
    rows = []
    for n in SAMPLE_SIZES:
        p10 = unpaired_power(n, p_base, 0.10, alpha=alpha)
        p5 = unpaired_power(n, p_base, 0.05, alpha=alpha)
        mde = minimum_detectable_drop(n, p_base, alpha=alpha, target_power=0.80, mode="unpaired")
        rows.append(
            {
                "n_queries": n,
                "power_10pp_pct": round(100.0 * p10, 1),
                "false_pass_rate_10pp_pct": round(100.0 * (1.0 - p10), 1),
                "power_5pp_pct": round(100.0 * p5, 1),
                "false_pass_rate_5pp_pct": round(100.0 * (1.0 - p5), 1),
                "mde_80_power_pp": round(100.0 * mde, 1),
            }
        )

    n30 = next(r for r in rows if r["n_queries"] == 30)
    payload = {
        "experiment": "exp01_small_suite_false_pass_rate",
        "baseline_pass_rate": p_base,
        "alpha": alpha,
        "false_pass_rate_n30_pct": n30["false_pass_rate_10pp_pct"],
        "false_pass_rate_n30_5pp_pct": n30["false_pass_rate_5pp_pct"],
        "mde_at_n30_pp": n30["mde_80_power_pp"],
        "sample_sizes": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    data = run()
    print("=== Experiment 01: Small-Suite False-Pass Rate (p_base=85%, alpha=0.05) ===")
    hdr = (
        f"{'n':>5} {'Power(10pp)':>12} {'False-Pass(10pp)':>17} "
        f"{'False-Pass(5pp)':>16} {'MDE@80%':>9}"
    )
    print(hdr)
    for r in data["sample_sizes"]:
        print(
            f"{r['n_queries']:>5} {r['power_10pp_pct']:>11.1f}% "
            f"{r['false_pass_rate_10pp_pct']:>16.1f}% "
            f"{r['false_pass_rate_5pp_pct']:>15.1f}% "
            f"{r['mde_80_power_pp']:>7.1f}pp"
        )
    print(
        f"At n=30: {data['false_pass_rate_n30_pct']:.1f}% false-pass rate on a 10pp drop; "
        f"MDE is {data['mde_at_n30_pp']:.1f} percentage points."
    )
