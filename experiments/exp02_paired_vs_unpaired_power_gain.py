"""Experiment 02: Paired McNemar vs Unpaired Pass-Rate Sample Size Requirements.

Answers: How many labeled queries does paired evaluation save over an unpaired
pass-rate threshold to detect a 5pp regression at 80% power, and how does high
discordance erode that gain?
"""

from __future__ import annotations

import json
from pathlib import Path

from prompt_eval_power.power import required_n_paired, required_n_unpaired

OUT_PATH = Path("docs/experiments/exp02_paired_vs_unpaired_power_gain.json")
DISCORDANCE_LEVELS = (0.08, 0.12, 0.15, 0.20, 0.30, 0.45)


def run() -> dict:
    p_base = 0.85
    delta = 0.05
    alpha = 0.05
    target_power = 0.80

    unpaired_n = required_n_unpaired(p_base, delta, alpha=alpha, target_power=target_power)
    rows = []
    for disc in DISCORDANCE_LEVELS:
        paired_n = required_n_paired(disc, delta, alpha=alpha, target_power=target_power)
        reduction_pct = round(100.0 * (unpaired_n - paired_n) / unpaired_n, 1)
        rows.append(
            {
                "discordant_rate": disc,
                "paired_n": paired_n,
                "unpaired_n": unpaired_n,
                "query_reduction_pct": reduction_pct,
            }
        )

    typical = next(r for r in rows if r["discordant_rate"] == 0.15)
    high_disc = next(r for r in rows if r["discordant_rate"] == 0.45)
    payload = {
        "experiment": "exp02_paired_vs_unpaired_power_gain",
        "baseline_pass_rate": p_base,
        "target_drop_pp": 5.0,
        "unpaired_n_for_5pp": unpaired_n,
        "paired_n_for_5pp": typical["paired_n"],
        "typical_reduction_pct": typical["query_reduction_pct"],
        "high_discordance_paired_n": high_disc["paired_n"],
        "discordance_sweep": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    data = run()
    print("=== Experiment 02: Paired McNemar vs Unpaired Sample Size (5pp Drop, 80% Power) ===")
    print(f"{'Discordant Rate':>16} {'Paired n':>10} {'Unpaired n':>12} {'Query Savings':>14}")
    for r in data["discordance_sweep"]:
        print(
            f"{100 * r['discordant_rate']:>15.0f}% {r['paired_n']:>10} "
            f"{r['unpaired_n']:>12} {r['query_reduction_pct']:>+13.1f}%"
        )
    print(
        f"Typical (15% discordance): {data['paired_n_for_5pp']} paired queries vs "
        f"{data['unpaired_n_for_5pp']} unpaired queries ({data['typical_reduction_pct']:.1f}% fewer)."
    )
