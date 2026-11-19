"""Command-line interface for prompt-eval-power with four semantic exit codes.

Exit codes:
    0: PASS (suite is sufficiently powered and no statistically significant regression found).
    2: REJECT_REGRESSION (statistically significant overall or bucket prompt regression).
    3: UNDERPOWERED (no regression proven, but suite lacks sample size to detect target MDE).
    1: Invalid input, configuration error, or runtime failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from prompt_eval_power.config import CorpusConfig, GatePolicy, PolicyValidationError
from prompt_eval_power.corpus import generate_rag_corpus
from prompt_eval_power.pipeline import PROMPT_TEMPLATES, RagStateGraph
from prompt_eval_power.power import (
    evaluate_gate,
    minimum_detectable_drop,
    required_n_paired,
    required_n_unpaired,
    unpaired_power,
)
from prompt_eval_power.stats import (
    StatsValidationError,
    required_n_for_failure_bound,
    summarize_zero_failure_bound,
)


def _cmd_gate(args: argparse.Namespace) -> int:
    policy = (
        GatePolicy.from_json(args.policy)
        if args.policy
        else GatePolicy(
            alpha=args.alpha,
            target_power=args.target_power,
            target_mde=args.target_mde,
        )
    )
    corpus = generate_rag_corpus(CorpusConfig(seed=args.seed, query_count=args.corpus_size))
    graph = RagStateGraph(corpus)
    base_outcomes = graph.evaluate_prompt(args.baseline, limit=args.sample_size)
    cand_outcomes = graph.evaluate_prompt(args.candidate, limit=args.sample_size)

    decision = evaluate_gate(
        baseline_pass=[o.passed for o in base_outcomes],
        candidate_pass=[o.passed for o in cand_outcomes],
        buckets=[o.bucket for o in base_outcomes],
        policy=policy,
    )

    payload = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "n_queries": decision.overall.n_total,
        "baseline_pass_pct": round(100.0 * decision.overall.baseline_pass_rate, 1),
        "candidate_pass_pct": round(100.0 * decision.overall.candidate_pass_rate, 1),
        "net_delta_pp": round(100.0 * decision.overall.net_delta, 1),
        "pass_to_fail_flips": decision.overall.n_pass_to_fail,
        "fail_to_pass_flips": decision.overall.n_fail_to_pass,
        "p_value_regression": round(decision.overall.p_value_regression, 5),
        "achieved_power_pct": round(100.0 * decision.achieved_power_at_target_mde, 1),
        "mde_pp": round(100.0 * decision.mde_at_current_n, 1),
        "required_n_paired": decision.required_n_paired,
        "sample_size_deficit": decision.sample_size_deficit,
        "verdict": decision.verdict,
        "exit_code": decision.exit_code,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"prompt-eval-power gate | {args.baseline} vs {args.candidate} "
            f"(n={decision.overall.n_total}, target MDE={100 * policy.target_mde:.1f}pp)"
        )
        print(
            f"  pass rate:  {payload['baseline_pass_pct']:.1f}% -> "
            f"{payload['candidate_pass_pct']:.1f}% ({payload['net_delta_pp']:+.1f} pp)"
        )
        print(
            f"  discordant: {payload['pass_to_fail_flips']} pass-to-fail flips, "
            f"{payload['fail_to_pass_flips']} fail-to-pass flips "
            f"(McNemar p={payload['p_value_regression']:.5f})"
        )
        print(
            f"  power:      {payload['achieved_power_pct']:.1f}% at target MDE | "
            f"current MDE={payload['mde_pp']:.1f} pp | "
            f"required={decision.required_n_paired} paired queries "
            f"(deficit={decision.sample_size_deficit})"
        )
        print(f"  verdict:    {decision.verdict} (exit {decision.exit_code})")
    return decision.exit_code


def _cmd_power(args: argparse.Namespace) -> int:
    pwr_u = unpaired_power(args.sample_size, args.baseline_rate, args.drop, alpha=args.alpha)
    mde_u = minimum_detectable_drop(
        args.sample_size, args.baseline_rate, alpha=args.alpha, target_power=args.target_power
    )
    req_u = required_n_unpaired(
        args.baseline_rate, args.drop, alpha=args.alpha, target_power=args.target_power
    )
    req_p = required_n_paired(
        args.discordant_rate, args.drop, alpha=args.alpha, target_power=args.target_power
    )
    payload = {
        "n_queries": args.sample_size,
        "baseline_rate": args.baseline_rate,
        "drop": args.drop,
        "unpaired_power_pct": round(100.0 * pwr_u, 1),
        "false_pass_rate_pct": round(100.0 * (1.0 - pwr_u), 1),
        "mde_unpaired_pp": round(100.0 * mde_u, 1),
        "required_n_unpaired": req_u,
        "required_n_paired": req_p,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Power check (n={args.sample_size}, p_base={100 * args.baseline_rate:.1f}%, "
            f"drop={100 * args.drop:.1f}pp):"
        )
        print(
            f"  unpaired power:  {payload['unpaired_power_pct']:.1f}% "
            f"({payload['false_pass_rate_pct']:.1f}% false-pass rate)"
        )
        print(f"  unpaired MDE:    {payload['mde_unpaired_pp']:.1f} percentage points")
        print(
            f"  required size:   {req_u} unpaired queries vs {req_p} paired queries "
            f"(at discordant rate={args.discordant_rate:.2f})"
        )
    return 0


def _cmd_zero_bound(args: argparse.Namespace) -> int:
    bound = summarize_zero_failure_bound(
        failures=args.failures,
        n_queries=args.sample_size,
        confidence_level=args.confidence,
    )
    req_1pct = required_n_for_failure_bound(0.01, confidence_level=args.confidence)
    payload = {
        "failures": bound.observed_failures,
        "n_queries": bound.n_queries,
        "resolution_floor_pct": round(100.0 * bound.resolution_floor, 2),
        "upper_bound_pct": round(100.0 * bound.upper_bound_rate, 1),
        "required_n_for_1pct": req_1pct,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Zero-failure resolution ({bound.observed_failures}/{bound.n_queries} failures, "
            f"{100 * args.confidence:.0f}% CI):"
        )
        print(f"  resolution floor: {payload['resolution_floor_pct']:.2f}%")
        print(f"  95% upper bound:  {payload['upper_bound_pct']:.1f}% true failure rate")
        print(f"  for <1% bound:    {req_1pct} zero-failure queries required")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="prompt-eval-power",
        description="Statistical regression gate and power analyzer for RAG prompt suites.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gate = sub.add_parser("gate", help="Run paired McNemar + power gate on RAG prompt templates.")
    p_gate.add_argument("--baseline", choices=sorted(PROMPT_TEMPLATES), default="v1_baseline_grounded")
    p_gate.add_argument("--candidate", choices=sorted(PROMPT_TEMPLATES), default="v2_concise_rewrite")
    p_gate.add_argument("--sample-size", type=int, default=300)
    p_gate.add_argument("--corpus-size", type=int, default=300)
    p_gate.add_argument("--seed", type=int, default=20260511)
    p_gate.add_argument("--alpha", type=float, default=0.05)
    p_gate.add_argument("--target-power", type=float, default=0.80)
    p_gate.add_argument("--target-mde", type=float, default=0.05)
    p_gate.add_argument("--policy", type=str, default=None)
    p_gate.add_argument("--json", action="store_true")

    p_pwr = sub.add_parser("power", help="Compute power, MDE, and required sample size.")
    p_pwr.add_argument("--sample-size", type=int, default=30)
    p_pwr.add_argument("--baseline-rate", type=float, default=0.85)
    p_pwr.add_argument("--drop", type=float, default=0.10)
    p_pwr.add_argument("--discordant-rate", type=float, default=0.15)
    p_pwr.add_argument("--alpha", type=float, default=0.05)
    p_pwr.add_argument("--target-power", type=float, default=0.80)
    p_pwr.add_argument("--json", action="store_true")

    p_zb = sub.add_parser("zero-bound", help="Compute Clopper-Pearson upper bound on failures.")
    p_zb.add_argument("--failures", type=int, default=0)
    p_zb.add_argument("--sample-size", type=int, default=30)
    p_zb.add_argument("--confidence", type=float, default=0.95)
    p_zb.add_argument("--json", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning semantic exit codes (0 pass, 2 reject, 3 underpowered, 1 error)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "gate":
            return _cmd_gate(args)
        if args.command == "power":
            return _cmd_power(args)
        if args.command == "zero-bound":
            return _cmd_zero_bound(args)
        return 1
    except (PolicyValidationError, StatsValidationError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
