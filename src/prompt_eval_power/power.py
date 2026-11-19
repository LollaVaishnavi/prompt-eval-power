"""Statistical power, minimum detectable effect (MDE), and CI gate decision logic.

Exists to answer whether a prompt evaluation suite is large enough to detect a
target regression drop, separating genuine safety (PASS) from statistical
blindness (UNDERPOWERED, exit code 3) and proven regressions (REJECT, exit 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from prompt_eval_power.config import GatePolicy
from prompt_eval_power.stats import McNemarResult, StatsValidationError, paired_mcnemar_test


@dataclass(frozen=True)
class BucketResult:
    """Per-bucket paired regression summary."""

    bucket: str
    n_queries: int
    baseline_pass_rate: float
    candidate_pass_rate: float
    net_delta: float
    pass_to_fail: int
    fail_to_pass: int
    p_value_regression: float
    breached: bool


@dataclass(frozen=True)
class GateDecision:
    """Structured verdict for a prompt evaluation CI gate."""

    verdict: str
    exit_code: int
    overall: McNemarResult
    required_n_paired: int
    required_n_unpaired: int
    sample_size_deficit: int
    achieved_power_at_target_mde: float
    mde_at_current_n: float
    bucket_results: tuple[BucketResult, ...]


def unpaired_power(
    n_per_arm: int,
    p_base: float,
    delta: float,
    alpha: float = 0.05,
) -> float:
    """One-sided statistical power for detecting a drop of `delta` in unpaired evaluation."""
    if n_per_arm < 2:
        raise StatsValidationError("n_per_arm must be >= 2")
    if not (0.0 < p_base < 1.0) or not (0.0 < delta < p_base):
        raise StatsValidationError("Invalid p_base or delta for power calculation")
    p_cand = p_base - delta
    p_bar = 0.5 * (p_base + p_cand)
    se_null = math.sqrt(2.0 * p_bar * (1.0 - p_bar) / n_per_arm)
    se_alt = math.sqrt((p_base * (1.0 - p_base) + p_cand * (1.0 - p_cand)) / n_per_arm)
    if se_alt <= 1e-12:
        return 1.0
    z_alpha = float(norm.ppf(1.0 - alpha))
    z_power = (delta - z_alpha * se_null) / se_alt
    return float(norm.cdf(z_power))


def required_n_unpaired(
    p_base: float,
    delta: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> int:
    """Minimum queries per arm required to detect `delta` drop at `target_power` (unpaired)."""
    if not (0.0 < p_base < 1.0) or not (0.0 < delta < p_base):
        raise StatsValidationError("Invalid p_base or delta")
    p_cand = p_base - delta
    p_bar = 0.5 * (p_base + p_cand)
    z_a = float(norm.ppf(1.0 - alpha))
    z_b = float(norm.ppf(target_power))
    num = z_a * math.sqrt(2.0 * p_bar * (1.0 - p_bar)) + z_b * math.sqrt(
        p_base * (1.0 - p_base) + p_cand * (1.0 - p_cand)
    )
    return int(math.ceil((num / delta) ** 2))


def paired_mcnemar_power(
    n_queries: int,
    p_discordant: float,
    delta: float,
    alpha: float = 0.05,
) -> float:
    """One-sided Connor (1987) analytical power for paired McNemar regression detection.

    Args:
        n_queries: Number of paired queries evaluated on both prompts.
        p_discordant: Total discordant probability (p10 + p01), must be >= delta.
        delta: Net regression drop (p10 - p01) > 0.
        alpha: One-sided significance threshold.
    """
    if n_queries < 2:
        raise StatsValidationError("n_queries must be >= 2")
    if not (0.0 < delta <= p_discordant <= 1.0):
        raise StatsValidationError(
            f"Require 0 < delta <= p_discordant <= 1, got delta={delta}, disc={p_discordant}"
        )
    z_a = float(norm.ppf(1.0 - alpha))
    var_alt = p_discordant - delta * delta
    if var_alt <= 1e-12:
        return 1.0
    num = delta * math.sqrt(n_queries) - z_a * math.sqrt(p_discordant)
    return float(norm.cdf(num / math.sqrt(var_alt)))


def required_n_paired(
    p_discordant: float,
    delta: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> int:
    """Minimum paired queries needed to detect drop `delta` at `target_power`."""
    if not (0.0 < delta <= p_discordant <= 1.0):
        raise StatsValidationError("Require 0 < delta <= p_discordant <= 1")
    z_a = float(norm.ppf(1.0 - alpha))
    z_b = float(norm.ppf(target_power))
    num = z_a * math.sqrt(p_discordant) + z_b * math.sqrt(max(0.0, p_discordant - delta * delta))
    return int(math.ceil((num / delta) ** 2))


def minimum_detectable_drop(
    n_queries: int,
    p_base: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    mode: str = "unpaired",
    p_discordant: float | None = None,
) -> float:
    """Binary search for minimum detectable regression drop (in [0, p_base]) at sample size n."""
    if mode not in {"unpaired", "paired"}:
        raise StatsValidationError(f"Unsupported mode '{mode}'")
    lo, hi = 0.001, min(p_base - 0.001, 0.99)
    if mode == "paired":
        disc = p_discordant if p_discordant is not None else min(0.25, 1.0 - 0.5 * p_base)
        hi = min(hi, disc)

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mode == "unpaired":
            pwr = unpaired_power(n_queries, p_base, mid, alpha=alpha)
        else:
            disc_eff = max(mid, p_discordant if p_discordant is not None else 0.15)
            pwr = paired_mcnemar_power(n_queries, disc_eff, mid, alpha=alpha)
        if pwr >= target_power:
            hi = mid
        else:
            lo = mid
    return hi


def evaluate_gate(
    baseline_pass: np.ndarray | list[bool],
    candidate_pass: np.ndarray | list[bool],
    buckets: list[str] | None = None,
    policy: GatePolicy | None = None,
) -> GateDecision:
    """Evaluate a prompt change with paired McNemar testing and explicit power gating.

    Exit codes:
        0: PASS (sufficiently powered and no statistically significant regression).
        2: REJECT_REGRESSION (statistically significant overall or bucket regression).
        3: UNDERPOWERED (no significant regression, but suite lacks power at target MDE).
    """
    active = policy or GatePolicy()
    base = np.asarray(baseline_pass, dtype=bool)
    cand = np.asarray(candidate_pass, dtype=bool)
    overall = paired_mcnemar_test(base, cand)

    p_base_eff = min(max(overall.baseline_pass_rate, active.target_mde + 0.02), 0.98)
    # Use observed discordance floored at max(target_mde * 1.5, 0.10) for conservative planning
    p_disc_plan = min(0.95, max(overall.discordant_rate, active.target_mde * 1.5, 0.10))

    req_paired = required_n_paired(
        p_discordant=p_disc_plan,
        delta=active.target_mde,
        alpha=active.alpha,
        target_power=active.target_power,
    )
    req_unpaired = required_n_unpaired(
        p_base=p_base_eff,
        delta=active.target_mde,
        alpha=active.alpha,
        target_power=active.target_power,
    )
    deficit = max(0, req_paired - overall.n_total)
    achieved_pwr = paired_mcnemar_power(
        n_queries=overall.n_total,
        p_discordant=p_disc_plan,
        delta=active.target_mde,
        alpha=active.alpha,
    )
    mde_curr = minimum_detectable_drop(
        n_queries=overall.n_total,
        p_base=p_base_eff,
        alpha=active.alpha,
        target_power=active.target_power,
        mode="paired",
        p_discordant=p_disc_plan,
    )

    bucket_list = buckets if buckets is not None else ["all"] * overall.n_total
    if len(bucket_list) != overall.n_total:
        raise StatsValidationError("buckets length must match baseline_pass length")
    bucket_arr = np.asarray(bucket_list)
    b_results: list[BucketResult] = []
    any_bucket_breach = False
    for b_name in sorted(set(bucket_list)):
        mask = bucket_arr == b_name
        res = paired_mcnemar_test(base[mask], cand[mask])
        breached = (res.net_delta <= -active.max_bucket_drop) and (
            res.p_value_regression < active.alpha
        )
        if breached:
            any_bucket_breach = True
        b_results.append(
            BucketResult(
                bucket=b_name,
                n_queries=res.n_total,
                baseline_pass_rate=res.baseline_pass_rate,
                candidate_pass_rate=res.candidate_pass_rate,
                net_delta=res.net_delta,
                pass_to_fail=res.n_pass_to_fail,
                fail_to_pass=res.n_fail_to_pass,
                p_value_regression=res.p_value_regression,
                breached=breached,
            )
        )

    if overall.p_value_regression < active.alpha or any_bucket_breach:
        verdict, exit_code = "REJECT_REGRESSION", 2
    elif deficit > 0:
        verdict, exit_code = "UNDERPOWERED", 3
    else:
        verdict, exit_code = "PASS", 0

    return GateDecision(
        verdict=verdict,
        exit_code=exit_code,
        overall=overall,
        required_n_paired=req_paired,
        required_n_unpaired=req_unpaired,
        sample_size_deficit=deficit,
        achieved_power_at_target_mde=achieved_pwr,
        mde_at_current_n=mde_curr,
        bucket_results=tuple(b_results),
    )
