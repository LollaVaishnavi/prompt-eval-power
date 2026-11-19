"""Exact binomial, Clopper-Pearson, unpaired proportion, and paired McNemar tests.

Exists so prompt regression gates use exact finite-sample statistics rather than
naive pass-rate differences that silently treat sampling noise as a signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, binomtest, norm


class StatsValidationError(ValueError):
    """Raised when statistical test inputs have invalid bounds or mismatched lengths."""


@dataclass(frozen=True)
class McNemarResult:
    """Exact paired comparison between baseline and candidate prompt outcomes."""

    n_total: int
    n_both_pass: int
    n_pass_to_fail: int
    n_fail_to_pass: int
    n_both_fail: int
    baseline_pass_rate: float
    candidate_pass_rate: float
    net_delta: float
    discordant_rate: float
    p_value_regression: float


@dataclass(frozen=True)
class ZeroFailureBound:
    """Finite-sample resolution floor when zero (or k) failures are observed."""

    n_queries: int
    observed_failures: int
    observed_failure_rate: float
    upper_bound_rate: float
    confidence_level: float
    resolution_floor: float


def clopper_pearson_upper(
    failures: int,
    n_queries: int,
    confidence_level: float = 0.95,
) -> float:
    """Return the exact one-sided Clopper-Pearson upper bound on true failure rate."""
    if n_queries < 1:
        raise StatsValidationError(f"n_queries must be >= 1, got {n_queries}")
    if failures < 0 or failures > n_queries:
        raise StatsValidationError(
            f"failures must be in [0, {n_queries}], got {failures}"
        )
    if not (0.5 < confidence_level < 1.0):
        raise StatsValidationError("confidence_level must be in (0.5, 1.0)")
    if failures == n_queries:
        return 1.0
    if failures == 0:
        return float(1.0 - (1.0 - confidence_level) ** (1.0 / n_queries))
    return float(beta.ppf(confidence_level, failures + 1, n_queries - failures))


def summarize_zero_failure_bound(
    failures: int,
    n_queries: int,
    confidence_level: float = 0.95,
) -> ZeroFailureBound:
    """Compute observed failure rate, 1/n resolution floor, and Clopper-Pearson upper bound."""
    upper = clopper_pearson_upper(failures, n_queries, confidence_level=confidence_level)
    return ZeroFailureBound(
        n_queries=n_queries,
        observed_failures=failures,
        observed_failure_rate=failures / n_queries,
        upper_bound_rate=upper,
        confidence_level=confidence_level,
        resolution_floor=1.0 / n_queries,
    )


def required_n_for_failure_bound(
    max_failure_rate: float,
    confidence_level: float = 0.95,
) -> int:
    """Return minimum zero-failure queries needed to certify failure rate <= max_failure_rate."""
    if not (0.0 < max_failure_rate < 1.0):
        raise StatsValidationError("max_failure_rate must be in (0, 1)")
    if not (0.5 < confidence_level < 1.0):
        raise StatsValidationError("confidence_level must be in (0.5, 1.0)")
    return int(math.ceil(math.log(1.0 - confidence_level) / math.log(1.0 - max_failure_rate)))


def paired_mcnemar_test(
    baseline_pass: np.ndarray | list[bool] | list[int],
    candidate_pass: np.ndarray | list[bool] | list[int],
) -> McNemarResult:
    """Run an exact one-sided McNemar test for prompt regression (n10 > n01)."""
    base = np.asarray(baseline_pass, dtype=bool)
    cand = np.asarray(candidate_pass, dtype=bool)
    if base.ndim != 1 or cand.ndim != 1 or base.shape != cand.shape or base.size == 0:
        raise StatsValidationError(
            "baseline_pass and candidate_pass must be non-empty 1D sequences of equal length"
        )

    n_total = int(base.size)
    n11 = int(np.sum(base & cand))
    n10 = int(np.sum(base & (~cand)))
    n01 = int(np.sum((~base) & cand))
    n00 = int(np.sum((~base) & (~cand)))

    n_disc = n10 + n01
    if n_disc == 0:
        p_val = 1.0
    else:
        p_val = float(binomtest(n10, n_disc, p=0.5, alternative="greater").pvalue)

    base_rate = float(np.mean(base))
    cand_rate = float(np.mean(cand))
    return McNemarResult(
        n_total=n_total,
        n_both_pass=n11,
        n_pass_to_fail=n10,
        n_fail_to_pass=n01,
        n_both_fail=n00,
        baseline_pass_rate=base_rate,
        candidate_pass_rate=cand_rate,
        net_delta=cand_rate - base_rate,
        discordant_rate=n_disc / n_total,
        p_value_regression=p_val,
    )


def unpaired_regression_pvalue(
    baseline_passes: int,
    n_base: int,
    candidate_passes: int,
    n_cand: int,
) -> float:
    """One-sided pooled z-test p-value for H1: p_candidate < p_baseline."""
    if n_base < 1 or n_cand < 1:
        raise StatsValidationError("Sample sizes must be >= 1")
    if not (0 <= baseline_passes <= n_base and 0 <= candidate_passes <= n_cand):
        raise StatsValidationError("Pass counts must lie within [0, n]")
    p1 = baseline_passes / n_base
    p2 = candidate_passes / n_cand
    pooled = (baseline_passes + candidate_passes) / (n_base + n_cand)
    se = math.sqrt(max(0.0, pooled * (1.0 - pooled) * (1.0 / n_base + 1.0 / n_cand)))
    if se <= 1e-12:
        return 1.0 if p2 >= p1 else 0.0
    z_stat = (p1 - p2) / se
    return float(1.0 - norm.cdf(z_stat))
