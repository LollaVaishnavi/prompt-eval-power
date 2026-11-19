"""Property tests for statistical power, MDE search, and three-way CI gate decisions."""

from __future__ import annotations

import unittest

from prompt_eval_power.config import GatePolicy
from prompt_eval_power.power import (
    evaluate_gate,
    minimum_detectable_drop,
    paired_mcnemar_power,
    required_n_paired,
    required_n_unpaired,
    unpaired_power,
)
from prompt_eval_power.stats import StatsValidationError


class TestPowerAndMDE(unittest.TestCase):
    def test_unpaired_power_increases_monotonically_with_sample_size(self) -> None:
        powers = [unpaired_power(n, 0.85, 0.10, alpha=0.05) for n in (20, 50, 100, 200, 400)]
        for lo, hi in zip(powers, powers[1:]):
            self.assertLess(lo, hi)

    def test_unpaired_power_increases_monotonically_with_effect_size(self) -> None:
        powers = [unpaired_power(50, 0.85, d, alpha=0.05) for d in (0.03, 0.05, 0.10, 0.20)]
        for lo, hi in zip(powers, powers[1:]):
            self.assertLess(lo, hi)

    def test_required_n_unpaired_achieves_target_power(self) -> None:
        n_req = required_n_unpaired(0.85, 0.05, alpha=0.05, target_power=0.80)
        self.assertGreaterEqual(unpaired_power(n_req, 0.85, 0.05, alpha=0.05), 0.80)
        self.assertLess(unpaired_power(n_req - 1, 0.85, 0.05, alpha=0.05), 0.80)

    def test_paired_mcnemar_power_increases_with_n_and_decreases_with_discordance(self) -> None:
        p_small_n = paired_mcnemar_power(50, 0.15, 0.05)
        p_large_n = paired_mcnemar_power(200, 0.15, 0.05)
        self.assertLess(p_small_n, p_large_n)

        p_low_disc = paired_mcnemar_power(150, 0.10, 0.05)
        p_high_disc = paired_mcnemar_power(150, 0.30, 0.05)
        self.assertGreater(p_low_disc, p_high_disc)

    def test_required_n_paired_achieves_target_power(self) -> None:
        n_req = required_n_paired(0.15, 0.05, alpha=0.05, target_power=0.80)
        self.assertGreaterEqual(paired_mcnemar_power(n_req, 0.15, 0.05, alpha=0.05), 0.80)
        self.assertLess(paired_mcnemar_power(n_req - 1, 0.15, 0.05, alpha=0.05), 0.80)

    def test_paired_testing_requires_far_fewer_queries_than_unpaired_at_realistic_discordance(self) -> None:
        n_unp = required_n_unpaired(0.85, 0.05, alpha=0.05, target_power=0.80)
        n_pair = required_n_paired(0.15, 0.05, alpha=0.05, target_power=0.80)
        self.assertLess(n_pair, n_unp * 0.6)

    def test_minimum_detectable_drop_shrinks_as_sample_size_grows(self) -> None:
        mdes = [minimum_detectable_drop(n, 0.85, mode="unpaired") for n in (30, 80, 200, 500)]
        for hi, lo in zip(mdes, mdes[1:]):
            self.assertGreater(hi, lo)

    def test_minimum_detectable_drop_works_in_paired_mode(self) -> None:
        mde_30 = minimum_detectable_drop(30, 0.85, mode="paired", p_discordant=0.15)
        mde_300 = minimum_detectable_drop(300, 0.85, mode="paired", p_discordant=0.15)
        self.assertGreater(mde_30, mde_300)

    def test_power_functions_reject_invalid_inputs(self) -> None:
        with self.assertRaises(StatsValidationError):
            unpaired_power(1, 0.85, 0.10)
        with self.assertRaises(StatsValidationError):
            unpaired_power(30, 1.0, 0.10)
        with self.assertRaises(StatsValidationError):
            required_n_unpaired(0.5, 0.6)
        with self.assertRaises(StatsValidationError):
            paired_mcnemar_power(1, 0.15, 0.05)
        with self.assertRaises(StatsValidationError):
            paired_mcnemar_power(50, 0.04, 0.05)
        with self.assertRaises(StatsValidationError):
            required_n_paired(0.04, 0.05)
        with self.assertRaises(StatsValidationError):
            minimum_detectable_drop(30, 0.85, mode="invalid")


class TestGateDecisions(unittest.TestCase):
    def test_small_suite_with_no_significant_drop_returns_underpowered_exit_code_3(self) -> None:
        base = [True] * 25 + [False] * 5
        cand = [True] * 24 + [False] * 6
        decision = evaluate_gate(base, cand, policy=GatePolicy(target_mde=0.05, target_power=0.80))
        self.assertEqual(decision.verdict, "UNDERPOWERED")
        self.assertEqual(decision.exit_code, 3)
        self.assertGreater(decision.sample_size_deficit, 0)

    def test_large_powered_suite_with_zero_regressions_returns_pass_exit_code_0(self) -> None:
        base = [True] * 240 + [False] * 60
        cand = [True] * 245 + [False] * 55
        decision = evaluate_gate(base, cand, policy=GatePolicy(target_mde=0.05, target_power=0.80))
        self.assertEqual(decision.verdict, "PASS")
        self.assertEqual(decision.exit_code, 0)
        self.assertEqual(decision.sample_size_deficit, 0)

    def test_significant_overall_regression_returns_reject_exit_code_2(self) -> None:
        base = [True] * 240 + [False] * 60
        cand = [True] * 210 + [False] * 90
        decision = evaluate_gate(base, cand, policy=GatePolicy(target_mde=0.05, target_power=0.80))
        self.assertEqual(decision.verdict, "REJECT_REGRESSION")
        self.assertEqual(decision.exit_code, 2)

    def test_bucket_breach_triggers_reject_even_when_overall_pass_rate_is_flat(self) -> None:
        # Bucket A improves by +10, Bucket B drops by -10 (significant breach on Bucket B)
        base_a = [False] * 10 + [True] * 40
        cand_a = [True] * 10 + [True] * 40
        base_b = [True] * 50
        cand_b = [False] * 10 + [True] * 40
        decision = evaluate_gate(
            base_a + base_b,
            cand_a + cand_b,
            buckets=["bucket_a"] * 50 + ["bucket_b"] * 50,
            policy=GatePolicy(max_bucket_drop=0.10),
        )
        self.assertEqual(decision.verdict, "REJECT_REGRESSION")
        self.assertEqual(decision.exit_code, 2)
        breached = [b.bucket for b in decision.bucket_results if b.breached]
        self.assertEqual(breached, ["bucket_b"])

    def test_evaluate_gate_rejects_mismatched_bucket_length(self) -> None:
        with self.assertRaises(StatsValidationError):
            evaluate_gate([True, False], [True, False], buckets=["only_one"])


if __name__ == "__main__":
    unittest.main()
