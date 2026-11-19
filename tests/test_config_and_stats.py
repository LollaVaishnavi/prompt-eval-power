"""Property tests for GatePolicy, CorpusConfig, Clopper-Pearson bounds, and McNemar tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prompt_eval_power.config import CorpusConfig, GatePolicy, PolicyValidationError
from prompt_eval_power.stats import (
    StatsValidationError,
    clopper_pearson_upper,
    paired_mcnemar_test,
    required_n_for_failure_bound,
    summarize_zero_failure_bound,
    unpaired_regression_pvalue,
)


class TestConfigValidation(unittest.TestCase):
    def test_gate_policy_round_trips_through_json(self) -> None:
        policy = GatePolicy(
            alpha=0.01,
            target_power=0.85,
            target_mde=0.04,
            confidence_level=0.95,
            max_bucket_drop=0.08,
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "gate.json"
            p.write_text(json.dumps(policy.to_dict()), encoding="utf-8")
            self.assertEqual(GatePolicy.from_json(p), policy)

    def test_gate_policy_rejects_out_of_range_parameters(self) -> None:
        with self.assertRaises(PolicyValidationError):
            GatePolicy(alpha=0.0)
        with self.assertRaises(PolicyValidationError):
            GatePolicy(alpha=0.5)
        with self.assertRaises(PolicyValidationError):
            GatePolicy(target_power=0.4)
        with self.assertRaises(PolicyValidationError):
            GatePolicy(target_mde=0.001)
        with self.assertRaises(PolicyValidationError):
            GatePolicy(confidence_level=0.5)
        with self.assertRaises(PolicyValidationError):
            GatePolicy(max_bucket_drop=0.0)

    def test_corpus_config_rejects_invalid_dimensions_or_counts(self) -> None:
        with self.assertRaises(PolicyValidationError):
            CorpusConfig(query_count=10)
        with self.assertRaises(PolicyValidationError):
            CorpusConfig(embedding_dim=4)
        with self.assertRaises(PolicyValidationError):
            CorpusConfig(top_k=0)


class TestClopperPearsonAndZeroFailureBounds(unittest.TestCase):
    def test_zero_of_30_failures_yields_approx_9_5_percent_upper_bound(self) -> None:
        upper = clopper_pearson_upper(0, 30, confidence_level=0.95)
        self.assertAlmostEqual(upper, 1.0 - 0.05 ** (1.0 / 30.0), places=10)
        self.assertAlmostEqual(upper * 100.0, 9.503, places=2)

    def test_all_failures_returns_upper_bound_of_one(self) -> None:
        self.assertEqual(clopper_pearson_upper(10, 10, 0.95), 1.0)

    def test_nonzero_failures_strictly_increase_upper_bound(self) -> None:
        u0 = clopper_pearson_upper(0, 50, 0.95)
        u1 = clopper_pearson_upper(1, 50, 0.95)
        u5 = clopper_pearson_upper(5, 50, 0.95)
        self.assertLess(u0, u1)
        self.assertLess(u1, u5)

    def test_larger_sample_size_strictly_tightens_zero_failure_bound(self) -> None:
        bounds = [clopper_pearson_upper(0, n, 0.95) for n in (10, 30, 100, 300)]
        for high, low in zip(bounds, bounds[1:]):
            self.assertGreater(high, low)

    def test_required_n_for_1_percent_bound_at_95_confidence_is_299(self) -> None:
        n_req = required_n_for_failure_bound(0.01, confidence_level=0.95)
        self.assertEqual(n_req, 299)
        self.assertLessEqual(clopper_pearson_upper(0, n_req, 0.95), 0.01)
        self.assertGreater(clopper_pearson_upper(0, n_req - 1, 0.95), 0.01)

    def test_summarize_zero_failure_bound_reports_exact_resolution_floor(self) -> None:
        summary = summarize_zero_failure_bound(0, 40, confidence_level=0.95)
        self.assertAlmostEqual(summary.resolution_floor, 0.025)
        self.assertEqual(summary.observed_failure_rate, 0.0)

    def test_clopper_pearson_and_required_n_reject_invalid_inputs(self) -> None:
        with self.assertRaises(StatsValidationError):
            clopper_pearson_upper(0, 0)
        with self.assertRaises(StatsValidationError):
            clopper_pearson_upper(-1, 10)
        with self.assertRaises(StatsValidationError):
            clopper_pearson_upper(11, 10)
        with self.assertRaises(StatsValidationError):
            clopper_pearson_upper(0, 10, confidence_level=0.5)
        with self.assertRaises(StatsValidationError):
            required_n_for_failure_bound(0.0)
        with self.assertRaises(StatsValidationError):
            required_n_for_failure_bound(0.05, confidence_level=1.0)


class TestMcNemarAndUnpairedTests(unittest.TestCase):
    def test_paired_mcnemar_counts_all_four_contingency_cells_exactly(self) -> None:
        base = [True, True, True, False, False, False]
        cand = [True, False, False, True, False, False]
        res = paired_mcnemar_test(base, cand)
        self.assertEqual(res.n_total, 6)
        self.assertEqual(res.n_both_pass, 1)
        self.assertEqual(res.n_pass_to_fail, 2)
        self.assertEqual(res.n_fail_to_pass, 1)
        self.assertEqual(res.n_both_fail, 2)
        self.assertAlmostEqual(res.discordant_rate, 0.5)

    def test_paired_mcnemar_returns_pvalue_one_when_zero_discordant_pairs(self) -> None:
        outcomes = [True, False, True, True]
        res = paired_mcnemar_test(outcomes, outcomes)
        self.assertEqual(res.p_value_regression, 1.0)
        self.assertEqual(res.net_delta, 0.0)

    def test_paired_mcnemar_detects_strong_regression_with_small_pvalue(self) -> None:
        base = [True] * 25 + [False] * 5
        cand = [True] * 13 + [False] * 17
        res = paired_mcnemar_test(base, cand)
        self.assertEqual(res.n_pass_to_fail, 12)
        self.assertEqual(res.n_fail_to_pass, 0)
        self.assertLess(res.p_value_regression, 0.001)

    def test_paired_mcnemar_rejects_empty_or_mismatched_sequences(self) -> None:
        with self.assertRaises(StatsValidationError):
            paired_mcnemar_test([], [])
        with self.assertRaises(StatsValidationError):
            paired_mcnemar_test([True, False], [True])

    def test_unpaired_regression_pvalue_is_small_for_large_drop_and_large_for_gain(self) -> None:
        p_drop = unpaired_regression_pvalue(90, 100, 70, 100)
        p_gain = unpaired_regression_pvalue(70, 100, 90, 100)
        self.assertLess(p_drop, 0.001)
        self.assertGreater(p_gain, 0.99)

    def test_unpaired_regression_pvalue_handles_degenerate_and_invalid_inputs(self) -> None:
        self.assertEqual(unpaired_regression_pvalue(10, 10, 10, 10), 1.0)
        with self.assertRaises(StatsValidationError):
            unpaired_regression_pvalue(0, 0, 0, 10)
        with self.assertRaises(StatsValidationError):
            unpaired_regression_pvalue(11, 10, 5, 10)


if __name__ == "__main__":
    unittest.main()
