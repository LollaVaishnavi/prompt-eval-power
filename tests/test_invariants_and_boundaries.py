"""Additional statistical, boundary, and Monte Carlo calibration property tests."""

from __future__ import annotations

import unittest

import numpy as np

from prompt_eval_power.config import CorpusConfig, GatePolicy
from prompt_eval_power.corpus import generate_rag_corpus
from prompt_eval_power.pipeline import RagStateGraph
from prompt_eval_power.power import (
    evaluate_gate,
    minimum_detectable_drop,
    paired_mcnemar_power,
    required_n_paired,
    required_n_unpaired,
    unpaired_power,
)
from prompt_eval_power.stats import (
    clopper_pearson_upper,
    paired_mcnemar_test,
    required_n_for_failure_bound,
    summarize_zero_failure_bound,
    unpaired_regression_pvalue,
)


class TestStatisticalInvariants(unittest.TestCase):
    def test_clopper_pearson_zero_failure_bound_matches_rule_of_three_at_large_n(self) -> None:
        upper = clopper_pearson_upper(0, 300, confidence_level=0.95)
        self.assertAlmostEqual(upper, 3.0 / 300.0, delta=0.0005)

    def test_higher_confidence_level_strictly_raises_zero_failure_upper_bound(self) -> None:
        u90 = clopper_pearson_upper(0, 30, 0.90)
        u95 = clopper_pearson_upper(0, 30, 0.95)
        u99 = clopper_pearson_upper(0, 30, 0.99)
        self.assertLess(u90, u95)
        self.assertLess(u95, u99)

    def test_required_n_for_failure_bound_scales_inversely_with_target_rate(self) -> None:
        n_5pct = required_n_for_failure_bound(0.05, 0.95)
        n_1pct = required_n_for_failure_bound(0.01, 0.95)
        self.assertGreater(n_1pct, 4 * n_5pct)

    def test_resolution_floor_is_strictly_less_than_95_percent_upper_bound(self) -> None:
        for n in (10, 30, 100, 300):
            zb = summarize_zero_failure_bound(0, n, 0.95)
            self.assertLess(zb.resolution_floor, zb.upper_bound_rate)

    def test_paired_mcnemar_net_delta_equals_fail_to_pass_minus_pass_to_fail_over_n(self) -> None:
        rng = np.random.default_rng(123)
        base = rng.random(120) < 0.8
        cand = rng.random(120) < 0.72
        res = paired_mcnemar_test(base, cand)
        expected = (res.n_fail_to_pass - res.n_pass_to_fail) / res.n_total
        self.assertAlmostEqual(res.net_delta, expected, places=12)

    def test_paired_mcnemar_contingency_cells_sum_to_total_queries(self) -> None:
        rng = np.random.default_rng(99)
        base = rng.random(85) < 0.75
        cand = rng.random(85) < 0.70
        res = paired_mcnemar_test(base, cand)
        total = res.n_both_pass + res.n_pass_to_fail + res.n_fail_to_pass + res.n_both_fail
        self.assertEqual(total, 85)

    def test_paired_mcnemar_pvalue_is_bounded_in_unit_interval(self) -> None:
        for n10, n01 in [(0, 10), (5, 5), (10, 0), (1, 1)]:
            base = [True] * n10 + [False] * n01
            cand = [False] * n10 + [True] * n01
            res = paired_mcnemar_test(base, cand)
            self.assertGreaterEqual(res.p_value_regression, 0.0)
            self.assertLessEqual(res.p_value_regression, 1.0)

    def test_paired_mcnemar_power_equals_one_when_delta_equals_discordant_rate_at_large_n(self) -> None:
        pwr = paired_mcnemar_power(200, p_discordant=0.10, delta=0.10, alpha=0.05)
        self.assertGreater(pwr, 0.99)

    def test_unpaired_power_matches_monte_carlo_simulation_within_3_points(self) -> None:
        n, p_base, delta = 80, 0.85, 0.10
        analytic = unpaired_power(n, p_base, delta, alpha=0.05)
        rng = np.random.default_rng(2026)
        trials = 1500
        b_pass = rng.binomial(n, p_base, size=trials)
        c_pass = rng.binomial(n, p_base - delta, size=trials)
        hits = sum(
            1 for bp, cp in zip(b_pass, c_pass)
            if unpaired_regression_pvalue(int(bp), n, int(cp), n) < 0.05
        )
        empirical = hits / trials
        self.assertAlmostEqual(analytic, empirical, delta=0.035)

    def test_required_n_unpaired_grows_quadratically_as_mde_halves(self) -> None:
        n_10pp = required_n_unpaired(0.85, 0.10, alpha=0.05, target_power=0.80)
        n_5pp = required_n_unpaired(0.85, 0.05, alpha=0.05, target_power=0.80)
        self.assertGreater(n_5pp, 3.2 * n_10pp)

    def test_required_n_paired_grows_with_discordant_rate(self) -> None:
        n_low = required_n_paired(0.10, 0.05, alpha=0.05, target_power=0.80)
        n_high = required_n_paired(0.25, 0.05, alpha=0.05, target_power=0.80)
        self.assertLess(n_low, n_high)

    def test_stricter_alpha_increases_required_sample_size(self) -> None:
        n_05 = required_n_paired(0.15, 0.05, alpha=0.05, target_power=0.80)
        n_01 = required_n_paired(0.15, 0.05, alpha=0.01, target_power=0.80)
        self.assertLess(n_05, n_01)

    def test_higher_target_power_increases_required_sample_size(self) -> None:
        n_80 = required_n_unpaired(0.85, 0.05, alpha=0.05, target_power=0.80)
        n_90 = required_n_unpaired(0.85, 0.05, alpha=0.05, target_power=0.90)
        self.assertLess(n_80, n_90)

    def test_mde_at_required_n_is_at_or_below_target_drop(self) -> None:
        n_req = required_n_unpaired(0.85, 0.05, alpha=0.05, target_power=0.80)
        mde = minimum_detectable_drop(n_req, 0.85, alpha=0.05, target_power=0.80, mode="unpaired")
        self.assertLessEqual(mde, 0.051)

    def test_unpaired_pvalue_zero_when_all_baseline_pass_and_zero_candidate_pass(self) -> None:
        self.assertLess(unpaired_regression_pvalue(20, 20, 0, 20), 1e-9)


class TestPipelineAndGateBoundaries(unittest.TestCase):
    def test_rag_pipeline_produces_identical_outcomes_across_repeated_runs(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=20260511, query_count=60))
        g = RagStateGraph(corpus)
        o1 = g.evaluate_prompt("v2_concise_rewrite")
        o2 = g.evaluate_prompt("v2_concise_rewrite")
        self.assertEqual(o1, o2)

    def test_different_corpus_seeds_produce_distinct_query_embeddings(self) -> None:
        c1 = generate_rag_corpus(CorpusConfig(seed=1, query_count=20))
        c2 = generate_rag_corpus(CorpusConfig(seed=2, query_count=20))
        self.assertFalse(np.allclose(c1.chunk_matrix, c2.chunk_matrix))

    def test_failed_outcomes_always_record_non_empty_failure_reason(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=20260511, query_count=120))
        g = RagStateGraph(corpus)
        for name in ("v1_baseline_grounded", "v2_concise_rewrite", "v3_strict_citation"):
            for outcome in g.evaluate_prompt(name):
                if not outcome.passed:
                    self.assertIsNotNone(outcome.failure_reason)
                else:
                    self.assertIsNone(outcome.failure_reason)

    def test_concise_rewrite_regresses_most_severely_on_negation_and_multi_hop_buckets(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=20260511, query_count=300))
        g = RagStateGraph(corpus)
        base = g.evaluate_prompt("v1_baseline_grounded")
        cand = g.evaluate_prompt("v2_concise_rewrite")
        decision = evaluate_gate(
            [o.passed for o in base],
            [o.passed for o in cand],
            buckets=[o.bucket for o in base],
        )
        by_b = {b.bucket: b for b in decision.bucket_results}
        self.assertGreater(by_b["negation_refusal"].pass_to_fail, by_b["direct_lookup"].pass_to_fail)
        self.assertGreater(by_b["multi_hop_policy"].pass_to_fail, by_b["direct_lookup"].pass_to_fail)

    def test_gate_decision_sample_size_deficit_is_zero_whenever_n_exceeds_required(self) -> None:
        base = [True] * 260 + [False] * 40
        cand = [True] * 260 + [False] * 40
        decision = evaluate_gate(base, cand, policy=GatePolicy(target_mde=0.05, target_power=0.80))
        self.assertEqual(decision.sample_size_deficit, 0)
        self.assertEqual(decision.verdict, "PASS")

    def test_gate_decision_reports_exact_deficit_for_small_suite(self) -> None:
        base = [True] * 25 + [False] * 5
        cand = [True] * 25 + [False] * 5
        decision = evaluate_gate(base, cand, policy=GatePolicy(target_mde=0.05, target_power=0.80))
        self.assertEqual(decision.sample_size_deficit, decision.required_n_paired - 30)
        self.assertEqual(decision.verdict, "UNDERPOWERED")

    def test_gate_decision_buckets_are_sorted_alphabetically(self) -> None:
        base = [True, True, False, True]
        cand = [True, True, False, True]
        decision = evaluate_gate(base, cand, buckets=["z_b", "a_b", "m_b", "a_b"])
        self.assertEqual([b.bucket for b in decision.bucket_results], ["a_b", "m_b", "z_b"])

    def test_evaluate_prompt_with_exact_corpus_size_limit_succeeds(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=5, query_count=40))
        g = RagStateGraph(corpus)
        self.assertEqual(len(g.evaluate_prompt("v1_baseline_grounded", limit=40)), 40)

    def test_evaluate_prompt_rejects_non_positive_limit(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=5, query_count=40))
        g = RagStateGraph(corpus)
        with self.assertRaises(ValueError):
            g.evaluate_prompt("v1_baseline_grounded", limit=0)

    def test_rerank_node_boosts_exclusion_chunk_for_refusal_queries(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=20260511, query_count=40))
        g = RagStateGraph(corpus)
        refusal_qs = [q for q in corpus.queries if q.requires_refusal]
        boosted_hits = sum(
            1 for q in refusal_qs
            if any(c.has_exclusion_clause for c in g.rerank_node(q, g.retrieve_node(q, 5), 3))
        )
        self.assertGreater(boosted_hits, len(refusal_qs) // 2)

    def test_unpaired_power_returns_one_when_se_alt_is_tiny(self) -> None:
        self.assertGreater(unpaired_power(5000, 0.85, 0.10, alpha=0.05), 0.9999)


if __name__ == "__main__":
    unittest.main()
