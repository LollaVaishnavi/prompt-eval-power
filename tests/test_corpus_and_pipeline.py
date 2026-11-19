"""Property tests for offline RAG corpus, vector retrieval, and LangGraph prompt templates."""

from __future__ import annotations

import unittest

import numpy as np

from prompt_eval_power.config import CorpusConfig
from prompt_eval_power.corpus import BUCKETS, generate_rag_corpus
from prompt_eval_power.pipeline import PROMPT_TEMPLATES, RagStateGraph
from prompt_eval_power.stats import paired_mcnemar_test


class TestCorpusAndPipeline(unittest.TestCase):
    def test_rag_corpus_is_strictly_deterministic_for_same_seed(self) -> None:
        c1 = generate_rag_corpus(CorpusConfig(seed=42, query_count=40))
        c2 = generate_rag_corpus(CorpusConfig(seed=42, query_count=40))
        self.assertEqual(c1.chunk_ids, c2.chunk_ids)
        self.assertEqual([q.gold_doc_id for q in c1.queries], [q.gold_doc_id for q in c2.queries])
        np.testing.assert_array_equal(c1.chunk_matrix, c2.chunk_matrix)

    def test_rag_corpus_balances_all_four_stratified_buckets(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=20260511, query_count=120))
        counts = {b: sum(1 for q in corpus.queries if q.bucket == b) for b in BUCKETS}
        self.assertEqual(counts, {b: 30 for b in BUCKETS})

    def test_chunk_embeddings_are_unit_normalized(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=7, query_count=40))
        norms = np.linalg.norm(corpus.chunk_matrix, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-6)

    def test_corpus_contains_both_gold_and_hard_distractor_chunks(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=7, query_count=40))
        distractors = [c for c in corpus.chunks if c.is_distractor]
        gold = [c for c in corpus.chunks if not c.is_distractor]
        self.assertEqual(len(gold), 80)
        self.assertEqual(len(distractors), 40)

    def test_retrieve_node_returns_exact_top_k_chunks_ordered_by_similarity(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=11, query_count=40))
        graph = RagStateGraph(corpus)
        q = corpus.queries[0]
        top5 = graph.retrieve_node(q, top_k=5)
        self.assertEqual(len(top5), 5)
        sims = [float(np.dot(c.embedding, q.embedding)) for c in top5]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_rerank_node_respects_prompt_context_limit(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=11, query_count=40))
        graph = RagStateGraph(corpus)
        q = corpus.queries[0]
        raw = graph.retrieve_node(q, top_k=5)
        reranked = graph.rerank_node(q, raw, limit=2)
        self.assertEqual(len(reranked), 2)

    def test_concise_rewrite_causes_significant_regressions_over_baseline(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=20260511, query_count=300))
        graph = RagStateGraph(corpus)
        base = graph.evaluate_prompt("v1_baseline_grounded")
        cand = graph.evaluate_prompt("v2_concise_rewrite")
        res = paired_mcnemar_test([o.passed for o in base], [o.passed for o in cand])
        self.assertGreater(res.n_pass_to_fail, 15)
        self.assertLess(res.p_value_regression, 0.001)

    def test_strict_citation_prompt_never_regresses_against_baseline_on_300_queries(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=20260511, query_count=300))
        graph = RagStateGraph(corpus)
        base = graph.evaluate_prompt("v1_baseline_grounded")
        cand = graph.evaluate_prompt("v3_strict_citation")
        res = paired_mcnemar_test([o.passed for o in base], [o.passed for o in cand])
        self.assertEqual(res.n_pass_to_fail, 0)
        self.assertGreater(res.n_fail_to_pass, 0)

    def test_evaluate_prompt_rejects_unknown_prompt_template_name(self) -> None:
        corpus = generate_rag_corpus(CorpusConfig(seed=7, query_count=20))
        graph = RagStateGraph(corpus)
        with self.assertRaises(ValueError):
            graph.evaluate_prompt("v99_unknown")

    def test_evaluate_prompt_rejects_limit_exceeding_corpus_size(self) -> None:
        """Pin boundary bug: requesting limit > len(corpus.queries) must raise ValueError
        rather than silently truncating to a smaller sample size."""
        corpus = generate_rag_corpus(CorpusConfig(seed=7, query_count=40))
        graph = RagStateGraph(corpus)
        with self.assertRaises(ValueError):
            graph.evaluate_prompt("v1_baseline_grounded", limit=100)

    def test_prompt_templates_registry_contains_all_three_expected_variants(self) -> None:
        self.assertEqual(
            set(PROMPT_TEMPLATES.keys()),
            {"v1_baseline_grounded", "v2_concise_rewrite", "v3_strict_citation"},
        )


if __name__ == "__main__":
    unittest.main()
