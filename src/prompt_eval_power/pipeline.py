"""Deterministic multi-step RAG state-graph pipeline for prompt engineering audits.

Exists to model a LangGraph-style retrieve -> rerank -> prompt_template ->
verify_grounded_answer workflow offline, so prompt edits can be evaluated on
identical retrieved contexts with exact paired pass/fail outcomes per query.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prompt_eval_power.corpus import DocumentChunk, GoldenQuery, RagCorpus


@dataclass(frozen=True)
class PromptTemplateSpec:
    """Declarative prompt engineering configuration for a RAG answer node."""

    name: str
    system_prompt: str
    enforces_refusal_guardrail: bool
    enforces_multi_hop_checklist: bool
    enforces_disambiguation: bool
    max_context_chunks: int
    token_budget: int


PROMPT_TEMPLATES: dict[str, PromptTemplateSpec] = {
    "v1_baseline_grounded": PromptTemplateSpec(
        name="v1_baseline_grounded",
        system_prompt=(
            "Answer strictly from retrieved policy chunks. Cite the source doc_id. "
            "If an item is excluded or unsupported, refuse explicitly. "
            "For multi-hop return rules, verify category, tier, and day window."
        ),
        enforces_refusal_guardrail=True,
        enforces_multi_hop_checklist=True,
        enforces_disambiguation=True,
        max_context_chunks=3,
        token_budget=420,
    ),
    "v2_concise_rewrite": PromptTemplateSpec(
        name="v2_concise_rewrite",
        system_prompt=(
            "Give a short, helpful answer using the top retrieved chunk. "
            "Keep responses concise under two sentences."
        ),
        enforces_refusal_guardrail=False,
        enforces_multi_hop_checklist=False,
        enforces_disambiguation=True,
        max_context_chunks=2,
        token_budget=180,
    ),
    "v3_strict_citation": PromptTemplateSpec(
        name="v3_strict_citation",
        system_prompt=(
            "Ground every sentence in retrieved doc_ids. Explicitly refuse unsupported "
            "requests, verify all multi-hop conditions, and list conflicting variants."
        ),
        enforces_refusal_guardrail=True,
        enforces_multi_hop_checklist=True,
        enforces_disambiguation=True,
        max_context_chunks=3,
        token_budget=460,
    ),
}


@dataclass(frozen=True)
class QueryOutcome:
    """Per-query state-graph execution trace and binary pass/fail grade."""

    query_id: str
    bucket: str
    retrieved_gold: bool
    passed: bool
    failure_reason: str | None


class RagStateGraph:
    """Offline LangGraph-style state machine: retrieve -> rerank -> prompt -> grade."""

    def __init__(self, corpus: RagCorpus) -> None:
        self.corpus = corpus
        self._chunk_by_id: dict[str, DocumentChunk] = {
            c.doc_id: c for c in corpus.chunks
        }

    def retrieve_node(self, query: GoldenQuery, top_k: int) -> list[DocumentChunk]:
        """Cosine-similarity top-k retrieval over the normalized chunk matrix."""
        scores = self.corpus.chunk_matrix @ query.embedding
        order = np.argsort(-scores, kind="mergesort")[:top_k]
        return [self.corpus.chunks[int(i)] for i in order]

    def rerank_node(
        self,
        query: GoldenQuery,
        candidates: list[DocumentChunk],
        limit: int,
    ) -> list[DocumentChunk]:
        """Deterministic reranker boosting policy/exclusion chunks matching query intent."""
        def score_chunk(chunk: DocumentChunk) -> tuple[float, str]:
            base_sim = float(np.dot(chunk.embedding, query.embedding))
            bonus = 0.0
            if query.requires_refusal and chunk.has_exclusion_clause:
                bonus += 0.08
            if query.requires_multi_hop and chunk.has_multi_hop_rule:
                bonus += 0.08
            return (base_sim + bonus, chunk.doc_id)

        ranked = sorted(candidates, key=score_chunk, reverse=True)
        return ranked[:limit]

    def grade_node(
        self,
        query: GoldenQuery,
        context: list[DocumentChunk],
        prompt: PromptTemplateSpec,
    ) -> QueryOutcome:
        """Deterministically grade grounded answer compliance for the prompt spec."""
        doc_ids = {c.doc_id for c in context}
        retrieved_gold = query.gold_doc_id in doc_ids
        if not retrieved_gold:
            return QueryOutcome(
                query_id=query.query_id,
                bucket=query.bucket,
                retrieved_gold=False,
                passed=False,
                failure_reason="retrieval_miss",
            )

        # Deterministic difficulty thresholds per prompt engineering rule
        if query.requires_refusal:
            if not prompt.enforces_refusal_guardrail and query.difficulty > 0.58:
                return QueryOutcome(
                    query.query_id, query.bucket, True, False, "hallucinated_unsupported_claim"
                )
            if prompt.name == "v1_baseline_grounded" and query.difficulty > 0.90:
                return QueryOutcome(
                    query.query_id, query.bucket, True, False, "weak_refusal_phrasing"
                )
            if prompt.name == "v3_strict_citation" and query.difficulty > 0.94:
                return QueryOutcome(
                    query.query_id, query.bucket, True, False, "weak_refusal_phrasing"
                )
            return QueryOutcome(query.query_id, query.bucket, True, True, None)

        if query.requires_multi_hop:
            if not prompt.enforces_multi_hop_checklist and query.difficulty > 0.64:
                return QueryOutcome(
                    query.query_id, query.bucket, True, False, "dropped_multi_hop_condition"
                )
            if prompt.name == "v1_baseline_grounded" and query.difficulty > 0.88:
                return QueryOutcome(
                    query.query_id, query.bucket, True, False, "partial_policy_synthesis"
                )
            if prompt.name == "v3_strict_citation" and query.difficulty > 0.92:
                return QueryOutcome(
                    query.query_id, query.bucket, True, False, "partial_policy_synthesis"
                )
            return QueryOutcome(query.query_id, query.bucket, True, True, None)

        if query.requires_disambiguation:
            cutoff = 0.84 if prompt.name == "v2_concise_rewrite" else (
                0.86 if prompt.name == "v1_baseline_grounded" else 0.88
            )
            if query.difficulty > cutoff:
                return QueryOutcome(
                    query.query_id, query.bucket, True, False, "missed_variant_disambiguation"
                )
            return QueryOutcome(query.query_id, query.bucket, True, True, None)

        # direct_lookup bucket
        cutoff = 0.93 if prompt.name != "v3_strict_citation" else 0.94
        if query.difficulty > cutoff:
            return QueryOutcome(query.query_id, query.bucket, True, False, "extraction_miss")
        return QueryOutcome(query.query_id, query.bucket, True, True, None)

    def evaluate_prompt(
        self,
        prompt_name: str,
        limit: int | None = None,
    ) -> list[QueryOutcome]:
        """Run the full retrieve -> rerank -> grade graph over the corpus."""
        if prompt_name not in PROMPT_TEMPLATES:
            raise ValueError(
                f"Unknown prompt '{prompt_name}'. Available: {sorted(PROMPT_TEMPLATES)}"
            )
        prompt = PROMPT_TEMPLATES[prompt_name]
        queries = self.corpus.queries[:limit] if limit is not None else self.corpus.queries
        outcomes: list[QueryOutcome] = []
        for q in queries:
            raw = self.retrieve_node(q, top_k=5)
            ctx = self.rerank_node(q, raw, limit=prompt.max_context_chunks)
            outcomes.append(self.grade_node(q, ctx, prompt))
        return outcomes
