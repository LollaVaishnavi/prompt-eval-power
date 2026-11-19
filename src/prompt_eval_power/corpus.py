"""Deterministic offline e-commerce RAG corpus and cosine vector index.

Exists to provide a seeded 300-query golden dataset spanning four stratified
query buckets (direct_lookup, multi_hop_policy, negation_refusal, ambiguous_edge)
and a pure-NumPy cosine vector store mirroring Pinecone retrieval offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prompt_eval_power.config import CorpusConfig

BUCKETS: tuple[str, ...] = (
    "direct_lookup",
    "multi_hop_policy",
    "negation_refusal",
    "ambiguous_edge",
)


@dataclass(frozen=True)
class DocumentChunk:
    """Single indexed knowledge-base chunk in the offline vector store."""

    doc_id: str
    bucket: str
    is_distractor: bool
    has_exclusion_clause: bool
    has_multi_hop_rule: bool
    embedding: np.ndarray


@dataclass(frozen=True)
class GoldenQuery:
    """Labeled evaluation query with ground-truth target chunk and constraints."""

    query_id: str
    bucket: str
    gold_doc_id: str
    requires_refusal: bool
    requires_multi_hop: bool
    requires_disambiguation: bool
    difficulty: float
    embedding: np.ndarray


@dataclass(frozen=True)
class RagCorpus:
    """Complete seeded corpus containing vector-indexed chunks and golden queries."""

    chunks: tuple[DocumentChunk, ...]
    queries: tuple[GoldenQuery, ...]
    chunk_matrix: np.ndarray
    chunk_ids: tuple[str, ...]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


def generate_rag_corpus(config: CorpusConfig | None = None) -> RagCorpus:
    """Generate a deterministic vector-indexed corpus and stratified golden query set."""
    cfg = config or CorpusConfig()
    rng = np.random.default_rng(cfg.seed)

    # Four orthogonal-ish topic centroids for the four query buckets
    centroids = {
        bucket: _normalize(rng.normal(size=cfg.embedding_dim))
        for bucket in BUCKETS
    }

    chunks: list[DocumentChunk] = []
    gold_by_bucket: dict[str, list[DocumentChunk]] = {b: [] for b in BUCKETS}

    # 20 gold chunks + 10 hard distractors per bucket = 120 total chunks
    for b_idx, bucket in enumerate(BUCKETS):
        for i in range(20):
            noise = rng.normal(scale=0.35, size=cfg.embedding_dim)
            emb = _normalize(centroids[bucket] * 1.4 + noise)
            chunk = DocumentChunk(
                doc_id=f"DOC-{b_idx + 1}-{i + 1:02d}",
                bucket=bucket,
                is_distractor=False,
                has_exclusion_clause=(bucket == "negation_refusal"),
                has_multi_hop_rule=(bucket == "multi_hop_policy"),
                embedding=emb,
            )
            chunks.append(chunk)
            gold_by_bucket[bucket].append(chunk)

        for j in range(10):
            noise = rng.normal(scale=0.40, size=cfg.embedding_dim)
            emb = _normalize(centroids[bucket] * 1.35 + noise)
            chunks.append(
                DocumentChunk(
                    doc_id=f"DIST-{b_idx + 1}-{j + 1:02d}",
                    bucket=bucket,
                    is_distractor=True,
                    has_exclusion_clause=False,
                    has_multi_hop_rule=False,
                    embedding=emb,
                )
            )

    queries: list[GoldenQuery] = []
    for idx in range(cfg.query_count):
        bucket = BUCKETS[idx % len(BUCKETS)]
        pool = gold_by_bucket[bucket]
        gold = pool[int(rng.integers(0, len(pool)))]
        noise_scale = 0.24 if bucket == "direct_lookup" else 0.34
        q_emb = _normalize(gold.embedding + rng.normal(scale=noise_scale, size=cfg.embedding_dim))
        difficulty = float(rng.uniform(0.0, 1.0))
        queries.append(
            GoldenQuery(
                query_id=f"Q-{idx + 1:03d}",
                bucket=bucket,
                gold_doc_id=gold.doc_id,
                requires_refusal=(bucket == "negation_refusal"),
                requires_multi_hop=(bucket == "multi_hop_policy"),
                requires_disambiguation=(bucket == "ambiguous_edge"),
                difficulty=difficulty,
                embedding=q_emb,
            )
        )

    chunk_matrix = np.vstack([c.embedding for c in chunks])
    chunk_ids = tuple(c.doc_id for c in chunks)
    return RagCorpus(
        chunks=tuple(chunks),
        queries=tuple(queries),
        chunk_matrix=chunk_matrix,
        chunk_ids=chunk_ids,
    )
