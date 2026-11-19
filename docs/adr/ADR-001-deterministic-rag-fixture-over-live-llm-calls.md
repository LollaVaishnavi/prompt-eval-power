# ADR-001: Deterministic Vector-Index and Prompt-Graph Fixture Over Live LLM Calls

## Status
Accepted

## Context
Prompt regression harnesses that call live Amazon Bedrock, OpenAI, or Pinecone endpoints during CI suffer from API key requirements, rate limits, upstream model-version drift, and nondeterministic decoding, making statistical power claims impossible to reproduce offline.

## Decision
Represent the RAG workflow as a deterministic pure-NumPy cosine vector index and LangGraph-style state machine (`retrieve -> rerank -> apply_prompt_template -> grade`) over a seeded corpus of 300 labeled queries across four stratified buckets.

## Alternatives Considered
1. **Live Bedrock + Pinecone calls in CI:** Realistic text generation, but requires cloud credentials and introduces confounding provider variance that cannot be separated from prompt edits.
2. **Recorded static JSON responses only:** Fast, but cannot test how changing prompt context limits (`max_context_chunks`) or rerank rules alters retrieval-to-generation outcomes.

## Consequences
* `make verify` runs with zero API keys and zero network calls, producing identical outcomes across Python 3.10, 3.11, and 3.12.
* The statistical engine (`stats.py`, `power.py`) is completely decoupled from the offline graph and accepts any binary pass/fail arrays from live LLM evaluation logs.
* Trade-off: Live decoding temperature variance and LLM-judge calibration bias are outside the fixture boundary (documented in ADR-005).
