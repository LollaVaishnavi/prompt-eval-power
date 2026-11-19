# ADR-004: Stratified Query Buckets Over Flat Golden Sets

## Status
Accepted

## Context
Enterprise RAG queries do not fail uniformly. Prompt compression edits often leave simple fact lookups intact while stripping refusal instructions or multi-hop policy checks, causing silent failures in high-risk slices.

## Decision
Stratify the 300 labeled queries evenly across four operational buckets (`direct_lookup`, `multi_hop_policy`, `negation_refusal`, `ambiguous_edge`) and run paired McNemar checks per bucket in addition to the overall gate.

## Alternatives Considered
1. **Unstratified random sample of production traffic:** Dominated by easy lookup queries, which dilutes regressions on rarer refusal and multi-hop queries.

## Consequences
* On `v2_concise_rewrite`, `direct_lookup` drops only 2.7pp (2 flips, non-significant), while `negation_refusal` drops 17.3pp (13 flips, p=0.00012) and `multi_hop_policy` drops 13.3pp (10 flips, p=0.00098).
* Any single bucket breaching `max_bucket_drop` with `p < alpha` triggers `REJECT_REGRESSION` (exit 2) even if other buckets improved.
* Trade-off: Per-bucket power is lower than overall power (`n / 4` per slice), so bucket checks catch large slice regressions while the overall gate tracks smaller portfolio drops.
