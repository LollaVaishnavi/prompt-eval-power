# ADR-002: Paired McNemar Testing Over Unpaired Pass-Rate Thresholds

## Status
Accepted

## Context
Many teams gate prompt edits by comparing aggregate pass rates (`p_new - p_old >= -0.02`) or running an unpaired two-proportion z-test. Because baseline and candidate prompts are evaluated on the exact same golden queries, unpaired tests throw away query-level correlation and inflate required sample sizes.

## Decision
Use exact one-sided McNemar testing (`scipy.stats.binomtest` on discordant pairs `n10` pass-to-fail vs `n01` fail-to-pass) paired with Connor's analytical McNemar power formula.

## Alternatives Considered
1. **Unpaired two-proportion z-test:** Simple, but requires 714 unpaired queries to detect a 5pp drop at 80% power (versus 369 paired queries at 15% discordance).
2. **Fixed pass-rate delta threshold without p-values:** Fails to separate a 5pp drop caused by binomial noise on `n=30` from a genuine regression.

## Consequences
* At realistic 15% discordance, paired McNemar cuts required labeled queries by 48.3% (369 paired queries vs 714 unpaired queries).
* Exposes discordant churn even when aggregate pass rates look flat (e.g. 10 regressions hidden by 10 improvements).
* Uncomfortable trade-off: When a radical prompt rewrite scrambles outcomes across queries (45% discordance), paired McNemar loses to unpaired testing and requires 1111 queries at 45% discordance.
