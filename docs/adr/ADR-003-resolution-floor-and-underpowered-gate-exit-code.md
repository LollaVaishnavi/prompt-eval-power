# ADR-003: Distinct Exit Code 3 for Underpowered Suites Instead of Returning Green

## Status
Accepted

## Context
A standard hypothesis-test CI gate returns exit code `0` whenever `p >= alpha`. On a typical 30-query prompt suite, a real 10pp regression fails to reach significance 75.2% of the time (75.2% false-pass rate), so returning `0` turns statistical blindness into a green checkmark.

## Decision
Implement a three-way decision gate with four semantic exit codes: `0` (PASS: powered and no significant regression), `2` (REJECT_REGRESSION: statistically significant drop), `3` (UNDERPOWERED: no significant drop, but `n < required_n` at target MDE and power), and `1` (runtime or input error).

## Alternatives Considered
1. **Binary pass/fail gate (`0` or `2`):** Masks underpowered evaluations as safe releases.
2. **Warning log with exit code `0`:** CI pipelines ignore stdout warnings when the exit status is zero.

## Consequences
* Running `v1_baseline_grounded` vs `v2_concise_rewrite` on `n=30` exits `3` (`UNDERPOWERED`, deficit of 216 queries) instead of shipping the broken prompt.
* Teams can decide explicitly in CI whether `exit 3` blocks merge or triggers a golden-set labeling task.
* Zero-failure results (`0/30`) are paired with their exact 9.5% upper bound (Clopper-Pearson 95%) so a measured zero is never presented as a 0% defect rate.
