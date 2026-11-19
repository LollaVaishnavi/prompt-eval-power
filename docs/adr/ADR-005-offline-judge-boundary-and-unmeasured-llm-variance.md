# ADR-005: Offline Judge Boundary and What This Repository Cannot Measure

## Status
Accepted

## Context
Finite-sample power arithmetic and paired McNemar tests are exact regardless of how binary pass/fail labels are generated, but an offline RAG fixture cannot observe stochastic production LLM behavior.

## Decision
Document explicitly which claims survive identically against live Bedrock/OpenAI/LangGraph pipelines and which failure modes remain outside this repository's measurement boundary.

## What Survives Identically Against Production LLM Runs
* Exact Clopper-Pearson zero-failure upper bounds (`0/30` = 9.5% upper bound; 299 zero-failure queries required for <1%).
* Unpaired vs. paired McNemar power curves, MDE calculations, and sample-size deficit arithmetic.
* The three-way CI gate (`0` PASS, `2` REJECT_REGRESSION, `3` UNDERPOWERED) and per-bucket breach detection.

## What This Repository Deliberately Cannot Measure
1. **Non-zero temperature decoding variance:** Running the same prompt twice at `temperature > 0` introduces within-query randomness not modeled by deterministic grading; production pipelines should average repeated trials or pin `temperature=0` during regression gating.
2. **LLM-as-a-judge bias and label noise:** If automated graders mislabel 5% of answers, true statistical power is lower than the noiseless calculations here.
3. **Production query distribution shift:** A static golden set cannot detect regressions on brand-new product categories or adversarial user phrasings absent from the corpus.
4. **Offline sandbox tooling boundary:** In offline environments without `pytest`, `ruff`, or headless Chromium/Playwright, `Makefile` falls back to stdlib `unittest` + `trace` (measuring 74 tests and 99.1% line coverage), `tools/lint_check.py`, and `weasyprint`/`pdftoppm` + `PIL`/`ffmpeg`, while `.github/workflows/ci.yml` executes `ruff` and `pytest` across Python 3.10, 3.11, and 3.12.
