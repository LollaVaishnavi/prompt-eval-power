"""Public package interface for prompt_eval_power.

Exposes gate policies, statistical tests, power/sample-size calculators, and
the deterministic offline RAG state-graph evaluator.
"""

from prompt_eval_power.config import CorpusConfig, GatePolicy

__all__ = [
    "CorpusConfig",
    "GatePolicy",
    "__version__",
]

__version__ = "0.1.0"
