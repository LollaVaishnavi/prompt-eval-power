"""Validated configuration dataclasses for statistical prompt evaluation gates.

Exists so significance levels, target power, minimum detectable effects (MDE),
and corpus generation seeds live in explicit policy objects rather than ad-hoc
constants scattered across CI scripts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


class PolicyValidationError(ValueError):
    """Raised when statistical gate or corpus parameters are out of valid bounds."""


@dataclass(frozen=True)
class GatePolicy:
    """Statistical decision rules for a prompt regression CI gate.

    Attributes:
        alpha: One-sided Type I error rate for detecting a pass-rate regression.
        target_power: Minimum required statistical power (1 - beta) at target_mde.
        target_mde: Minimum detectable regression drop (e.g. 0.05 = 5 percentage points).
        confidence_level: Two-sided confidence level for Clopper-Pearson upper bounds.
        max_bucket_drop: Hard alert threshold for any single query bucket regression.
    """

    alpha: float = 0.05
    target_power: float = 0.80
    target_mde: float = 0.05
    confidence_level: float = 0.95
    max_bucket_drop: float = 0.10

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 0.5):
            raise PolicyValidationError(f"alpha must be in (0, 0.5), got {self.alpha}")
        if not (0.5 <= self.target_power < 1.0):
            raise PolicyValidationError(
                f"target_power must be in [0.5, 1.0), got {self.target_power}"
            )
        if not (0.005 <= self.target_mde < 0.5):
            raise PolicyValidationError(
                f"target_mde must be in [0.005, 0.5), got {self.target_mde}"
            )
        if not (0.5 < self.confidence_level < 1.0):
            raise PolicyValidationError(
                f"confidence_level must be in (0.5, 1.0), got {self.confidence_level}"
            )
        if not (0.0 < self.max_bucket_drop <= 1.0):
            raise PolicyValidationError(
                f"max_bucket_drop must be in (0, 1], got {self.max_bucket_drop}"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> GatePolicy:
        """Load a GatePolicy from a JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def to_dict(self) -> dict[str, float]:
        """Serialize policy fields for audit reports and manifests."""
        return asdict(self)


@dataclass(frozen=True)
class CorpusConfig:
    """Deterministic configuration for the offline e-commerce RAG golden corpus."""

    seed: int = 20260511
    query_count: int = 300
    embedding_dim: int = 32
    top_k: int = 3

    def __post_init__(self) -> None:
        if self.query_count < 20:
            raise PolicyValidationError(
                f"query_count must be at least 20, got {self.query_count}"
            )
        if self.embedding_dim < 8:
            raise PolicyValidationError(
                f"embedding_dim must be at least 8, got {self.embedding_dim}"
            )
        if self.top_k < 1:
            raise PolicyValidationError(f"top_k must be >= 1, got {self.top_k}")
