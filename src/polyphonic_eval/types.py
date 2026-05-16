"""Pydantic models that make up the public wire format.

Type-level invariant: ``PolyphonicResult`` does **not** expose ``__float__`` /
``__int__``. ``to_scalar(policy='refuse')`` raises. This is load-bearing — see
``docs/design/0001-no-consensus-collapse.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class JudgeVerdict(BaseModel):
    """One judge's verdict on a single evaluation item."""

    model_config = ConfigDict(frozen=True)

    judge_id: str
    score: float | None = None
    label: str | None = None
    rationale: str | None = None
    confidence: float | None = None
    metadata: Mapping[str, str] = Field(default_factory=dict)


class ScoreSummary(BaseModel):
    """Per-cluster (or per-item) summary statistics of numeric scores."""

    model_config = ConfigDict(frozen=True)

    mean: float | None
    median: float | None
    stdev: float | None
    n: int
    label_distribution: Mapping[str, int] | None = None


class DisagreementCluster(BaseModel):
    """A cluster of judges that voiced semantically similar rationales."""

    model_config = ConfigDict(frozen=True)

    cluster_id: int
    members: tuple[str, ...]
    centroid_rationale: str
    score_summary: ScoreSummary
    weight: float
    keywords: tuple[str, ...]


class ConsensusClaim(BaseModel):
    """Claim about whether judges reached a consensus."""

    model_config = ConfigDict(frozen=True)

    has_consensus: bool
    consensus_score: float | None = None
    consensus_label: str | None = None
    agreeing_judges: tuple[str, ...]
    agreement_strength: float


class IrreducibleDisagreement(BaseModel):
    """Disagreement structure with falsifiable irreducibility claim."""

    model_config = ConfigDict(frozen=True)

    is_irreducible: bool
    clusters: tuple[DisagreementCluster, ...]
    minority_clusters: tuple[int, ...]
    bootstrap_stability: float
    explanation: str


class PolyphonicResult(BaseModel):
    """Typed aggregate of multi-judge verdicts.

    The library's central type. Has no ``__float__`` / ``__int__``; ``to_scalar``
    refuses by default. Callers must opt in to scalar collapse explicitly.
    """

    model_config = ConfigDict(frozen=True)

    item_id: str
    verdicts: tuple[JudgeVerdict, ...]
    consensus: ConsensusClaim
    disagreement: IrreducibleDisagreement
    disagreement_spectrum: float
    n_judges: int
    schema_version: Literal["1"] = "1"

    def to_scalar(
        self,
        policy: Literal["majority", "mean", "refuse"] = "refuse",
    ) -> float:
        """Explicit, opt-in collapse to a single float.

        - ``refuse`` (default): raise ``TypeError``. Refusing is the point.
        - ``mean``: arithmetic mean of numeric scores.
        - ``majority``: median of numeric scores (robust central tendency),
          or ratio of the modal label for label-only data.

        Raises:
            TypeError: when ``policy='refuse'``.
            ValueError: when no usable scores/labels exist for the chosen policy.
        """
        if policy == "refuse":
            raise TypeError(
                "PolyphonicResult refuses scalar collapse. "
                "Pass policy='mean' or policy='majority' to override. "
                "See docs/design/0001-no-consensus-collapse.md for the rationale."
            )
        from polyphonic_eval.collapse import collapse_majority, collapse_mean

        if policy == "mean":
            return collapse_mean(self.verdicts)
        if policy == "majority":
            return collapse_majority(self.verdicts)
        raise ValueError(f"Unknown policy: {policy!r}")
