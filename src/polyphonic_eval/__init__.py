"""polyphonic-eval: multi-judge LLM evaluation that doesn't collapse minority signal.

Public API re-exports. Implementation lives in submodules.
"""

from polyphonic_eval.core import AggregatorConfig, PolyphonicAggregator, aggregate
from polyphonic_eval.embed import Embedder, default_embedder
from polyphonic_eval.types import (
    ConsensusClaim,
    DisagreementCluster,
    IrreducibleDisagreement,
    JudgeVerdict,
    PolyphonicResult,
    ScoreSummary,
)

__all__ = [
    "AggregatorConfig",
    "ConsensusClaim",
    "DisagreementCluster",
    "Embedder",
    "IrreducibleDisagreement",
    "JudgeVerdict",
    "PolyphonicAggregator",
    "PolyphonicResult",
    "ScoreSummary",
    "__version__",
    "aggregate",
    "default_embedder",
]

__version__ = "0.1.1"
