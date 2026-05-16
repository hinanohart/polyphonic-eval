"""Opt-in scalar collapse policies.

Exposed as standalone functions for direct testing. ``PolyphonicResult.to_scalar``
re-uses them via a lazy import to avoid circular imports.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from statistics import median

from polyphonic_eval.types import JudgeVerdict


def collapse_mean(verdicts: Sequence[JudgeVerdict]) -> float:
    """Arithmetic mean of numeric scores. Raises if no numeric scores exist."""
    scores = [v.score for v in verdicts if v.score is not None]
    if not scores:
        raise ValueError("collapse_mean requires at least one numeric score; got none.")
    return sum(scores) / len(scores)


def collapse_majority(verdicts: Sequence[JudgeVerdict]) -> float:
    """Robust central tendency.

    - When numeric scores exist: returns their median.
    - When only labels exist: returns the ratio of judges voting for the modal label.
    """
    scores = [v.score for v in verdicts if v.score is not None]
    if scores:
        return float(median(scores))
    labels = [v.label for v in verdicts if v.label is not None]
    if labels:
        counter = Counter(labels)
        _, count = counter.most_common(1)[0]
        return count / len(labels)
    raise ValueError("collapse_majority requires either scores or labels; got neither.")
