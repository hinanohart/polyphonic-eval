"""Opt-in scalar collapse policies.

Exposed as standalone functions for direct testing. ``PolyphonicResult.to_scalar``
re-uses them via a lazy import to avoid circular imports.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from statistics import median

from polyphonic_eval.types import JudgeVerdict


def _has_mixed_modality(verdicts: Sequence[JudgeVerdict]) -> bool:
    """True when both modalities appear but neither is uniform across the panel.

    Catches the subtle case where some judges supply BOTH score+label and
    others supply label-only (or score-only): naive score-path collapse would
    silently erase the partial-modality judge's contribution.
    """
    any_score = any(v.score is not None for v in verdicts)
    any_label = any(v.label is not None for v in verdicts)
    all_score = all(v.score is not None for v in verdicts)
    all_label = all(v.label is not None for v in verdicts)
    return any_score and any_label and not (all_score and all_label)


def collapse_mean(verdicts: Sequence[JudgeVerdict]) -> float:
    """Arithmetic mean of numeric scores. Raises if no numeric scores exist.

    Refuses mixed-modality input (some judges score-only, others label-only):
    averaging would silently discard label-only judges — the exact failure mode
    the library exists to surface.
    """
    if _has_mixed_modality(verdicts):
        raise TypeError(
            "collapse_mean refuses mixed-modality verdicts: some judges supplied "
            "scores only, others labels only. Averaging would silently discard "
            "label-only judges. Filter to one modality, or build a joint "
            "ScoreSummary explicitly."
        )
    scores = [v.score for v in verdicts if v.score is not None]
    if not scores:
        raise ValueError("collapse_mean requires at least one numeric score; got none.")
    return sum(scores) / len(scores)


def collapse_majority(verdicts: Sequence[JudgeVerdict]) -> float:
    """Robust central tendency.

    - When numeric scores exist: returns their median.
    - When only labels exist: returns the ratio of judges voting for the modal label.

    Refuses mixed-modality input for the same reason as :func:`collapse_mean`.
    """
    if _has_mixed_modality(verdicts):
        raise TypeError(
            "collapse_majority refuses mixed-modality verdicts: some judges "
            "supplied scores only, others labels only. Taking the score-median "
            "would silently discard label-only judges."
        )
    scores = [v.score for v in verdicts if v.score is not None]
    if scores:
        return float(median(scores))
    labels = [v.label for v in verdicts if v.label is not None]
    if labels:
        counter = Counter(labels)
        _, count = counter.most_common(1)[0]
        return count / len(labels)
    raise ValueError("collapse_majority requires either scores or labels; got neither.")
