"""Consensus detection.

Detects whether judges substantively agree on a numeric scale (scores within a
tolerance window) or on a categorical label (strict unanimity or majority).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from polyphonic_eval.types import ConsensusClaim, JudgeVerdict


def compute_consensus(
    verdicts: Sequence[JudgeVerdict],
    *,
    score_tolerance: float = 0.15,
    label_strict: bool = True,
) -> ConsensusClaim:
    """Detect whether judges agree, on either scores or labels.

    Args:
        verdicts: sequence of judge verdicts.
        score_tolerance: max allowed (max - min) score spread for consensus.
        label_strict: when labels are used, require unanimity (True) or
            majority (False).

    Returns:
        ConsensusClaim with ``has_consensus``, optional consensus value, the
        agreeing judges, and a 0–1 ``agreement_strength`` score.
    """
    if not verdicts:
        return ConsensusClaim(has_consensus=False, agreeing_judges=(), agreement_strength=0.0)

    scores = [v.score for v in verdicts if v.score is not None]
    labels = [v.label for v in verdicts if v.label is not None]

    if scores:
        return _score_consensus(verdicts, scores, score_tolerance)
    if labels:
        return _label_consensus(verdicts, labels, label_strict)

    return ConsensusClaim(has_consensus=False, agreeing_judges=(), agreement_strength=0.0)


def _score_consensus(
    verdicts: Sequence[JudgeVerdict],
    scores: list[float],
    tolerance: float,
) -> ConsensusClaim:
    arr = np.asarray(scores, dtype=np.float64)
    spread = float(arr.max() - arr.min())
    has = spread <= tolerance
    mean_score = float(arr.mean())
    agreeing = tuple(
        v.judge_id
        for v in verdicts
        if v.score is not None and abs(v.score - mean_score) <= tolerance
    )
    denom = max(1.0, float(arr.max()) if float(arr.max()) > 0 else 1.0)
    strength = max(0.0, min(1.0, 1.0 - (spread / denom)))
    return ConsensusClaim(
        has_consensus=has,
        consensus_score=mean_score if has else None,
        consensus_label=None,
        agreeing_judges=agreeing,
        agreement_strength=float(strength),
    )


def _label_consensus(
    verdicts: Sequence[JudgeVerdict],
    labels: list[str],
    strict: bool,
) -> ConsensusClaim:
    counter = Counter(labels)
    most_common, count = counter.most_common(1)[0]
    ratio = count / len(labels)
    has = ratio == 1.0 if strict else ratio > 0.5
    agreeing = tuple(v.judge_id for v in verdicts if v.label == most_common)
    return ConsensusClaim(
        has_consensus=has,
        consensus_score=None,
        consensus_label=most_common if has else None,
        agreeing_judges=agreeing,
        agreement_strength=float(ratio),
    )
