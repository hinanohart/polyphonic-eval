"""Tests for the opt-in collapse policies."""

from __future__ import annotations

import pytest

from polyphonic_eval import JudgeVerdict
from polyphonic_eval.collapse import collapse_majority, collapse_mean


def test_collapse_mean_simple() -> None:
    verdicts = [JudgeVerdict(judge_id=f"j{i}", score=s) for i, s in enumerate([1.0, 0.5, 0.0])]
    assert collapse_mean(verdicts) == pytest.approx(0.5)


def test_collapse_mean_no_scores_raises() -> None:
    verdicts = [JudgeVerdict(judge_id="j1", label="good")]
    with pytest.raises(ValueError, match="numeric score"):
        collapse_mean(verdicts)


def test_collapse_majority_scores_uses_median() -> None:
    verdicts = [JudgeVerdict(judge_id=f"j{i}", score=s) for i, s in enumerate([0.0, 0.5, 1.0])]
    assert collapse_majority(verdicts) == 0.5


def test_collapse_majority_robust_to_outlier() -> None:
    verdicts = [
        JudgeVerdict(judge_id=f"j{i}", score=s) for i, s in enumerate([0.9, 0.85, 0.9, 0.0])
    ]
    # median of [0, 0.85, 0.9, 0.9] = 0.875 (mean of two middle values)
    assert collapse_majority(verdicts) == pytest.approx(0.875)


def test_collapse_majority_labels_only() -> None:
    verdicts = [
        JudgeVerdict(judge_id="j1", label="good"),
        JudgeVerdict(judge_id="j2", label="good"),
        JudgeVerdict(judge_id="j3", label="bad"),
    ]
    assert collapse_majority(verdicts) == pytest.approx(2 / 3)


def test_collapse_majority_empty_raises() -> None:
    with pytest.raises(ValueError, match="scores or labels"):
        collapse_majority([JudgeVerdict(judge_id="j1")])


def _mixed_modality_verdicts() -> list[JudgeVerdict]:
    return [
        JudgeVerdict(judge_id="j1", score=0.9),
        JudgeVerdict(judge_id="j2", score=0.9),
        JudgeVerdict(judge_id="j3", label="harmful"),
        JudgeVerdict(judge_id="j4", label="harmful"),
    ]


def test_collapse_mean_refuses_mixed_modality() -> None:
    with pytest.raises(TypeError, match="mixed-modality"):
        collapse_mean(_mixed_modality_verdicts())


def test_collapse_majority_refuses_mixed_modality() -> None:
    with pytest.raises(TypeError, match="mixed-modality"):
        collapse_majority(_mixed_modality_verdicts())


def _dual_modality_plus_label_only() -> list[JudgeVerdict]:
    # Regression: judge with BOTH score+label paired with label-only judge.
    # Previously slipped past the disjoint-subset detector.
    return [
        JudgeVerdict(judge_id="j1", score=0.9, label="good"),
        JudgeVerdict(judge_id="j2", label="harmful"),
    ]


def test_collapse_mean_refuses_dual_modality_gap() -> None:
    with pytest.raises(TypeError, match="mixed-modality"):
        collapse_mean(_dual_modality_plus_label_only())


def test_collapse_majority_refuses_dual_modality_gap() -> None:
    with pytest.raises(TypeError, match="mixed-modality"):
        collapse_majority(_dual_modality_plus_label_only())
