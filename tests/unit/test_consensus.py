"""Tests for ``compute_consensus``."""

from __future__ import annotations

from polyphonic_eval import JudgeVerdict
from polyphonic_eval.consensus import compute_consensus


def test_unanimous_scores_yields_consensus() -> None:
    verdicts = [JudgeVerdict(judge_id=f"j{i}", score=0.9) for i in range(5)]
    c = compute_consensus(verdicts, score_tolerance=0.05)
    assert c.has_consensus is True
    assert c.consensus_score is not None
    assert 0.85 < c.consensus_score < 0.95
    assert len(c.agreeing_judges) == 5
    assert c.agreement_strength >= 0.95


def test_split_scores_yields_no_consensus() -> None:
    verdicts = [
        JudgeVerdict(judge_id="j1", score=0.9),
        JudgeVerdict(judge_id="j2", score=0.85),
        JudgeVerdict(judge_id="j3", score=0.1),
    ]
    c = compute_consensus(verdicts, score_tolerance=0.15)
    assert c.has_consensus is False
    assert c.agreement_strength < 1.0


def test_labels_strict_majority_not_consensus() -> None:
    verdicts = [
        JudgeVerdict(judge_id="j1", label="good"),
        JudgeVerdict(judge_id="j2", label="good"),
        JudgeVerdict(judge_id="j3", label="bad"),
    ]
    c = compute_consensus(verdicts, label_strict=True)
    assert c.has_consensus is False
    assert c.agreement_strength == 2 / 3


def test_labels_unanimous_is_consensus() -> None:
    verdicts = [JudgeVerdict(judge_id=f"j{i}", label="good") for i in range(4)]
    c = compute_consensus(verdicts, label_strict=True)
    assert c.has_consensus is True
    assert c.consensus_label == "good"


def test_empty_verdicts_no_consensus() -> None:
    c = compute_consensus([])
    assert c.has_consensus is False
    assert c.agreeing_judges == ()
