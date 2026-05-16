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


def test_mixed_modality_refuses_consensus() -> None:
    # Three score-only judges agreeing + two label-only judges flagging harm.
    # Pre-fix bug: the function looked at scores first and returned
    # has_consensus=True, silently erasing the label-only safety judges.
    verdicts = [
        JudgeVerdict(judge_id="j1", score=0.9),
        JudgeVerdict(judge_id="j2", score=0.9),
        JudgeVerdict(judge_id="j3", score=0.9),
        JudgeVerdict(judge_id="j4", label="harmful"),
        JudgeVerdict(judge_id="j5", label="harmful"),
    ]
    c = compute_consensus(verdicts)
    assert c.has_consensus is False
    assert c.agreeing_judges == ()
    assert c.agreement_strength == 0.0


def test_dual_modality_judge_plus_label_only_refuses() -> None:
    # j1 supplies BOTH score and label; j2 supplies label-only. The first
    # mixed-modality fix attempt only checked disjoint score-only/label-only
    # subsets and let this slip through, silently erasing j2 on the score path.
    verdicts = [
        JudgeVerdict(judge_id="j1", score=0.9, label="good"),
        JudgeVerdict(judge_id="j2", label="harmful"),
    ]
    c = compute_consensus(verdicts)
    assert c.has_consensus is False
    assert c.consensus_score is None


def test_all_dual_modality_consensus_uses_score_path() -> None:
    # All judges supply BOTH score and label uniformly — this is consistent
    # multi-modality, not mixed. Score path is used.
    verdicts = [
        JudgeVerdict(judge_id="j1", score=0.9, label="good"),
        JudgeVerdict(judge_id="j2", score=0.92, label="good"),
        JudgeVerdict(judge_id="j3", score=0.88, label="good"),
    ]
    c = compute_consensus(verdicts, score_tolerance=0.1)
    assert c.has_consensus is True
