"""Tests for ``is_irreducible``."""

from __future__ import annotations

from polyphonic_eval import JudgeVerdict
from polyphonic_eval.irreducible import is_irreducible


def test_unanimous_is_not_irreducible(deterministic_embedder, unanimous_verdicts) -> None:
    is_irr, _, _ = is_irreducible(
        unanimous_verdicts,
        deterministic_embedder,
        n_bootstrap=50,
    )
    assert is_irr is False


def test_split_with_distinct_rationales_produces_measurable_stability(
    deterministic_embedder,
) -> None:
    # 8+8 = 16 verdicts; bootstrap is more reliable than at N=6.
    good = [
        JudgeVerdict(judge_id=f"a{i}", score=0.9, rationale="helpful concise answer good")
        for i in range(8)
    ]
    bad = [
        JudgeVerdict(
            judge_id=f"b{i}", score=0.1, rationale="contains factual error chemistry wrong"
        )
        for i in range(8)
    ]
    is_irr, stab, _ = is_irreducible(
        good + bad,
        deterministic_embedder,
        n_bootstrap=100,
    )
    # Two perfectly-duplicate text groups of size 8 each give HDBSCAN a
    # textbook separable problem; mean ARI under bootstrap is essentially 1.0.
    assert stab > 0.6
    assert is_irr is True


def test_too_few_verdicts_returns_not_irreducible(deterministic_embedder) -> None:
    verdicts = [
        JudgeVerdict(judge_id="j1", score=0.9, rationale="x"),
        JudgeVerdict(judge_id="j2", score=0.1, rationale="y"),
    ]
    is_irr, stab, _ = is_irreducible(verdicts, deterministic_embedder, n_bootstrap=20)
    assert is_irr is False
    assert stab == 0.0
