"""Tests for ``aggregate`` and ``PolyphonicAggregator``."""

from __future__ import annotations

import pytest

from polyphonic_eval import PolyphonicResult
from polyphonic_eval.core import AggregatorConfig, PolyphonicAggregator, aggregate


def test_aggregate_unanimous(deterministic_embedder, unanimous_verdicts) -> None:
    result = aggregate(
        unanimous_verdicts,
        item_id="unanimous",
        embedder=deterministic_embedder,
        config=AggregatorConfig(bootstrap_iters=20),
    )
    assert isinstance(result, PolyphonicResult)
    assert result.item_id == "unanimous"
    assert result.n_judges == 5
    assert result.consensus.has_consensus is True
    assert result.disagreement.is_irreducible is False
    assert result.disagreement_spectrum >= 0.0


def test_aggregate_split(deterministic_embedder, split_verdicts) -> None:
    result = aggregate(
        split_verdicts,
        item_id="split",
        embedder=deterministic_embedder,
        config=AggregatorConfig(bootstrap_iters=30, consensus_score_tolerance=0.1),
    )
    assert result.consensus.has_consensus is False
    # Split scores → spread is wide → low agreement_strength
    assert result.consensus.agreement_strength < 0.95


def test_aggregate_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one verdict"):
        aggregate([], item_id="x")


def test_aggregator_class_caches_config(deterministic_embedder, unanimous_verdicts) -> None:
    agg = PolyphonicAggregator(
        embedder=deterministic_embedder,
        config=AggregatorConfig(bootstrap_iters=10),
    )
    result1 = agg.aggregate(unanimous_verdicts, item_id="a")
    result2 = agg.aggregate(unanimous_verdicts, item_id="b")
    assert result1.item_id == "a"
    assert result2.item_id == "b"
    assert result1.consensus.has_consensus == result2.consensus.has_consensus


def test_aggregate_preserves_all_verdicts(deterministic_embedder, split_verdicts) -> None:
    result = aggregate(
        split_verdicts,
        item_id="preserve",
        embedder=deterministic_embedder,
        config=AggregatorConfig(bootstrap_iters=10),
    )
    assert len(result.verdicts) == len(split_verdicts)
    assert {v.judge_id for v in result.verdicts} == {v.judge_id for v in split_verdicts}


def test_aggregate_refuses_to_scalar(deterministic_embedder, split_verdicts) -> None:
    result = aggregate(
        split_verdicts,
        item_id="refuse",
        embedder=deterministic_embedder,
        config=AggregatorConfig(bootstrap_iters=10),
    )
    with pytest.raises(TypeError):
        result.to_scalar()


def test_aggregate_explicit_mean_works(deterministic_embedder, split_verdicts) -> None:
    result = aggregate(
        split_verdicts,
        item_id="mean",
        embedder=deterministic_embedder,
        config=AggregatorConfig(bootstrap_iters=10),
    )
    expected = sum(v.score for v in split_verdicts if v.score is not None) / len(split_verdicts)
    assert result.to_scalar(policy="mean") == pytest.approx(expected)
