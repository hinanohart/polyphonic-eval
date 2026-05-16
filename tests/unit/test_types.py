"""Type-level invariants of the public models."""

from __future__ import annotations

import pytest

from polyphonic_eval import (
    ConsensusClaim,
    DisagreementCluster,
    IrreducibleDisagreement,
    JudgeVerdict,
    PolyphonicResult,
    ScoreSummary,
)


def _make_result() -> PolyphonicResult:
    ss = ScoreSummary(mean=0.5, median=0.5, stdev=0.1, n=1)
    cluster = DisagreementCluster(
        cluster_id=0,
        members=("j1",),
        centroid_rationale="r",
        score_summary=ss,
        weight=1.0,
        keywords=("foo",),
    )
    return PolyphonicResult(
        item_id="x",
        verdicts=(JudgeVerdict(judge_id="j1", score=0.5, rationale="r"),),
        consensus=ConsensusClaim(
            has_consensus=True, agreeing_judges=("j1",), agreement_strength=1.0
        ),
        disagreement=IrreducibleDisagreement(
            is_irreducible=False,
            clusters=(cluster,),
            minority_clusters=(),
            bootstrap_stability=0.0,
            explanation="single cluster",
        ),
        disagreement_spectrum=0.0,
        n_judges=1,
    )


def test_result_has_no_float() -> None:
    result = _make_result()
    with pytest.raises(TypeError):
        float(result)


def test_to_scalar_default_refuses() -> None:
    result = _make_result()
    with pytest.raises(TypeError, match="refuses scalar collapse"):
        result.to_scalar()


def test_to_scalar_refuse_explicit() -> None:
    result = _make_result()
    with pytest.raises(TypeError):
        result.to_scalar(policy="refuse")


def test_to_scalar_mean() -> None:
    result = _make_result()
    assert result.to_scalar(policy="mean") == 0.5


def test_to_scalar_majority_uses_median() -> None:
    result = _make_result()
    # single score → median == score
    assert result.to_scalar(policy="majority") == 0.5


def test_models_are_frozen() -> None:
    v = JudgeVerdict(judge_id="j1", score=0.5)
    with pytest.raises((TypeError, AttributeError, ValueError)):
        v.judge_id = "j2"  # type: ignore[misc]


def test_schema_version_is_one() -> None:
    result = _make_result()
    assert result.schema_version == "1"


def test_round_trip_json() -> None:
    result = _make_result()
    payload = result.model_dump_json()
    restored = PolyphonicResult.model_validate_json(payload)
    assert restored == result


def test_bool_on_polyphonic_result_refuses() -> None:
    result = _make_result()
    with pytest.raises(TypeError, match="bool"):
        bool(result)


def test_judge_verdict_rejects_nan_score() -> None:
    with pytest.raises(ValueError, match="finite"):
        JudgeVerdict(judge_id="j1", score=float("nan"))


def test_judge_verdict_rejects_inf_score() -> None:
    with pytest.raises(ValueError, match="finite"):
        JudgeVerdict(judge_id="j1", score=float("inf"))


def test_judge_verdict_rejects_nan_confidence() -> None:
    with pytest.raises(ValueError, match="finite"):
        JudgeVerdict(judge_id="j1", score=0.5, confidence=float("nan"))


def test_int_and_arithmetic_on_result_refused() -> None:
    result = _make_result()
    with pytest.raises(TypeError):
        int(result)
    with pytest.raises(TypeError):
        result + 1  # type: ignore[operator]
