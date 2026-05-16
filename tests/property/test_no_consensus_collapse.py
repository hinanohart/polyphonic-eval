"""Property test: no PolyphonicResult ever silently collapses to a scalar.

This encodes the load-bearing forcing constraint of the library.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polyphonic_eval import JudgeVerdict
from polyphonic_eval.core import AggregatorConfig, aggregate


def _verdicts_strategy(min_n: int = 3, max_n: int = 8) -> st.SearchStrategy[list[JudgeVerdict]]:
    return st.lists(
        st.builds(
            JudgeVerdict,
            judge_id=st.text(
                min_size=1,
                max_size=10,
                alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            ),
            score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            rationale=st.text(min_size=1, max_size=80),
        ),
        min_size=min_n,
        max_size=max_n,
        unique_by=lambda v: v.judge_id,
    )


@given(verdicts=_verdicts_strategy())
@settings(
    max_examples=1000,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_float_always_raises(verdicts: list[JudgeVerdict], deterministic_embedder) -> None:
    """For any non-empty verdicts list, the result must refuse float()."""
    cfg = AggregatorConfig(bootstrap_iters=10)
    result = aggregate(verdicts, item_id="prop", embedder=deterministic_embedder, config=cfg)
    import pytest

    with pytest.raises(TypeError):
        float(result)
    with pytest.raises(TypeError):
        result.to_scalar(policy="refuse")
    with pytest.raises(TypeError):
        result.to_scalar()  # default == refuse


@given(verdicts=_verdicts_strategy(min_n=3, max_n=6))
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_to_scalar_mean_is_idempotent(verdicts: list[JudgeVerdict], deterministic_embedder) -> None:
    """Repeated to_scalar('mean') returns the same value."""
    cfg = AggregatorConfig(bootstrap_iters=5)
    result = aggregate(verdicts, item_id="prop", embedder=deterministic_embedder, config=cfg)
    a = result.to_scalar(policy="mean")
    b = result.to_scalar(policy="mean")
    assert a == b


@given(verdicts=_verdicts_strategy(min_n=3, max_n=6))
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_spectrum_in_unit_interval(verdicts: list[JudgeVerdict], deterministic_embedder) -> None:
    """Disagreement spectrum must stay in [0, 1]."""
    cfg = AggregatorConfig(bootstrap_iters=5)
    result = aggregate(verdicts, item_id="prop", embedder=deterministic_embedder, config=cfg)
    assert 0.0 <= result.disagreement_spectrum <= 1.0
