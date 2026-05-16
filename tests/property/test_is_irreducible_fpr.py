"""Property test: ``is_irreducible`` does not fire on H0 inputs.

Null hypothesis (H0):
    All judges hold the same view — their rationales are paraphrases of one
    underlying statement, with only trivial score noise.

The test embedder (:class:`tests.conftest.DeterministicHashEmbedder`) is a
bag-of-words hasher, not a semantic embedder; under that embedder, two
lexically distinct paraphrases produce non-identical vectors, so the *real*
test of "paraphrase equivalence" cannot be done with this fixture. We
therefore approximate H0 by giving every judge the **same rationale string**:
the embedder maps them to identical vectors and HDBSCAN cannot find any
structural disagreement. ``is_irreducible`` must return ``False``.

What this asserts:
    For every Hypothesis sample, ``is_irreducible`` returns ``False`` when
    every judge supplies the same rationale. It is **not** a statistical FPR
    rate estimate — that would require Wilson-style confidence intervals on
    real semantic embeddings. Rigorous FPR calibration is on the v0.2.0
    roadmap (see :doc:`/docs/theory.md`).
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polyphonic_eval import JudgeVerdict
from polyphonic_eval.irreducible import is_irreducible

H0_RATIONALE = "the answer is helpful and accurate"


def _h0_verdicts(rng: np.random.Generator, n: int) -> list[JudgeVerdict]:
    """All ``n`` judges hold the same view; scores wiggle around the same mean."""
    mu = 0.85
    sigma = 0.05
    out: list[JudgeVerdict] = []
    for i in range(n):
        score = float(np.clip(rng.normal(mu, sigma), 0.0, 1.0))
        out.append(JudgeVerdict(judge_id=f"j{i}", score=score, rationale=H0_RATIONALE))
    return out


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_h0_never_fires(seed: int, deterministic_embedder) -> None:
    rng = np.random.default_rng(seed)
    verdicts = _h0_verdicts(rng, n=6)
    is_irr, _, _ = is_irreducible(
        verdicts,
        deterministic_embedder,
        n_bootstrap=50,
        rng_seed=int(seed),
    )
    assert is_irr is False
