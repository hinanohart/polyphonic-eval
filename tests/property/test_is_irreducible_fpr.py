"""Property test: false-positive rate of ``is_irreducible`` under H0.

Null hypothesis: rationales are i.i.d. paraphrases of a single underlying view.
We approximate that by sampling judges whose rationales differ only in
trivial filler words, plus near-identical scores. Under this null, the test
should rarely report ``irreducible=True``. We enforce FPR < 10% to keep
the property test cheap while still asserting the core claim. The spec
target FPR is < 5% with the default 200-bootstrap, here we use a smaller
budget for CI speed.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polyphonic_eval import JudgeVerdict
from polyphonic_eval.irreducible import is_irreducible

FILLERS = (
    "the answer is helpful and accurate",
    "the answer is correct and useful",
    "this response is good and clear",
    "the reply is helpful and informative",
    "the response is accurate and helpful",
)


def _h0_verdicts(rng: np.random.Generator, n: int) -> list[JudgeVerdict]:
    """Generate ``n`` judges all sampling from the same 'helpful' rationale pool."""
    mu = 0.85
    sigma = 0.05
    out: list[JudgeVerdict] = []
    for i in range(n):
        score = float(np.clip(rng.normal(mu, sigma), 0.0, 1.0))
        rationale = FILLERS[rng.integers(0, len(FILLERS))]
        out.append(JudgeVerdict(judge_id=f"j{i}", score=score, rationale=rationale))
    return out


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_fpr_under_h0_is_low(seed: int, deterministic_embedder) -> None:
    rng = np.random.default_rng(seed)
    verdicts = _h0_verdicts(rng, n=6)
    is_irr, _, _ = is_irreducible(
        verdicts,
        deterministic_embedder,
        n_bootstrap=50,
        rng_seed=int(seed),
    )
    # Per-example assertion is fine: the deterministic embedder + paraphrase pool
    # rarely yields ≥2 stable clusters. If it ever does, that's a real false positive
    # worth investigating.
    assert is_irr is False
