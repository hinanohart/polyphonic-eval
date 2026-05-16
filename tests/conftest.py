"""Shared fixtures for polyphonic-eval test suite."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from polyphonic_eval import Embedder, JudgeVerdict


class DeterministicHashEmbedder:
    """Hash-based deterministic embedder. No model load; suitable for unit tests.

    Maps a string to a 64-dim vector by hashing token n-grams. Distances roughly
    track Jaccard overlap. Not semantically meaningful — used only to exercise
    clustering machinery without network or model dependencies.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            for tok in tokens:
                idx = hash(tok) % self.dim
                out[i, idx] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


@pytest.fixture
def deterministic_embedder() -> Embedder:
    return DeterministicHashEmbedder()


@pytest.fixture
def unanimous_verdicts() -> list[JudgeVerdict]:
    return [
        JudgeVerdict(judge_id=f"j{i}", score=0.9, rationale="Helpful and accurate.")
        for i in range(5)
    ]


@pytest.fixture
def split_verdicts() -> list[JudgeVerdict]:
    return [
        JudgeVerdict(judge_id="j1", score=0.9, rationale="Helpful and accurate."),
        JudgeVerdict(judge_id="j2", score=0.85, rationale="Useful answer overall."),
        JudgeVerdict(
            judge_id="j3", score=0.2, rationale="Contains factual errors about chemistry."
        ),
        JudgeVerdict(judge_id="j4", score=0.15, rationale="Factually wrong on the chemistry part."),
        JudgeVerdict(
            judge_id="j5", score=0.1, rationale="Safety concern: instructions could harm a child."
        ),
    ]
