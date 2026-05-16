"""Shared fixtures for polyphonic-eval test suite."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from polyphonic_eval import Embedder, JudgeVerdict


def _stable_bucket(token: str, dim: int) -> int:
    """Hash a token to a bucket index, stable across processes.

    Python's builtin ``hash()`` is randomized per process via ``PYTHONHASHSEED``,
    so the previous implementation was only "deterministic" within one run.
    Using ``blake2b`` makes the fixture genuinely repeatable in CI.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class DeterministicHashEmbedder:
    """Hash-based deterministic embedder. No model load; suitable for unit tests.

    Maps a string to a 64-dim vector by hashing token n-grams via blake2b.
    Distances roughly track Jaccard overlap. Not semantically meaningful —
    used only to exercise clustering machinery without network or model
    dependencies. Repeatable across processes regardless of ``PYTHONHASHSEED``.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                out[i, _stable_bucket(tok, self.dim)] += 1.0
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
