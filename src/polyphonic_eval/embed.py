"""Embedder protocol and the lazy ``sentence-transformers`` default."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps a sequence of strings to a (n, d) float array."""

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


_DEFAULT_INSTANCE: Embedder | None = None
_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def default_embedder() -> Embedder:
    """Return a process-wide cached default Embedder.

    Lazily imports ``sentence_transformers``. The optional ``[embed]`` extra
    pulls the dependency in. Raises ``ImportError`` with installation guidance
    if it is not available.

    .. warning::
       Embedding model choice **dominates** clustering results. For reproducible
       eval pipelines, pin a specific embedder rather than rely on this default.
       See ``docs/design/0003-embedder-protocol.md``.
    """
    global _DEFAULT_INSTANCE  # noqa: PLW0603
    if _DEFAULT_INSTANCE is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "default_embedder() requires sentence-transformers. "
                "Install with: pip install 'polyphonic-eval[embed]', "
                "or pass a custom Embedder to aggregate()."
            ) from exc
        model = SentenceTransformer(_DEFAULT_MODEL_NAME)
        _DEFAULT_INSTANCE = _SentenceTransformerEmbedder(model)
    return _DEFAULT_INSTANCE


class _SentenceTransformerEmbedder:
    """Thin Embedder adapter around a SentenceTransformer model."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        result = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        arr = np.asarray(result, dtype=np.float32)
        return cast(np.ndarray, arr)
