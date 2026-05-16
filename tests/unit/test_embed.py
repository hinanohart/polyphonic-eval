"""Tests for the Embedder protocol and default embedder error path."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from polyphonic_eval import Embedder


class _DummyEmbedder:
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.eye(len(texts), 4, dtype=np.float32)


def test_protocol_runtime_check() -> None:
    e = _DummyEmbedder()
    assert isinstance(e, Embedder)


def test_non_protocol_object_rejected() -> None:
    assert not isinstance(object(), Embedder)


def test_default_embedder_raises_helpful_error_when_no_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polyphonic_eval.embed as embed_mod

    monkeypatch.setattr(embed_mod, "_DEFAULT_INSTANCE", None, raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"polyphonic-eval\[embed\]"):
        embed_mod.default_embedder()
