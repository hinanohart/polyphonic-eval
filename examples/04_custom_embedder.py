"""04 — Plug in a custom embedder for reproducibility.

Run with::

    pip install 'polyphonic-eval[embed]'
    python examples/04_custom_embedder.py

The embedder choice dominates clustering results. For reproducible eval
pipelines, pin a specific embedder rather than relying on the lazy default
(which may upgrade across versions of sentence-transformers).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from polyphonic_eval import JudgeVerdict, aggregate


class PinnedEmbedder:
    """Wraps a specific sentence-transformers model. Pin the version too in your
    environment file (requirements.txt / pyproject.toml).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True),
            dtype=np.float32,
        )


def main() -> None:
    embedder = PinnedEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    verdicts = [
        JudgeVerdict(judge_id=f"j{i}", score=0.9 - 0.1 * i, rationale=f"Reason {i}")
        for i in range(5)
    ]
    result = aggregate(verdicts, item_id="custom-emb-1", embedder=embedder)
    print(f"used embedder      = {embedder.model_name}")
    print(f"disagreement_spec  = {result.disagreement_spectrum:.3f}")
    print(f"n_clusters         = {len(result.disagreement.clusters)}")


if __name__ == "__main__":
    main()
