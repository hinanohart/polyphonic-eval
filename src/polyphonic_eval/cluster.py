"""HDBSCAN-backed default clustering, with a ``cluster_fn`` injection point."""

from __future__ import annotations

from collections.abc import Callable

import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def default_cluster_fn(
    *,
    min_cluster_size: int = 2,
    metric: str = "cosine",
    cluster_selection_epsilon: float = 0.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable ``(embeddings) -> labels`` wrapping HDBSCAN.

    Cosine distance is handled by precomputing a distance matrix (HDBSCAN does
    not natively support cosine). Other metrics are passed through verbatim.

    Labels are integer cluster ids; ``-1`` denotes HDBSCAN's noise cluster.
    """

    def _cluster(embeddings: np.ndarray) -> np.ndarray:
        n = embeddings.shape[0]
        if n < 2:
            return np.zeros(n, dtype=np.int64)
        if metric == "cosine":
            dist = cosine_distances(embeddings).astype(np.float64)
            np.fill_diagonal(dist, 0.0)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric="precomputed",
                cluster_selection_epsilon=cluster_selection_epsilon,
            )
            labels = clusterer.fit_predict(dist)
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric=metric,
                cluster_selection_epsilon=cluster_selection_epsilon,
            )
            labels = clusterer.fit_predict(embeddings.astype(np.float64))
        return np.asarray(labels, dtype=np.int64)

    return _cluster
