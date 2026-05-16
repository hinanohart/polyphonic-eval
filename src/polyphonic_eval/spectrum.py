"""Disagreement spectrum index.

Combines a Simpson-style concentration term (how evenly weight is split across
clusters) with a semantic-distance term (how far apart clusters are in
embedding space).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from polyphonic_eval.types import DisagreementCluster


def disagreement_spectrum(
    clusters: Sequence[DisagreementCluster],
    pairwise_dist: np.ndarray,
) -> float:
    """Combined concentration × semantic-distance index.

    ::

        S = (1 - Σ w_i²) × normalize(mean off-diagonal pairwise_dist)

    Inputs:
        clusters: ordered sequence of ``DisagreementCluster``; ``weight`` is the
            cluster's share of judges.
        pairwise_dist: square matrix of cluster-centroid distances. For ``k``
            clusters, shape ``(k, k)`` is expected; cosine distance is the
            convention but any metric in ``[0, 1]`` works.

    Returns:
        Spectrum in ``[0, 1]``. ``0`` when ≤1 cluster (no structured spread).
    """
    if len(clusters) <= 1:
        return 0.0
    weights = np.asarray([c.weight for c in clusters], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    weights = weights / total
    simpson = 1.0 - float((weights**2).sum())

    n = pairwise_dist.shape[0]
    if n <= 1:
        return float(max(0.0, min(1.0, simpson)))
    mask = ~np.eye(n, dtype=bool)
    mean_dist = float(pairwise_dist[mask].mean())
    mean_dist = max(0.0, min(1.0, mean_dist))
    return float(max(0.0, min(1.0, simpson * mean_dist)))
