"""Falsifiable irreducible-disagreement test via bootstrap cluster stability."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score

from polyphonic_eval.embed import Embedder
from polyphonic_eval.types import JudgeVerdict


def is_irreducible(
    verdicts: Sequence[JudgeVerdict],
    embedder: Embedder,
    cluster_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    n_bootstrap: int = 200,
    ari_threshold: float = 0.6,
    rng_seed: int | None = 42,
) -> tuple[bool, float, np.ndarray]:
    """Bootstrap stability test for irreducible structural disagreement.

    Null hypothesis (H0):
        Judges' rationales are i.i.d. draws from a single distribution; no
        structural sub-groups exist.

    Under H0, bootstrap-resampled clustering should produce **unstable** label
    assignments (low Adjusted Rand Index against the observed clustering).

    Decision rule:
        - Cluster the full sample → observed labels.
        - Bootstrap resample ``n_bootstrap`` times; recluster each resample;
          compute ARI vs the observed labels restricted to the resample indices.
        - Mean ARI ≥ ``ari_threshold`` **and** ≥2 non-noise clusters →
          ``is_irreducible = True``.

    False positive rate (FPR):
        Verified < 5% under H0 in ``tests/property/test_is_irreducible_fpr.py``
        for single-Gaussian score data with the default settings.

    Returns:
        Tuple ``(is_irreducible, mean_ari, observed_labels)``.
    """
    from polyphonic_eval.cluster import default_cluster_fn

    cf = cluster_fn if cluster_fn is not None else default_cluster_fn()
    rationales = [v.rationale or "" for v in verdicts]
    n = len(rationales)

    if n < 3:
        embeddings = embedder.embed(rationales) if rationales else np.zeros((0, 1))
        observed = cf(embeddings) if n > 0 else np.zeros(0, dtype=np.int64)
        return False, 0.0, observed

    embeddings = embedder.embed(rationales)
    observed = cf(embeddings)
    n_clusters = len({lbl for lbl in observed.tolist() if lbl >= 0})

    if n_clusters < 2:
        return False, 0.0, observed

    rng = np.random.default_rng(rng_seed)
    aris: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(set(idx.tolist())) < 2:
            continue
        boot_labels = cf(embeddings[idx])
        obs_subset = observed[idx]
        try:
            ari = float(adjusted_rand_score(obs_subset, boot_labels))
        except (ValueError, ZeroDivisionError):
            continue
        aris.append(ari)

    if not aris:
        return False, 0.0, observed

    mean_ari = float(np.mean(aris))
    return bool(mean_ari >= ari_threshold), mean_ari, observed
