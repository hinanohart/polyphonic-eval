"""Top-level ``aggregate`` and ``PolyphonicAggregator``."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from statistics import mean as stat_mean
from statistics import median as stat_median
from statistics import stdev as stat_stdev

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.metrics.pairwise import cosine_distances

from polyphonic_eval.cluster import default_cluster_fn
from polyphonic_eval.consensus import compute_consensus
from polyphonic_eval.embed import Embedder, default_embedder
from polyphonic_eval.irreducible import is_irreducible
from polyphonic_eval.spectrum import disagreement_spectrum
from polyphonic_eval.types import (
    DisagreementCluster,
    IrreducibleDisagreement,
    JudgeVerdict,
    PolyphonicResult,
    ScoreSummary,
)

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "if",
        "so",
        "not",
        "no",
        "yes",
        "you",
        "your",
        "they",
        "them",
        "their",
        "we",
        "our",
        "i",
        "me",
        "my",
    }
)


class AggregatorConfig(BaseModel):
    """Tunable thresholds for ``aggregate``. All immutable."""

    model_config = ConfigDict(frozen=True)

    min_cluster_size: int = 2
    bootstrap_iters: int = 200
    ari_threshold: float = 0.6
    consensus_score_tolerance: float = 0.15
    keyword_top_k: int = 5
    minority_weight_cutoff: float = 0.5
    rng_seed: int | None = 42


def aggregate(
    verdicts: Sequence[JudgeVerdict],
    *,
    item_id: str,
    embedder: Embedder | None = None,
    cluster_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    config: AggregatorConfig | None = None,
) -> PolyphonicResult:
    """Aggregate multi-judge verdicts into a typed ``PolyphonicResult``.

    The function refuses to collapse the result to a scalar. Use
    :meth:`PolyphonicResult.to_scalar` with an explicit policy to opt in.
    """
    if not verdicts:
        raise ValueError("aggregate() requires at least one verdict.")

    cfg = config if config is not None else AggregatorConfig()
    emb = embedder if embedder is not None else default_embedder()
    cf = (
        cluster_fn
        if cluster_fn is not None
        else default_cluster_fn(min_cluster_size=cfg.min_cluster_size)
    )

    verdicts_t = tuple(verdicts)
    rationales = [v.rationale or "" for v in verdicts_t]

    embeddings = emb.embed(rationales)

    is_irr, stability, labels = is_irreducible(
        verdicts_t,
        emb,
        cf,
        n_bootstrap=cfg.bootstrap_iters,
        ari_threshold=cfg.ari_threshold,
        rng_seed=cfg.rng_seed,
        embeddings=embeddings,
    )

    clusters = _build_clusters(verdicts_t, embeddings, labels, top_k=cfg.keyword_top_k)
    consensus = compute_consensus(
        verdicts_t,
        score_tolerance=cfg.consensus_score_tolerance,
    )

    pair_dist = _cluster_pairwise_distance(verdicts_t, embeddings, clusters)
    spectrum = disagreement_spectrum(clusters, pair_dist)

    minority_ids = tuple(c.cluster_id for c in clusters if c.weight < cfg.minority_weight_cutoff)

    return PolyphonicResult(
        item_id=item_id,
        verdicts=verdicts_t,
        consensus=consensus,
        disagreement=IrreducibleDisagreement(
            is_irreducible=is_irr,
            clusters=tuple(clusters),
            minority_clusters=minority_ids,
            bootstrap_stability=stability,
            explanation=_explain_irreducible(
                is_irreducible=is_irr,
                stability=stability,
                n_clusters=len(clusters),
                threshold=cfg.ari_threshold,
            ),
        ),
        disagreement_spectrum=spectrum,
        n_judges=len(verdicts_t),
    )


class PolyphonicAggregator:
    """Stateful aggregator that caches embedder/config across calls."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        cluster_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        config: AggregatorConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._cluster_fn = cluster_fn
        self._config = config if config is not None else AggregatorConfig()

    def aggregate(
        self,
        verdicts: Sequence[JudgeVerdict],
        *,
        item_id: str,
    ) -> PolyphonicResult:
        return aggregate(
            verdicts,
            item_id=item_id,
            embedder=self._embedder,
            cluster_fn=self._cluster_fn,
            config=self._config,
        )


def _build_clusters(
    verdicts: tuple[JudgeVerdict, ...],
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    top_k: int,
) -> list[DisagreementCluster]:
    """Group verdict indices by HDBSCAN label.

    HDBSCAN noise points (label ``-1``) are gathered into a single
    "unclustered" cluster rather than promoted to one synthetic singleton each.
    The old behaviour inflated ``disagreement_spectrum`` whenever the embedder
    produced any noise — unanimous panels of 4+ paraphrased rationales would
    show 4 fake clusters. See critic audit 2026-05-17.
    """
    groups: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels.tolist()):
        groups[int(lbl)].append(i)

    real = {k: v for k, v in groups.items() if k >= 0}
    noise = groups.get(-1, [])

    out: list[DisagreementCluster] = []
    total = len(verdicts)

    if not real:
        # All-noise: HDBSCAN found no structure. Represent as one "unclustered"
        # cluster carrying every judge — concentration term will be 1, giving
        # spectrum 0 (no detected disagreement).
        if noise:
            out.append(_cluster_from_indices(0, noise, verdicts, embeddings, total, top_k))
        return out

    for cid in sorted(real):
        out.append(_cluster_from_indices(cid, real[cid], verdicts, embeddings, total, top_k))
    if noise:
        unclustered_id = max(real) + 1
        out.append(_cluster_from_indices(unclustered_id, noise, verdicts, embeddings, total, top_k))
    return out


def _cluster_from_indices(
    cluster_id: int,
    indices: list[int],
    verdicts: tuple[JudgeVerdict, ...],
    embeddings: np.ndarray,
    total: int,
    top_k: int,
) -> DisagreementCluster:
    members = tuple(verdicts[i].judge_id for i in indices)
    sub = embeddings[indices]
    centroid = sub.mean(axis=0)
    dists = np.linalg.norm(sub - centroid, axis=1)
    centroid_idx = indices[int(np.argmin(dists))]
    centroid_rationale = verdicts[centroid_idx].rationale or ""

    scores: list[float] = []
    labels: list[str] = []
    for i in indices:
        s = verdicts[i].score
        if s is not None:
            scores.append(s)
        lbl = verdicts[i].label
        if lbl is not None:
            labels.append(lbl)

    ss = ScoreSummary(
        mean=float(stat_mean(scores)) if scores else None,
        median=float(stat_median(scores)) if scores else None,
        stdev=float(stat_stdev(scores)) if len(scores) > 1 else None,
        n=len(indices),
        label_distribution=dict(Counter(labels)) if labels else None,
    )

    weight = (len(indices) / total) if total > 0 else 0.0
    keywords = tuple(_extract_keywords(verdicts[i].rationale or "" for i in indices)[:top_k])

    return DisagreementCluster(
        cluster_id=cluster_id,
        members=members,
        centroid_rationale=centroid_rationale,
        score_summary=ss,
        weight=weight,
        keywords=keywords,
    )


def _cluster_pairwise_distance(
    verdicts: tuple[JudgeVerdict, ...],
    embeddings: np.ndarray,
    clusters: Sequence[DisagreementCluster],
) -> np.ndarray:
    if len(clusters) <= 1:
        return np.zeros((max(1, len(clusters)), max(1, len(clusters))))
    judge_to_idx = {v.judge_id: i for i, v in enumerate(verdicts)}
    centroids: list[np.ndarray] = []
    for c in clusters:
        idxs = [judge_to_idx[j] for j in c.members if j in judge_to_idx]
        if idxs:
            centroids.append(embeddings[idxs].mean(axis=0))
    if len(centroids) < 2:
        return np.zeros((1, 1))
    centroid_arr = np.vstack(centroids)
    return np.asarray(cosine_distances(centroid_arr))


def _extract_keywords(rationales: Iterable[str]) -> list[str]:
    """Trivial whitespace + stopword filter. TF only; no IDF in v0.1.0."""
    words: list[str] = []
    for r in rationales:
        words.extend(re.findall(r"[a-zA-Z]{3,}", r.lower()))
    counter = Counter(w for w in words if w not in _STOPWORDS)
    return [w for w, _ in counter.most_common(20)]


def _explain_irreducible(
    *,
    is_irreducible: bool,
    stability: float,
    n_clusters: int,
    threshold: float,
) -> str:
    if is_irreducible:
        return (
            f"Bootstrap stability {stability:.2f} >= threshold {threshold:.2f} "
            f"with {n_clusters} clusters: judges form structurally distinct groups "
            f"that survive resampling."
        )
    if stability < threshold:
        return (
            f"Bootstrap stability {stability:.2f} < threshold {threshold:.2f}: "
            f"cluster structure is unstable under resampling, consistent with H0 "
            f"of single-population draws."
        )
    return (
        f"Stability {stability:.2f} >= threshold but only {n_clusters} cluster(s) "
        f"detected: no structural disagreement to call irreducible."
    )
