"""Tests for ``disagreement_spectrum``."""

from __future__ import annotations

import numpy as np

from polyphonic_eval import DisagreementCluster, ScoreSummary
from polyphonic_eval.spectrum import disagreement_spectrum


def _make_cluster(cid: int, members: tuple[str, ...], weight: float) -> DisagreementCluster:
    return DisagreementCluster(
        cluster_id=cid,
        members=members,
        centroid_rationale="r",
        score_summary=ScoreSummary(mean=None, median=None, stdev=None, n=len(members)),
        weight=weight,
        keywords=(),
    )


def test_single_cluster_is_zero() -> None:
    clusters = [_make_cluster(0, ("j1", "j2", "j3"), 1.0)]
    assert disagreement_spectrum(clusters, np.zeros((1, 1))) == 0.0


def test_zero_clusters_is_zero() -> None:
    assert disagreement_spectrum([], np.zeros((1, 1))) == 0.0


def test_balanced_split_high_distance_high_spectrum() -> None:
    c1 = _make_cluster(0, ("j1", "j2"), 0.5)
    c2 = _make_cluster(1, ("j3", "j4"), 0.5)
    pairwise = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    s = disagreement_spectrum([c1, c2], pairwise)
    # Simpson(0.5, 0.5) = 1 - 0.5 = 0.5; mean off-diagonal = 1.0 → 0.5
    assert abs(s - 0.5) < 1e-6


def test_balanced_split_zero_distance_zero_spectrum() -> None:
    c1 = _make_cluster(0, ("j1", "j2"), 0.5)
    c2 = _make_cluster(1, ("j3", "j4"), 0.5)
    pairwise = np.zeros((2, 2), dtype=np.float64)
    s = disagreement_spectrum([c1, c2], pairwise)
    assert s == 0.0


def test_unbalanced_split_lowers_spectrum() -> None:
    c_big = _make_cluster(0, ("j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9"), 0.9)
    c_small = _make_cluster(1, ("j10",), 0.1)
    pairwise = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    s = disagreement_spectrum([c_big, c_small], pairwise)
    # Simpson(0.9, 0.1) = 1 - 0.82 = 0.18
    assert 0.15 < s < 0.21
