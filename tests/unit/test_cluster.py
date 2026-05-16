"""Tests for the HDBSCAN-backed default cluster_fn."""

from __future__ import annotations

import numpy as np

from polyphonic_eval.cluster import default_cluster_fn


def test_default_cluster_fn_returns_labels_for_distinct_groups() -> None:
    # Two well-separated groups in 4-D, three points each.
    pts = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.95, 0.05, 0.0, 0.0],
            [0.92, 0.08, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.95, 0.05, 0.0],
            [0.0, 0.92, 0.08, 0.0],
        ],
        dtype=np.float32,
    )
    cf = default_cluster_fn(min_cluster_size=2)
    labels = cf(pts)
    assert labels.shape == (6,)
    # HDBSCAN may label noise points -1; we still expect at least two real clusters
    # for well-separated input.
    real = {lbl for lbl in labels.tolist() if lbl >= 0}
    assert len(real) >= 2


def test_default_cluster_fn_single_point() -> None:
    cf = default_cluster_fn()
    labels = cf(np.array([[1.0, 0.0]], dtype=np.float32))
    assert labels.shape == (1,)


def test_default_cluster_fn_returns_int64() -> None:
    cf = default_cluster_fn()
    pts = np.eye(4, dtype=np.float32)
    labels = cf(pts)
    assert labels.dtype == np.int64
