"""Tests for the lm-evaluation-harness adapter."""

from __future__ import annotations

import pytest

from polyphonic_eval.adapters.lm_eval import (
    aggregate_polyphonic,
    list_judge_verdicts_from_juries,
    list_judge_verdicts_from_jurys,
    polyphonic_metric,
)


def test_list_judge_verdicts_from_juries() -> None:
    out = list_judge_verdicts_from_juries(
        [
            {"judge_id": "j1", "score": 0.5},
            {"judge_id": "j2", "score": 0.7, "rationale": "ok"},
        ]
    )
    assert len(out) == 2
    assert out[1].rationale == "ok"


def test_jurys_alias_still_works() -> None:
    # Back-compat alias kept for v0.1.x; deprecated in v0.2.0.
    assert list_judge_verdicts_from_jurys is list_judge_verdicts_from_juries


def test_aggregate_polyphonic_invalid_policy() -> None:
    with pytest.raises(ValueError, match="scalar_policy"):
        aggregate_polyphonic([], scalar_policy="median")


def test_polyphonic_metric_passes_verdicts_through() -> None:
    item = {
        "item_id": "x",
        "verdicts": [
            {"judge_id": "j1", "score": 0.9, "rationale": "good"},
            {"judge_id": "j2", "score": 0.1, "rationale": "bad"},
        ],
    }
    out = polyphonic_metric(item)
    assert out["item_id"] == "x"
    assert len(out["verdicts"]) == 2


def test_polyphonic_metric_rejects_unshaped_input() -> None:
    with pytest.raises(ValueError, match="verdicts"):
        polyphonic_metric({"item_id": "x"})


def test_aggregate_polyphonic_default_mean(deterministic_embedder, monkeypatch) -> None:
    import polyphonic_eval.core as core_mod

    monkeypatch.setattr(core_mod, "default_embedder", lambda: deterministic_embedder)
    items = [
        {
            "item_id": "x1",
            "verdicts": [
                {"judge_id": "j1", "score": 0.9, "rationale": "good"},
                {"judge_id": "j2", "score": 0.85, "rationale": "good"},
            ],
        },
        {
            "item_id": "x2",
            "verdicts": [
                {"judge_id": "j1", "score": 0.7, "rationale": "ok"},
                {"judge_id": "j2", "score": 0.6, "rationale": "ok"},
            ],
        },
    ]
    out = aggregate_polyphonic(items)
    assert out["n_items"] == 2
    assert out["scalar_policy"] == "mean"
    assert "score" in out
    assert 0 <= out["score"] <= 1
    assert len(out["polyphonic_results"]) == 2


def test_aggregate_polyphonic_refuse_raises() -> None:
    with pytest.raises(TypeError, match="refuse"):
        aggregate_polyphonic([], scalar_policy="refuse")
