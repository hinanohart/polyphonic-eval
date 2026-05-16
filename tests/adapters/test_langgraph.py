"""Tests for the LangGraph reducer."""

from __future__ import annotations

from polyphonic_eval import JudgeVerdict
from polyphonic_eval.adapters.langgraph import finalize_polyphonic, polyphonic_reducer


def test_reducer_single_verdict() -> None:
    new = JudgeVerdict(judge_id="j1", score=0.5)
    out = polyphonic_reducer(None, new)
    assert len(out) == 1
    assert out[0].judge_id == "j1"


def test_reducer_appends() -> None:
    v1 = JudgeVerdict(judge_id="j1", score=0.5)
    v2 = JudgeVerdict(judge_id="j2", score=0.9)
    out = polyphonic_reducer([v1], v2)
    assert len(out) == 2
    assert [v.judge_id for v in out] == ["j1", "j2"]


def test_reducer_appends_list() -> None:
    v1 = JudgeVerdict(judge_id="j1", score=0.5)
    v2 = JudgeVerdict(judge_id="j2", score=0.9)
    v3 = JudgeVerdict(judge_id="j3", score=0.1)
    out = polyphonic_reducer([v1], [v2, v3])
    assert [v.judge_id for v in out] == ["j1", "j2", "j3"]


def test_reducer_does_not_mutate_input() -> None:
    base = [JudgeVerdict(judge_id="j1", score=0.5)]
    polyphonic_reducer(base, JudgeVerdict(judge_id="j2", score=0.9))
    assert len(base) == 1  # unchanged


def test_finalize_polyphonic_runs(deterministic_embedder, monkeypatch, unanimous_verdicts) -> None:
    import polyphonic_eval.core as core_mod

    monkeypatch.setattr(core_mod, "default_embedder", lambda: deterministic_embedder)
    result = finalize_polyphonic(unanimous_verdicts, item_id="lg")
    assert result.item_id == "lg"
    assert result.n_judges == len(unanimous_verdicts)
