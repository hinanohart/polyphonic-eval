"""Tests for JSON / JSONL I/O round trips."""

from __future__ import annotations

from pathlib import Path

import pytest

from polyphonic_eval import (
    ConsensusClaim,
    DisagreementCluster,
    IrreducibleDisagreement,
    JudgeVerdict,
    PolyphonicResult,
    ScoreSummary,
)
from polyphonic_eval.io import from_json, read_jsonl, to_json, write_jsonl


def _make_result(item_id: str = "x") -> PolyphonicResult:
    ss = ScoreSummary(mean=0.5, median=0.5, stdev=None, n=1)
    cluster = DisagreementCluster(
        cluster_id=0,
        members=("j1",),
        centroid_rationale="r",
        score_summary=ss,
        weight=1.0,
        keywords=("k",),
    )
    return PolyphonicResult(
        item_id=item_id,
        verdicts=(JudgeVerdict(judge_id="j1", score=0.5, rationale="r"),),
        consensus=ConsensusClaim(
            has_consensus=True, agreeing_judges=("j1",), agreement_strength=1.0
        ),
        disagreement=IrreducibleDisagreement(
            is_irreducible=False,
            clusters=(cluster,),
            minority_clusters=(),
            bootstrap_stability=0.0,
            explanation="ok",
        ),
        disagreement_spectrum=0.0,
        n_judges=1,
    )


def test_json_roundtrip() -> None:
    original = _make_result()
    restored = from_json(to_json(original))
    assert restored == original


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    a, b = _make_result("a"), _make_result("b")
    n = write_jsonl(path, [a, b])
    assert n == 2
    out = list(read_jsonl(path))
    assert len(out) == 2
    assert {r.item_id for r in out} == {"a", "b"}


def test_jsonl_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert list(read_jsonl(path)) == []


def test_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "with_blanks.jsonl"
    payload = to_json(_make_result())
    path.write_text(f"\n{payload}\n\n{payload}\n")
    out = list(read_jsonl(path))
    assert len(out) == 2


def test_json_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="JSON"):
        from_json("{not valid json")
