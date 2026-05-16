"""Tests for the CLI fallback path (typer-independent)."""

from __future__ import annotations

import json
from pathlib import Path

from polyphonic_eval.cli import main


def test_cli_help_zero_exit(capsys) -> None:
    rc = main(["--help"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "polyphonic-eval" in captured.out


def test_cli_aggregate_basic(tmp_path: Path, capsys, monkeypatch) -> None:
    # Build a tiny verdicts JSON
    verdicts = [
        {"judge_id": "j1", "score": 0.9, "rationale": "helpful good answer"},
        {"judge_id": "j2", "score": 0.85, "rationale": "useful answer overall"},
        {"judge_id": "j3", "score": 0.9, "rationale": "good and helpful response"},
    ]
    input_path = tmp_path / "verdicts.json"
    input_path.write_text(json.dumps(verdicts), encoding="utf-8")

    # Patch default_embedder to avoid sentence-transformers download
    import polyphonic_eval.core as core_mod
    from tests.conftest import DeterministicHashEmbedder

    monkeypatch.setattr(core_mod, "default_embedder", lambda: DeterministicHashEmbedder())

    rc = main(["aggregate", str(input_path)])
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["item_id"] == "cli-item"
    assert len(data["verdicts"]) == 3


def test_cli_unknown_command_error() -> None:
    rc = main(["unknown"])
    assert rc == 2
