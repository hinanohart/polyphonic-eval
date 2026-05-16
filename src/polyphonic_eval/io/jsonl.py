"""Append-friendly JSONL I/O for ``PolyphonicResult`` batches."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from polyphonic_eval.types import PolyphonicResult


def write_jsonl(path: str | Path, results: Iterable[PolyphonicResult]) -> int:
    """Write results to a JSONL file, one ``PolyphonicResult`` per line.

    Returns the number of records written.
    """
    p = Path(path)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json())
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[PolyphonicResult]:
    """Yield ``PolyphonicResult`` objects from a JSONL file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield PolyphonicResult.model_validate_json(line)
