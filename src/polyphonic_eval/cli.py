"""Command-line entry point: ``polyphonic-eval aggregate``.

Uses :mod:`argparse` (stdlib) so the CLI works on a bare ``pip install
polyphonic-eval`` with no optional ``[cli]`` extra. ``typer`` is no longer a
runtime dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from polyphonic_eval.core import aggregate
from polyphonic_eval.io.json_codec import to_json
from polyphonic_eval.types import JudgeVerdict


def _run(input_path: str, item_id: str, output_path: str | None) -> int:
    """Shared implementation."""
    p = Path(input_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(
            f"Input JSON must be an array of verdict objects; got {type(payload).__name__}."
        )
    verdicts: list[JudgeVerdict] = [JudgeVerdict.model_validate(v) for v in payload]
    result = aggregate(verdicts, item_id=item_id)
    out = to_json(result)
    if output_path:
        Path(output_path).write_text(out + "\n", encoding="utf-8")
    else:
        sys.stdout.write(out + "\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polyphonic-eval",
        description="polyphonic-eval: typed multi-judge LLM evaluation.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    agg = sub.add_parser("aggregate", help="Aggregate verdicts from a JSON file.")
    agg.add_argument("input_path", help="Path to a JSON array of verdict objects.")
    agg.add_argument("--item-id", default="cli-item", help="Item identifier (default: cli-item).")
    agg.add_argument(
        "-o", "--output", default=None, help="Write JSON output to this path (default: stdout)."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the ``polyphonic-eval`` console script.

    Returns an int return code rather than calling ``sys.exit``. Argparse's own
    ``SystemExit`` (raised on ``--help`` or argument errors) is caught and its
    code propagated, so programmatic callers always get an integer back.
    """
    parser = _build_parser()
    try:
        ns = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    if ns.command == "aggregate":
        return _run(ns.input_path, ns.item_id, ns.output)
    return 2


# Backwards-compat alias so any consumer importing ``cli.app`` still finds a
# callable entry point.
app = main


if __name__ == "__main__":
    raise SystemExit(main())
