"""Command-line entry point: ``polyphonic-eval aggregate``.

Optional dependency on ``typer``. The ``[cli]`` extra pulls it in. If typer is
not present, the entry point raises a helpful ``ImportError`` when invoked.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from polyphonic_eval.core import aggregate
from polyphonic_eval.io.json_codec import to_json
from polyphonic_eval.types import JudgeVerdict


def _run(input_path: str, item_id: str, output_path: str | None) -> int:
    """Shared implementation used by both typer and the no-typer fallback."""
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


try:
    import typer
except ImportError:  # pragma: no cover
    app: object | None = None
else:
    app = typer.Typer(
        help="polyphonic-eval: typed multi-judge LLM evaluation.",
        no_args_is_help=True,
        add_completion=False,
    )

    @app.command(name="aggregate")
    def aggregate_cli(
        input_path: str = typer.Argument(..., help="Path to a JSON array of verdicts."),
        item_id: str = typer.Option("cli-item", "--item-id", help="Item identifier."),
        output: str | None = typer.Option(
            None, "--output", "-o", help="Write JSON output to this path."
        ),
    ) -> None:
        """Aggregate verdicts from a JSON file."""
        raise typer.Exit(_run(input_path, item_id, output))


def main(argv: Sequence[str] | None = None) -> int:
    """No-typer fallback entry. Honors ``argv[1]`` as input path."""
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        sys.stdout.write(
            "usage: polyphonic-eval aggregate INPUT [--item-id ID] [--output PATH]\n"
            "  Install the [cli] extra for the full typer-based CLI.\n"
        )
        return 0
    if args[0] != "aggregate" or len(args) < 2:
        sys.stderr.write("error: expected 'aggregate INPUT'\n")
        return 2
    return _run(args[1], "cli-item", None)


if __name__ == "__main__":
    raise SystemExit(main())
