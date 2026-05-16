"""LangGraph reducer that preserves every judge's verdict.

Use it in a state schema::

    from typing import Annotated, TypedDict
    from polyphonic_eval import JudgeVerdict
    from polyphonic_eval.adapters.langgraph import polyphonic_reducer

    class JuryState(TypedDict):
        verdicts: Annotated[list[JudgeVerdict], polyphonic_reducer]

The reducer **appends** rather than overwrites, so each judge node's vote
survives into the final aggregation step. Call :func:`finalize_polyphonic` at
the terminal node to compute a typed ``PolyphonicResult``.
"""

from __future__ import annotations

from collections.abc import Sequence

from polyphonic_eval.core import aggregate
from polyphonic_eval.types import JudgeVerdict, PolyphonicResult


def polyphonic_reducer(
    current: list[JudgeVerdict] | None,
    new: JudgeVerdict | list[JudgeVerdict] | tuple[JudgeVerdict, ...],
) -> list[JudgeVerdict]:
    """Append-style reducer for ``JudgeVerdict`` lists.

    LangGraph reducers receive ``(current_state_value, new_state_value)`` and
    return the merged result. Passing a single verdict or a list both work.
    """
    base: list[JudgeVerdict] = list(current) if current else []
    if isinstance(new, JudgeVerdict):
        return [*base, new]
    return [*base, *list(new)]


def finalize_polyphonic(
    verdicts: Sequence[JudgeVerdict],
    *,
    item_id: str = "langgraph-item",
) -> PolyphonicResult:
    """Convenience wrapper around :func:`polyphonic_eval.aggregate` for use in
    a LangGraph terminal node.
    """
    return aggregate(verdicts, item_id=item_id)
