"""JSON encode/decode for ``PolyphonicResult`` (and related types)."""

from __future__ import annotations

from polyphonic_eval.types import PolyphonicResult


def to_json(result: PolyphonicResult) -> str:
    """Serialize a ``PolyphonicResult`` to a JSON string (round-trippable)."""
    return result.model_dump_json()


def from_json(payload: str) -> PolyphonicResult:
    """Deserialize a JSON string previously produced by :func:`to_json`."""
    return PolyphonicResult.model_validate_json(payload)
