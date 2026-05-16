"""Adapter for ``lm-evaluation-harness``.

The harness's standard aggregation contract collapses metrics to scalars. This
adapter preserves the full ``PolyphonicResult`` alongside, and exposes a thin
scalar so the harness's logging machinery still works.

Usage in a custom task YAML::

    metric_list:
      - metric: !function polyphonic_eval.adapters.lm_eval.polyphonic_metric
        aggregation: !function polyphonic_eval.adapters.lm_eval.aggregate_polyphonic
        higher_is_better: true
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast

from polyphonic_eval.core import aggregate
from polyphonic_eval.types import JudgeVerdict, PolyphonicResult

_VALID_POLICIES = ("mean", "majority")


def polyphonic_metric(item: Any) -> dict[str, Any]:
    """Per-item metric: returns the raw verdicts dict for downstream aggregation.

    ``item`` is the task's per-doc output; the convention here is that the task
    populates ``item['verdicts']`` with a list of judge dicts. The metric simply
    passes that through; aggregation happens in :func:`aggregate_polyphonic`.
    """
    if not isinstance(item, dict) or "verdicts" not in item:
        raise ValueError(
            "polyphonic_metric expects item['verdicts'] = list of JudgeVerdict-shaped dicts."
        )
    return {"verdicts": item["verdicts"], "item_id": item.get("item_id", "unknown")}


def aggregate_polyphonic(
    items: Sequence[dict[str, Any]],
    *,
    scalar_policy: str = "mean",
) -> dict[str, Any]:
    """Harness-side aggregation. Computes a typed PolyphonicResult per item AND
    returns a single scalar score (mean of per-item scalar collapses) so the
    harness's table-formatter has something numeric to display.

    The full per-item ``PolyphonicResult`` JSON is included in the return dict
    under ``"polyphonic_results"`` for downstream consumption.
    """
    if scalar_policy == "refuse":
        raise TypeError(
            "aggregate_polyphonic was called with scalar_policy='refuse', but "
            "lm-evaluation-harness expects a scalar. Pass 'mean' or 'majority'."
        )
    if scalar_policy not in _VALID_POLICIES:
        raise ValueError(f"scalar_policy must be one of {_VALID_POLICIES}, got {scalar_policy!r}.")
    policy_lit = cast(Literal["mean", "majority"], scalar_policy)

    results: list[PolyphonicResult] = []
    for raw in items:
        verdicts = [JudgeVerdict.model_validate(v) for v in raw["verdicts"]]
        results.append(aggregate(verdicts, item_id=raw.get("item_id", "unknown")))

    scalars = [r.to_scalar(policy=policy_lit) for r in results]
    mean_scalar = sum(scalars) / len(scalars) if scalars else 0.0
    return {
        "score": mean_scalar,
        "polyphonic_results": [r.model_dump() for r in results],
        "n_items": len(results),
        "scalar_policy": scalar_policy,
    }


def list_judge_verdicts_from_juries(jury_outputs: Iterable[dict[str, Any]]) -> list[JudgeVerdict]:
    """Convenience: convert a sequence of plain dicts into ``JudgeVerdict`` objects."""
    return [JudgeVerdict.model_validate(j) for j in jury_outputs]


# Deprecated alias (typo: "jurys"). Will be removed in v0.2.0.
list_judge_verdicts_from_jurys = list_judge_verdicts_from_juries
