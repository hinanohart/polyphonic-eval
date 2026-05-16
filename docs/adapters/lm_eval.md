# lm-evaluation-harness adapter

``polyphonic-eval`` exposes a custom-metric + custom-aggregator pair for
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Install

```bash
pip install polyphonic-eval[lm-eval,embed]
```

## Task YAML

In your task config:

```yaml
metric_list:
  - metric: !function polyphonic_eval.adapters.lm_eval.polyphonic_metric
    aggregation: !function polyphonic_eval.adapters.lm_eval.aggregate_polyphonic
    higher_is_better: true
```

Your task's per-doc output should be a dict with a ``verdicts`` list. Each
entry of ``verdicts`` is the JSON form of a ``JudgeVerdict``:

```json
{
  "item_id": "doc-42",
  "verdicts": [
    {"judge_id": "j1", "score": 0.9, "rationale": "helpful"},
    {"judge_id": "j2", "score": 0.1, "rationale": "factually wrong"}
  ]
}
```

## Output shape

The aggregator returns:

```python
{
  "score": float,                    # mean of per-item scalar collapses
  "polyphonic_results": list[dict],  # full PolyphonicResult JSON per item
  "n_items": int,
  "scalar_policy": "mean" | "majority",
}
```

The harness's table-formatter sees ``score`` (a scalar) and behaves normally.
The full structured results are available under ``polyphonic_results`` for
downstream consumption — e.g. plotting per-cluster keywords, filtering items
with irreducible disagreement for human review, or training a downstream model
on the structure.

## Upstream collaboration

We plan to open a PR/issue on EleutherAI/lm-evaluation-harness within two
weeks of v0.1.0 release proposing first-class support for aggregators that
return structured (dict) outputs rather than scalars. Track the issue in the
repo for upstream-status updates.
