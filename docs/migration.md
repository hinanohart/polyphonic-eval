# Migration from scalar aggregators

If you currently aggregate multi-rater data with ``mean`` or ``majority``,
moving to polyphonic-eval is intentionally a small step.

## From ``mean``

Before:

```python
def aggregate_mean(scores: list[float]) -> float:
    return sum(scores) / len(scores)

quality = aggregate_mean([v.score for v in verdicts])
```

After:

```python
from polyphonic_eval import aggregate

result = aggregate(verdicts, item_id=item_id)
quality = result.to_scalar(policy="mean")   # same number you had before
disagreement = result.disagreement_spectrum # new signal you didn't have
```

The numeric output is identical; you now also have the structured disagreement
on the side.

## From ``majority``

Before:

```python
from collections import Counter
def aggregate_majority(labels: list[str]) -> str:
    return Counter(labels).most_common(1)[0][0]
```

After:

```python
result = aggregate(verdicts, item_id=item_id)
if result.consensus.has_consensus:
    label = result.consensus.consensus_label
else:
    # Decide explicitly how you want to handle the lack of consensus.
    label = None
```

This is a deliberate behavior change: the library wants you to **decide**
what to do when judges disagree, not pretend disagreement didn't happen.

## From a custom dict-of-stats aggregator

If you previously rolled your own dict like
``{"mean": ..., "std": ..., "n": ...}``, you can produce a closer drop-in:

```python
result = aggregate(verdicts, item_id=item_id)
stats = {
    "mean": result.to_scalar(policy="mean"),
    "median": result.to_scalar(policy="majority"),
    "n": result.n_judges,
    "consensus": result.consensus.has_consensus,
    "spectrum": result.disagreement_spectrum,
    "irreducible": result.disagreement.is_irreducible,
}
```

## Schema versioning

``PolyphonicResult.schema_version`` is ``"1"`` for v0.1.x. If we change the
wire format we will bump this number and document the migration here.
