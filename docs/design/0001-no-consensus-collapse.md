# ADR 0001 — No consensus collapse

- Status: accepted
- Date: 2026-05-17

## Context

Standard multi-rater aggregators (``mean``, ``majority``, weighted variants)
project an ordered set of verdicts onto a single scalar. Information loss is
inherent. For LLM evaluation specifically, the information that is lost often
includes:

- minority-but-correct safety concerns
- minority-but-correct factuality flags
- structural bias between judges (one model systematically scores higher on
  long answers, etc.)

There is a large body of evidence in the multi-agent-debate literature
(Free-MAD, DMAD, X-MAS, etc.) that ensemble methods that *preserve* this
structure improve evaluation quality. The standard library landscape does not
expose a type-level way to keep that structure intact at the aggregation
layer.

## Decision

The public ``PolyphonicResult`` type:

1. Does **not** define ``__float__`` or ``__int__``. ``float(result)`` raises.
2. Provides ``to_scalar(policy=...)`` with default ``policy="refuse"`` that
   raises ``TypeError`` with an explanatory message.
3. Carries the full structured disagreement (``IrreducibleDisagreement``,
   ``ConsensusClaim``, per-cluster summaries) as required fields.

This makes accidental scalar collapse a *type error*, not a silent quality
regression.

## Consequences

**Positive**:

- Downstream code that calls ``result.to_scalar("mean")`` is greppable and
  explicit. The decision to collapse is auditable.
- Linters and code review can spot ``to_scalar()`` with no policy argument
  and reject it.

**Negative**:

- Existing pipelines that consume ``mean(metric_values)`` need a one-line
  change to call ``to_scalar(policy="mean")``.
- The aggregator can't be a drop-in replacement for arbitrary numeric metric
  APIs without an explicit collapse step.

## Alternatives considered

- **Implicit ``__float__`` returning the mean.** Rejected: defeats the entire
  purpose of the library.
- **``policy="mean"`` as default.** Rejected: makes the safe path the easy
  path, but doesn't surface the decision in the call site.
- **``policy="refuse"`` only as a flag, no method.** Rejected: removing the
  method removes the affordance; callers would instead pass ``mean(scores)``
  inline and the structure would never be inspected.
