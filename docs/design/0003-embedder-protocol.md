# ADR 0003 — Embedder protocol and reproducibility

- Status: accepted
- Date: 2026-05-17

## Context

The library clusters judges by the semantic content of their rationales. The
choice of embedding model **dominates** the cluster structure: a swap from
``all-MiniLM-L6-v2`` to a larger encoder can move a borderline case from
"two clusters, irreducible" to "one cluster, no disagreement". For evaluation
pipelines this matters: results must be reproducible across runs.

Two failure modes to avoid:

1. **Hidden default**: embedder choice baked into ``aggregate`` without
   surfacing it. Two users running the same code on the same data get
   different results because of an upgrade in a transitive dependency.
2. **Forced explicitness**: requiring every caller to construct an embedder.
   Hurts onboarding ergonomics.

## Decision

- Public ``Embedder`` Protocol: anything with ``embed(texts) -> np.ndarray``.
- ``aggregate`` accepts ``embedder`` as a keyword argument.
- When ``embedder is None``, ``default_embedder()`` lazily constructs a
  ``sentence-transformers`` wrapper over ``all-MiniLM-L6-v2``. This is
  cached process-wide.
- ``cluster_fn`` is a *separate* injection point so callers can swap the
  clustering algorithm without rebuilding the embedder.
- README and docstring **explicitly warn** that embedder choice dominates
  results; for reproducible eval pipelines, the embedder must be pinned.

## Consequences

**Positive**:

- The default works out of the box (``pip install polyphonic-eval[embed]``).
- Reproducibility is achievable: pass an explicit ``embedder``.
- The protocol is small enough to satisfy with one line of glue.

**Negative**:

- A user who skips the documentation can be surprised when a
  ``sentence-transformers`` update changes their clustering. Mitigated by
  the README warning and by pinning the model name in ``embed.py``.

## Alternatives considered

- **No default embedder**: rejected; hurts adoption.
- **Multiple default embedders auto-selected per language**: deferred to a
  future version; for v0.1.0 we pin a single English-only default.
