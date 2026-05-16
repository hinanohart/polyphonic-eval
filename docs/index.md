# polyphonic-eval

Typed multi-judge LLM evaluation that preserves disagreement structure.

- **What**: ``aggregate(verdicts)`` returns a ``PolyphonicResult`` with typed
  consensus, irreducible disagreement, and a disagreement-spectrum index.
- **Why**: averaging or majority-voting throws away minority signals that an
  ensemble could otherwise surface. Treat disagreement as data.
- **How**: HDBSCAN clustering over caller-pluggable embeddings, bootstrap
  stability test for irreducibility, opt-in scalar collapse.

## Get started

- [Quickstart](quickstart.md)
- [API reference](api.md)
- Adapters: [lm-evaluation-harness](adapters/lm_eval.md), [LangGraph](adapters/langgraph.md)
- [Migration from majority/mean aggregators](migration.md)

## Background

- [Theory and naming](theory.md) — the Bakhtin reference is intentional but
  not load-bearing for use.

## Design records

- [0001 — No consensus collapse](design/0001-no-consensus-collapse.md)
- [0002 — Jargon ceiling](design/0002-jargon-ceiling.md)
- [0003 — Embedder protocol](design/0003-embedder-protocol.md)
