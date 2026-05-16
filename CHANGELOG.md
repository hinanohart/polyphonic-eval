# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-05-17

### Added

- Core `aggregate()` function returning typed `PolyphonicResult`.
- Pydantic v2 model layer: `JudgeVerdict`, `DisagreementCluster`, `ConsensusClaim`, `IrreducibleDisagreement`, `PolyphonicResult`, `ScoreSummary`.
- `Embedder` protocol with lazy `sentence-transformers/all-MiniLM-L6-v2` default; `cluster_fn` injection point for custom clustering.
- HDBSCAN-based clustering with cosine metric and minimum cluster size of 2.
- Krippendorff alpha consensus measurement (`consensus.py`).
- `disagreement_spectrum` index (Simpson concentration × normalized inter-cluster semantic distance).
- `is_irreducible` falsifiable boundary with bootstrap stability test (ARI threshold = 0.6, N=200 resamples) and documented null hypothesis.
- `to_scalar(policy=...)` collapse with default `"refuse"` raising `TypeError`.
- Adapter for `lm-evaluation-harness` (custom aggregator).
- Adapter for LangGraph (`polyphonic_reducer`).
- CLI: `polyphonic-eval aggregate`.
- Property-based test suite (hypothesis, 1000 examples) verifying no-consensus-collapse invariant and FPR < 5% for `is_irreducible`.
- 12-cell CI matrix (Python 3.10–3.13 × Ubuntu/macOS/Windows).
- OIDC-based PyPI Trusted Publishing.

### Notes

- This is an alpha release. The wire format of `PolyphonicResult` carries `schema_version: "1"`; future releases will track schema evolution explicitly.
- Embedder choice dominates cluster structure. For reproducible eval pipelines, pin a specific embedder.
- v0.1.0 ships without golden fixtures (`tests/golden/`) or formal benchmarks (`benchmarks/`); both are slated for v0.1.1.
- `consensus.py` uses a simple score-spread-tolerance test in v0.1.0. A Krippendorff α implementation is on the v0.2.0 roadmap.
