# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] — 2026-05-17

### Fixed

- **Silent minority erasure on mixed-modality verdicts.** ``compute_consensus``,
  ``collapse_mean``, and ``collapse_majority`` previously discarded label-only
  judges when the panel also contained score-only judges — the exact failure
  mode the library exists to prevent. They now refuse mixed-modality input
  with a ``TypeError`` (``ConsensusClaim`` returns ``has_consensus=False``).
- **HDBSCAN noise inflation.** ``_build_clusters`` previously promoted every
  HDBSCAN noise point (label ``-1``) to its own synthetic singleton cluster,
  which made ``disagreement_spectrum`` go up purely because of embedder noise.
  Noise is now lumped into one "unclustered" cluster (or returned as a single
  cluster containing every judge when nothing was clustered).
- **Broken ``aggregate`` subcommand.** The documented
  ``polyphonic-eval aggregate verdicts.json`` form failed because the
  ``typer`` single-command app flattened the subcommand into an argument.
  The CLI is now an ``argparse``-based subcommand group (no ``typer``
  dependency).
- **Non-deterministic test embedder.** ``DeterministicHashEmbedder`` used
  Python's ``PYTHONHASHSEED``-randomized ``hash()``; tests passed locally
  and failed sporadically in CI. Switched to ``blake2b``.
- **``test_is_irreducible_fpr`` flake.** The H0 generator's paraphrase pool
  contained lexically distinct fillers, which the hash embedder treated as
  semantically distinct, occasionally letting HDBSCAN find structure under
  H0. Test rewritten to use a single shared rationale (true H0 for a
  bag-of-words embedder).

### Added

- ``__bool__`` on ``PolyphonicResult`` now raises ``TypeError`` (symmetric
  with ``__float__`` / ``__int__`` refusal).
- ``JudgeVerdict.score`` and ``.confidence`` reject ``NaN`` / ``±inf`` via a
  pydantic field validator.
- ``aggregate_polyphonic`` (lm-eval adapter) validates ``scalar_policy``
  before running the full pipeline; unknown values raise ``ValueError``
  instead of crashing inside ``to_scalar``.
- ``list_judge_verdicts_from_juries`` exposed; the old typo
  ``list_judge_verdicts_from_jurys`` is kept as a back-compat alias and
  will be removed in v0.2.0.
- ``AggregatorConfig`` re-exported from the package root.

### Changed

- ``is_irreducible`` accepts an optional precomputed ``embeddings`` argument,
  so ``aggregate`` no longer recomputes embeddings twice per call.
- ``is_irreducible`` clamps ``mean_ari`` to ``[0, 1]`` for the public
  ``bootstrap_stability`` field (raw ARI can be slightly negative on
  pathological inputs; that is reported as ``0.0``).
- README install order now recommends ``[embed]`` first; bare core install
  is annotated as bring-your-own-embedder.
- ``typer`` dropped as a runtime / optional dependency.

### Notes

- This is a bug-fix release; the wire format (``schema_version="1"``) is
  unchanged.

## [0.1.0] — 2026-05-17

### Added

- Core `aggregate()` function returning typed `PolyphonicResult`.
- Pydantic v2 model layer: `JudgeVerdict`, `DisagreementCluster`, `ConsensusClaim`, `IrreducibleDisagreement`, `PolyphonicResult`, `ScoreSummary`.
- `Embedder` protocol with lazy `sentence-transformers/all-MiniLM-L6-v2` default; `cluster_fn` injection point for custom clustering.
- HDBSCAN-based clustering with cosine metric and minimum cluster size of 2.
- Score-spread-tolerance consensus measurement (`consensus.py`); Krippendorff α implementation deferred to v0.2.0 (see Notes).
- `disagreement_spectrum` index (Simpson concentration × normalized inter-cluster semantic distance).
- `is_irreducible` falsifiable boundary with bootstrap stability test (ARI threshold = 0.6, N=200 resamples) and documented null hypothesis.
- `to_scalar(policy=...)` collapse with default `"refuse"` raising `TypeError`.
- Adapter for `lm-evaluation-harness` (custom aggregator).
- Adapter for LangGraph (`polyphonic_reducer`).
- CLI: `polyphonic-eval aggregate`.
- Property-based test suite (hypothesis, 1000 examples) verifying no-consensus-collapse invariant; H0 sanity floor for `is_irreducible`.
- 12-cell CI matrix (Python 3.10–3.13 × Ubuntu/macOS/Windows).
- OIDC-based PyPI Trusted Publishing.

### Notes

- This is an alpha release. The wire format of `PolyphonicResult` carries `schema_version: "1"`; future releases will track schema evolution explicitly.
- Embedder choice dominates cluster structure. For reproducible eval pipelines, pin a specific embedder.
- v0.1.0 ships without golden fixtures (`tests/golden/`) or formal benchmarks (`benchmarks/`); both are slated for v0.2.0.
- `consensus.py` uses a simple score-spread-tolerance test in v0.1.0. A Krippendorff α implementation is on the v0.2.0 roadmap.
- **Correction (post-release):** several v0.1.0 claims were superseded by v0.1.1. The FPR claim above is reframed as a sanity floor; ``compute_consensus`` and the ``collapse_*`` functions now refuse mixed-modality input rather than silently using the score path; HDBSCAN noise no longer inflates ``disagreement_spectrum``; the ``aggregate`` CLI subcommand actually works. See the v0.1.1 section for details.
