# ADR 0002 — Jargon ceiling

- Status: accepted
- Date: 2026-05-17

## Context

The library's intellectual seed is Mikhail Bakhtin's concept of *polyphony*.
The temptation when seeding a library on a rich theoretical vocabulary is to
expose that vocabulary in the API: ``HeteroglossiaIndex``, ``ChronotopeFrame``,
``CarnivalInversion``, ``DialogicState``, etc. The history of
literary-critical NLP toolkits is unkind to that choice:

- ReaderBench (Bakhtin-based readability/discourse toolkit, 2010-2018): 7
  GitHub stars, 0 forks. Production use: none observable.
- Multiple "rhetoric"-flavored academic NLP packages have died on the same
  hill — vocabulary that requires a humanities degree to read makes the
  library's docs unreadable to its actual users.

Conversely, libraries that drew on rich theory but used ordinary engineering
vocabulary in the API (e.g., GoF design patterns, named after their function
not their philosophical roots) had no such trouble.

## Decision

Public API names use ordinary disagreement/consensus/cluster vocabulary:

- ``aggregate``, ``ConsensusClaim``, ``DisagreementCluster``,
  ``IrreducibleDisagreement``, ``PolyphonicResult``, ``disagreement_spectrum``,
  ``Embedder``, ``cluster_fn``, ``to_scalar``.

**The single exception** is the package name itself: ``polyphonic_eval``.
This is one occurrence, and it is recoverable via ``pip install``-time tab
completion. After import, the user does not have to spell or pronounce any
literary-critical terms.

Specifically banned from public API surface:

- ``Heteroglossia*`` (originally proposed for the spread index — renamed to
  ``disagreement_spectrum``)
- ``Chronotope*``, ``Carnival*``, ``Dialog(ic|al)*``, ``Addressivity*``,
  ``Outsideness*``, ``Answerability*``, ``Unfinalizability*``.

Background theory lives in ``docs/theory.md``.

## Consequences

**Positive**:

- New users can read the API without prior knowledge. README, quickstart, and
  API reference do not contain any term that requires a footnote.
- Adoption math improves: the curve is set by usefulness, not by tolerance
  for jargon.

**Negative**:

- Some loss of "this is conceptually unique" signal at the surface. Mitigated
  by the package name and by a clear ``docs/theory.md`` for readers who want
  to understand the philosophical heritage.

## Enforcement

- ``CONTRIBUTING.md`` lists this rule as a design constraint reviewers should
  enforce.
- The PR template requires authors to confirm: *"no new Bakhtin/literary
  jargon in public function names"*.
