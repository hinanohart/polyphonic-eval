# Contributing to polyphonic-eval

Thanks for your interest. This document captures the minimum to get a change merged.

## Quick loop

```bash
git clone https://github.com/hinanohart/polyphonic-eval
cd polyphonic-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,langgraph,lm-eval]"
pytest
ruff check . && ruff format --check . && mypy src/
```

## Design constraints (please read before proposing API changes)

1. **No silent scalar collapse.** `PolyphonicResult` has no `__float__`. `to_scalar(policy='refuse')` must keep raising `TypeError`. This is the load-bearing forcing constraint of the library — proposals to relax it need an ADR.
2. **Public API jargon ceiling.** No new Bakhtin/literary-critical terminology in public function names. Use ordinary "disagreement / consensus / cluster" vocabulary. Theory references live in `docs/theory.md`.
3. **Schema versioning.** Wire format changes increment `schema_version` on `PolyphonicResult` and add a migration note in `docs/migration.md`.
4. **Embedder choice is public.** Don't hide embedder selection inside internal defaults — always allow `cluster_fn` / `embedder` injection.

## Tests required

- Unit tests for new modules.
- For new invariants: a property-based test under `tests/property/` using hypothesis. The codebase enforces `--cov-fail-under=85`, but coverage is not a substitute for property tests on invariants.

## Commit / PR style

- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`).
- One logical change per PR.
- Link relevant ADRs in the description.

## Releasing

Maintainers only. Tag `vX.Y.Z` triggers OIDC PyPI publish via `.github/workflows/publish.yml`.

## Code of Conduct

Participation is governed by `CODE_OF_CONDUCT.md`.
