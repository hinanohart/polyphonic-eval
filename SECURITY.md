# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:       |

## Reporting a vulnerability

If you discover a security vulnerability:

1. **Do not open a public GitHub issue.**
2. Open a GitHub Security Advisory: <https://github.com/hinanohart/polyphonic-eval/security/advisories/new>.

You should receive an initial response within 7 days. We will work with you to assess and disclose responsibly.

## Threat model (initial)

This library:

- does **not** make outbound network calls at import-time or in `aggregate()`. The default embedder (`sentence-transformers`) downloads model weights on first use; pin or pre-cache if your environment forbids that.
- accepts arbitrary `rationale` strings from caller-supplied `JudgeVerdict` values. These are passed to embedders (`numpy` array path only). They are not eval'd, deserialized into Python objects, or used as format strings.
- writes nothing to disk except via `polyphonic_eval.io.jsonl` when explicitly invoked by the caller.

Dependency security is monitored via Dependabot (configured in `.github/dependabot.yml`) and `pip-audit` recommended for downstream users.
