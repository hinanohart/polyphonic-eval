"""01 — Quickstart. Five judges, one item, typed result.

Install the embedder extra (default sentence-transformers wrapper) first::

    pip install 'polyphonic-eval[embed]'
    python examples/01_quickstart.py
"""

from __future__ import annotations

from polyphonic_eval import JudgeVerdict, aggregate


def main() -> None:
    verdicts = [
        JudgeVerdict(judge_id="gpt-4o", score=0.9, rationale="Helpful and concise."),
        JudgeVerdict(judge_id="claude", score=0.85, rationale="Useful answer overall."),
        JudgeVerdict(judge_id="gemini", score=0.9, rationale="Good explanation."),
        JudgeVerdict(judge_id="llama", score=0.2, rationale="Factually wrong about X."),
        JudgeVerdict(judge_id="qwen", score=0.1, rationale="Has a safety concern."),
    ]
    result = aggregate(verdicts, item_id="quickstart-1")

    print(f"item_id              = {result.item_id}")
    print(f"n_judges             = {result.n_judges}")
    print(f"has_consensus        = {result.consensus.has_consensus}")
    print(f"is_irreducible       = {result.disagreement.is_irreducible}")
    print(f"n_clusters           = {len(result.disagreement.clusters)}")
    print(f"disagreement_spectrum= {result.disagreement_spectrum:.3f}")
    print(f"bootstrap_stability  = {result.disagreement.bootstrap_stability:.3f}")
    print()
    print("clusters:")
    for c in result.disagreement.clusters:
        print(f"  [{c.cluster_id}] weight={c.weight:.2f} members={c.members} keywords={c.keywords}")

    print()
    print(f"to_scalar(mean)     = {result.to_scalar(policy='mean'):.3f}")
    print(f"to_scalar(majority) = {result.to_scalar(policy='majority'):.3f}")
    try:
        result.to_scalar()
    except TypeError as e:
        print(f"to_scalar()         = TypeError: {e}")


if __name__ == "__main__":
    main()
