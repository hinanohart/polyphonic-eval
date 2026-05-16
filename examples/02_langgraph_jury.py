"""02 — LangGraph jury with polyphonic_reducer.

Run with::

    pip install 'polyphonic-eval[langgraph,embed]'
    python examples/02_langgraph_jury.py

This example uses LangGraph's reducer mechanism so each judge node's verdict
survives into the final aggregation step.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from polyphonic_eval import JudgeVerdict
from polyphonic_eval.adapters.langgraph import finalize_polyphonic, polyphonic_reducer


class JuryState(TypedDict):
    item_id: str
    verdicts: Annotated[list[JudgeVerdict], polyphonic_reducer]


def judge(state: JuryState, *, judge_id: str, score: float, rationale: str) -> JuryState:
    return JuryState(
        item_id=state["item_id"],
        verdicts=[JudgeVerdict(judge_id=judge_id, score=score, rationale=rationale)],
    )


def main() -> None:
    state: JuryState = JuryState(item_id="lg-1", verdicts=[])

    # In a real graph these would be parallel nodes. We mimic the fan-in here.
    panel = [
        ("gpt-4o", 0.9, "Helpful and accurate."),
        ("claude", 0.85, "Useful answer."),
        ("gemini", 0.9, "Good explanation."),
        ("llama", 0.2, "Contains factual error."),
        ("qwen", 0.1, "Has safety concern."),
    ]
    for judge_id, score, rationale in panel:
        verdicts = polyphonic_reducer(
            state["verdicts"],
            JudgeVerdict(judge_id=judge_id, score=score, rationale=rationale),
        )
        state = JuryState(item_id=state["item_id"], verdicts=verdicts)

    result = finalize_polyphonic(state["verdicts"], item_id=state["item_id"])
    print(
        f"item={result.item_id} n={result.n_judges} "
        f"consensus={result.consensus.has_consensus} "
        f"irreducible={result.disagreement.is_irreducible} "
        f"spectrum={result.disagreement_spectrum:.3f}"
    )


if __name__ == "__main__":
    main()
