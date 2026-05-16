# LangGraph adapter

``polyphonic-eval`` exposes a state-reducer for [LangGraph](https://github.com/langchain-ai/langgraph)
that preserves every judge's verdict instead of overwriting.

## Install

```bash
pip install polyphonic-eval[langgraph,embed]
```

## State schema

```python
from typing import Annotated, TypedDict
from polyphonic_eval import JudgeVerdict
from polyphonic_eval.adapters.langgraph import polyphonic_reducer

class JuryState(TypedDict):
    item_id: str
    verdicts: Annotated[list[JudgeVerdict], polyphonic_reducer]
```

Each judge node returns a ``JudgeVerdict``; the reducer appends to the list
rather than overwriting. By the time the terminal node runs, every judge's
vote has survived.

## Aggregating at the terminal node

```python
from polyphonic_eval.adapters.langgraph import finalize_polyphonic

def aggregate_node(state: JuryState) -> dict:
    result = finalize_polyphonic(state["verdicts"], item_id=state["item_id"])
    return {"result": result}
```

``finalize_polyphonic`` is just a thin wrapper around ``aggregate``.

## Pattern: parallel judges

Use LangGraph's ``Send`` API to fan out to N judges. Each judge node returns
``{"verdicts": [its_verdict]}`` and the reducer takes care of merging.

```python
from langgraph.types import Send

def fan_out(state):
    return [Send("judge", {"item_id": state["item_id"], "judge_id": jid})
            for jid in ["gpt-4o", "claude", "gemini", "llama", "qwen"]]
```

## Why this matters

The default LangGraph state behavior overwrites scalar fields. Annotating
``verdicts`` with ``polyphonic_reducer`` is the LangGraph-side commitment to
**not collapse** disagreement at the graph layer either.
