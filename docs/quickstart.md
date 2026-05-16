# Quickstart

## Install

```bash
pip install polyphonic-eval[embed]
```

The ``[embed]`` extra pulls ``sentence-transformers`` for the default embedder.
Skip it if you supply a custom ``Embedder``.

## First aggregation

```python
from polyphonic_eval import aggregate, JudgeVerdict

verdicts = [
    JudgeVerdict(judge_id="gpt-4o", score=0.9, rationale="Helpful and concise."),
    JudgeVerdict(judge_id="claude",  score=0.85, rationale="Useful answer."),
    JudgeVerdict(judge_id="gemini",  score=0.9, rationale="Good explanation."),
    JudgeVerdict(judge_id="llama",   score=0.2, rationale="Factually wrong about X."),
    JudgeVerdict(judge_id="qwen",    score=0.1, rationale="Has a safety concern."),
]

result = aggregate(verdicts, item_id="example-1")
```

## Inspect the result

```python
print(result.consensus.has_consensus)         # False
print(result.disagreement.is_irreducible)     # True (likely)
print(len(result.disagreement.clusters))      # 2 or 3
print(result.disagreement_spectrum)           # ~0.7

for c in result.disagreement.clusters:
    print(c.cluster_id, c.members, c.weight, c.keywords)
```

## Opt in to a scalar

```python
result.to_scalar()                  # TypeError — refusing
result.to_scalar(policy="mean")     # 0.59
result.to_scalar(policy="majority") # median = 0.85
```

## Reproducibility

The clustering result depends on the embedding model. For reproducible eval
pipelines, **pin an embedder**:

```python
from sentence_transformers import SentenceTransformer
from polyphonic_eval import aggregate

class MyEmbedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def embed(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True)

result = aggregate(verdicts, item_id="repro", embedder=MyEmbedder())
```
