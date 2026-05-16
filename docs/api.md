# API reference

## Top-level

### ``aggregate``

```python
def aggregate(
    verdicts: Sequence[JudgeVerdict],
    *,
    item_id: str,
    embedder: Embedder | None = None,
    cluster_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    config: AggregatorConfig | None = None,
) -> PolyphonicResult: ...
```

Compute a typed ``PolyphonicResult`` from multi-judge verdicts.

### ``PolyphonicAggregator``

Stateful version that caches embedder/config across calls.

```python
agg = PolyphonicAggregator(embedder=..., config=AggregatorConfig(...))
for batch in batches:
    yield agg.aggregate(batch, item_id=batch.id)
```

### ``Embedder`` protocol

```python
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...
```

Anything with an ``embed(texts) -> (n, d)`` method is an embedder. The library
calls ``isinstance(obj, Embedder)`` at runtime when needed.

## Model types

All models are frozen Pydantic v2.

### ``JudgeVerdict``

| field | type | notes |
|------|------|-------|
| ``judge_id`` | ``str`` | required |
| ``score`` | ``float | None`` | numeric verdict |
| ``label`` | ``str | None`` | categorical verdict |
| ``rationale`` | ``str | None`` | free-text reason (used for clustering) |
| ``confidence`` | ``float | None`` | optional self-reported confidence |
| ``metadata`` | ``Mapping[str, str]`` | freeform extension |

### ``PolyphonicResult``

| field | type | notes |
|------|------|-------|
| ``item_id`` | ``str`` | unique id for the evaluation item |
| ``verdicts`` | ``tuple[JudgeVerdict, ...]`` | exact inputs preserved |
| ``consensus`` | ``ConsensusClaim`` | did the panel agree? |
| ``disagreement`` | ``IrreducibleDisagreement`` | structured disagreement |
| ``disagreement_spectrum`` | ``float`` | Simpson × semantic-distance index |
| ``n_judges`` | ``int`` | count |
| ``schema_version`` | ``Literal["1"]`` | wire-format version |

Methods:

```python
.to_scalar(policy="refuse")    # default raises TypeError
.to_scalar(policy="mean")      # arithmetic mean
.to_scalar(policy="majority")  # median (scores) or modal-label ratio
```

### ``ConsensusClaim``

```python
has_consensus: bool
consensus_score: float | None
consensus_label: str | None
agreeing_judges: tuple[str, ...]
agreement_strength: float  # in [0, 1]
```

### ``IrreducibleDisagreement``

```python
is_irreducible: bool
clusters: tuple[DisagreementCluster, ...]
minority_clusters: tuple[int, ...]      # cluster_ids with weight < cutoff
bootstrap_stability: float              # mean ARI in [0, 1]
explanation: str
```

### ``DisagreementCluster``

```python
cluster_id: int
members: tuple[str, ...]            # judge_ids in this cluster
centroid_rationale: str             # closest-to-centroid member's text
score_summary: ScoreSummary
weight: float                       # share of judges in this cluster
keywords: tuple[str, ...]
```

## Configuration

### ``AggregatorConfig``

| field | default | notes |
|------|---------|-------|
| ``min_cluster_size`` | 2 | HDBSCAN parameter |
| ``bootstrap_iters`` | 200 | resamples for is_irreducible |
| ``ari_threshold`` | 0.6 | irreducibility decision threshold |
| ``consensus_score_tolerance`` | 0.15 | max spread for score consensus |
| ``keyword_top_k`` | 5 | keywords per cluster |
| ``minority_weight_cutoff`` | 0.5 | cluster weight below = minority |
| ``rng_seed`` | 42 | bootstrap RNG seed |

## I/O

```python
from polyphonic_eval.io import to_json, from_json, write_jsonl, read_jsonl
```

JSON serialization round-trips PolyphonicResult exactly. JSONL helpers write
one result per line for batch eval pipelines.

## CLI

```bash
polyphonic-eval aggregate verdicts.json --item-id my-item -o result.json
```

Input is a JSON array of JudgeVerdict-shaped objects. Output is JSON.
