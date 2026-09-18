# TextGraphicalizer

TextGraphicalizer converts one paragraph into an ontology-constrained
`networkx.DiGraph` using the [Laya](https://github.com/NandhaKishorM/laya)
decision model and a SciPy mixed-integer optimizer.

## Installation

```bash
python -m pip install -e .
```

The first call to `fit()` downloads `convaiinnovations/laya` into the standard
Hugging Face cache. The model weights are not stored in this repository. To
run offline, pass a previously downloaded snapshot with `model_path`.

## Ontology

```yaml
version: 1
concepts:
  - id: infection
    label: Infection
    description: A disease-causing invasion by pathogens.
  - id: fever
    label: Fever
    description: An elevated body temperature.
relations:
  - id: causes
    label: causes
    description: The source concept produces or leads to the target concept.
```

Relations may optionally specify `source_concepts` and `target_concepts`.

## Usage

```python
from textgraphicalizer import TextGraphicalizer

graph = (
    TextGraphicalizer("ontology.yaml", connected=False)
    .fit()
    .transform("An infection caused the patient to develop a fever.")
)

print(graph.nodes(data=True))
print(graph.edges(data=True))
```

The estimator returns a directed graph. Nodes and edges contain `label`,
`probability`, and, when available, Laya's distinct `confidence` value.

## Model smoke test

The optional integration test downloads and loads the model:

```bash
pytest -m model
```

The regular test suite uses a deterministic fake backend and does not download
model weights.
