# TextGraphicalizer

TextGraphicalizer converts one paragraph into an ontology-constrained
`networkx.DiGraph` using the [Laya](https://github.com/NandhaKishorM/laya)
decision model and a SciPy mixed-integer optimizer.

## Installation

```bash
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -e .
```

For the notebook, install the optional Jupyter dependencies and register its
Python 3.12 kernel:

```bash
.venv312/bin/python -m pip install -e ".[notebook]"
.venv312/bin/python -m ipykernel install --user \
  --name textgraphicalizer-py312 \
  --display-name "TextGraphicalizer (Python 3.12)"
```

The first call to `fit()` downloads the pinned `convaiinnovations/laya`
checkpoint into the standard Hugging Face cache. The model weights are not
stored in this repository. To run offline, pass a previously downloaded
snapshot with `model_path`.

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

## Initial WordNet ontology

The repository includes [`ontology.yaml`](ontology.yaml), an initial
high-level ontology aligned with Princeton WordNet. It uses broad WordNet
synsets such as `entity.n.01`, `physical_entity.n.01`, `person.n.01`, and
`event.n.01`, together with WordNet-inspired relations including `is_a`,
`part_of`, `member_of`, `made_of`, `causes`, and `entails`.

This is intentionally a coarse first pass: WordNet represents distinct senses
as synsets, so domain-specific work should later add narrower synsets rather
than treating a word string as one universal concept. See the [Princeton
WordNet overview](https://wordnet.princeton.edu/) and its documentation of
semantic relations.

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
TEXTGRAPHICALIZER_RUN_MODEL_TESTS=1 pytest -m model
```

The regular test suite uses a deterministic fake backend and does not download
model weights.
