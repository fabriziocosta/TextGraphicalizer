# TextGraphicalizer

TextGraphicalizer converts one paragraph into an ontology-constrained
`networkx.DiGraph` using the [Laya](https://github.com/NandhaKishorM/laya)
decision model and a SciPy mixed-integer optimizer.

## Installation

```bash
source ~/.venvs/py312/bin/activate
python -m pip install -e .
```

For the notebook, install the optional Jupyter dependencies and use the
existing `py312` kernel:

```bash
python -m pip install -e ".[notebook]"
python -m ipykernel install --user --name py312 --display-name "py312"
```

Constructing `TextGraphicalizer` initializes Laya and the BERT grounding model;
this may download both checkpoints into the standard Hugging Face cache.
`.load_model()` remains available and idempotent. The model weights are not
stored in this repository. To run offline, pass a previously downloaded Laya
snapshot with `model_path` and ensure the configured BERT checkpoint is cached.

By default, graph selection uses the MILP optimizer. Set `use_milp=False` to
select nodes with `node_threshold` and edges with `edge_threshold` directly;
in that mode, `connected` and `max_node_degree` are ignored.

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

extractor = TextGraphicalizer(
    ontology="ontology.yaml",
    stopwords_path="stopwords.yaml",
    grounding_model_id="bert-base-uncased",
    connected=False,
)
graph = extractor.fit_transform("An infection caused the patient to develop a fever.")

print(graph.nodes(data=True))
print(graph.edges(data=True))
```

The estimator returns a directed graph for one string, or a list of graphs
when passed a sequence of strings. Nodes contain `label`, `probability`, and,
when available, Laya's distinct `confidence` value. Edges retain `probability`
as the winning relation probability and also expose separate
`existence_probability` and `relation_probability` attributes.

With the default `use_milp=True`, all ontology concepts are scored before
optimization. `node_threshold` and `edge_threshold` are threshold-relative
MILP objective settings, not early filters; the optimizer can select a
below-threshold node when it improves the overall graph. With `use_milp=False`,
the thresholds are direct filters and both endpoints of a retained edge must
pass `node_threshold`.

Graph metadata includes `input_truncated`. This flag is computed against the
full Laya sequence for every node and relation question, including question
instructions, relation options, special tokens, and the configured 512-token
budget—not against the paragraph token count alone.

For candidate nodes and selected edges, TextGraphicalizer uses contextual BERT
embeddings. It encodes each non-stopword candidate in the complete paragraph,
encodes each ontology concept or relation description, and assigns the word
with the highest cosine similarity. Long paragraphs are encoded with
overlapping BERT windows. Stopwords are removed from this choice set using
[`stopwords.yaml`](stopwords.yaml); supply `stopwords_path=...` to use another
YAML file. The result is stored as `word`, `word_index`, and `word_score` on
the corresponding node or edge. Set `grounding_model_id=...` to use another
Hugging Face BERT checkpoint.
The checked-in default is derived from the [Snowball English stopword
list](https://snowballstem.org/algorithms/english/stop.txt).

## Rendering graphs

`TextGraphicalizer.display()` renders any resulting graph and returns its
Matplotlib `(figure, axes)` pair. Rendering options can be tuned without
duplicating visualization code:

When reciprocal edges have the same relation label, the renderer displays
them as one undirected edge. Node and edge annotations include both the
ontology/relation label and the associated sentence word.

The default layout is `kamada_kawai`.

```python
figure, axes = extractor.display(
    graph,
    layout="circular",          # spring, circular, shell, kamada_kawai, or callable
    node_size=1400,
    node_cmap="viridis",
    edge_cmap="plasma",
    color_by_probability=True,
    scale_node_size_by_probability=True,
    scale_edge_width_by_probability=True,
    show_probabilities=True,
    show=False,                  # useful when composing several plots
)
```

Pass `text="..."` instead of `graph` to transform and render in one call.

## Model smoke test

The optional integration test downloads and loads the model:

```bash
TEXTGRAPHICALIZER_RUN_MODEL_TESTS=1 pytest -m model
```

The regular test suite uses a deterministic fake backend and does not download
model weights.

## Development checks

Run the local quality checks with:

```bash
python -m pytest -m "not model"
ruff check src tests
mypy src
```

The CI workflow tests Python 3.10, 3.11, and 3.12, executes the notebook with
a fake backend, and verifies installation into a separate clean virtual
environment.
