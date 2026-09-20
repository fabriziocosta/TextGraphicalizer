# TextGraphicalizer

TextGraphicalizer converts one document into an ontology-constrained
`networkx.DiGraph` using the [Laya](https://github.com/NandhaKishorM/laya)
decision model and a SciPy mixed-integer optimizer.

## Installation

```bash
# Run this from the Python environment you want to use for the project.
python -m pip install -e .
```

For the notebook, install the optional Jupyter dependencies and register that
same environment as a project-specific kernel:

```bash
python -m pip install -e ".[notebook]"
python -m ipykernel install --user --name textgraphicalizer --display-name "TextGraphicalizer"
```

If you need a fresh environment, create it in the repository with
`python -m venv .venv` and activate it. On macOS/Linux use
`source .venv/bin/activate`; on Windows
PowerShell use `.venv\Scripts\Activate.ps1`. The commands above always use
the interpreter selected by `python`.

Constructing `TextGraphicalizer()` automatically initializes Laya and, by
default, the span-grounding cross-encoder;
this may download both checkpoints into the standard Hugging Face cache.
The ontology and `stopwords_path` are optional at construction and can be set
before the first transformation. `.load_model()` remains available and
idempotent. The model weights are not stored in this repository. To run
offline, pass a previously downloaded Laya snapshot with `model_path` and
ensure the configured span-grounding checkpoint is cached.

Set `use_llm=True` to replace cross-encoder grounding with structured LLM
calls over the selected graph. The default provider remains `openai`, using
`gpt-4.1-mini` and `OPENAI_API_KEY`. To use a locally hosted Ollama model,
select `llm_provider="ollama"`; its default model is `gemma4:12b-mlx` and its
default endpoint is `http://localhost:11434`:

```python
extractor = TextGraphicalizer(
    ontology="ontology.yaml",
    use_llm=True,
    llm_provider="ollama",
)
```

Set `llm_model` to override either provider's default, or set
`ollama_base_url` for another Ollama endpoint. Local requests allow up to ten
minutes by default because a 12B model may need time to load; set
`ollama_timeout` to change that limit. Thinking output is disabled so the
structured graph response is returned directly. The backend asks for a
concise, context-sensitive paraphrase for each selected concept and relation.
These LLM paraphrases are allowed to express implicit concepts and are stored
as `paraphrase`; unlike cross-encoder spans, they do not have document offsets.

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
The span grounder uses the ontology label and description directly, so no
handwritten grounding vocabulary is required. Optional single-word
`grounding_terms` act as lexical anchors when a generic longer span receives a
higher raw STS score.

## General WordNet ontology

The repository includes [`ontology.yaml`](ontology.yaml), a 23-concept
starter ontology aligned with Princeton WordNet. It deliberately stays at
general categories such as `entity`, `person`, `location`, `artifact`,
`event`, `process`, and `state`; example-domain entities are left to a
separate application ontology.

This keeps the reference vocabulary reusable and limits the number of model
questions. WordNet represents distinct senses as synsets, so domain-specific
work can add narrower synsets when that precision is actually required rather
than baking example entities into the default ontology. See the [Princeton
WordNet overview](https://wordnet.princeton.edu/) and its documentation of
semantic relations.

## Fairy-tale ontology

[`fairy_tale_ontology.yaml`](fairy_tale_ontology.yaml) is a separate,
Propp-inspired application ontology for fairy tales. The reference is Vladimir
Propp—not Fodor—whose *Morphology of the Folktale* describes recurring
character functions (hero, adversary, donor, helper, dispatcher, false hero,
and sought person/prize) and recurring plot functions. The ontology groups
those functions into reusable events such as departure, prohibition,
misfortune/lack, quest, trial, gift, struggle, pursuit, rescue, recognition,
punishment, and reward/union rather than copying all 31 functions as separate
concepts. See [Propp's *Morphology of the Folktale*](https://www.jstor.org/stable/10.7560/783911.19)
and the overview of its seven character classes and 31 functions in [Bikakis
et al.](https://journals.sagepub.com/doi/full/10.3233/SW-200417).

It also adds reusable story-world concepts for people, animals, magical beings,
groups/families, places, dwellings, ordinary objects, and magical objects.
Use it explicitly when graphicalizing the Aesop corpus:

```python
extractor.ontology = "fairy_tale_ontology.yaml"
graph = extractor.transform(stories[0])
```

## Aesop corpus

`load_aesop_fables()` downloads the UTF-8 text of [Project Gutenberg eBook
#11339](https://www.gutenberg.org/ebooks/11339), removes its book front matter,
illustration list, and license text, and returns one cleaned string per fable.
The parsed stories are cached as JSON under
`~/.cache/textgraphicalizer/aesop-fables-11339.json` by default; pass
`cache_path=...` to choose another location or `refresh=True` to rebuild it.

```python
from textgraphicalizer import TextGraphicalizer, load_aesop_fables

stories = load_aesop_fables()
extractor = TextGraphicalizer()
extractor.ontology = "ontology.yaml"
graph = extractor.transform(stories[0])
extractor.display(graph, document=stories[0], max_char=100)
```

## Usage

```python
from textgraphicalizer import TextGraphicalizer

extractor = TextGraphicalizer()
extractor.ontology = "ontology.yaml"
extractor.stopwords_path = "stopwords.yaml"
graph = extractor.transform("An infection caused the patient to develop a fever.")

print(graph.nodes(data=True))
print(graph.edges(data=True))
```

For LLM-based grounding:

```python
extractor = TextGraphicalizer(
    ontology="ontology.yaml",
    use_llm=True,
    llm_provider="openai",
    llm_model="gpt-4.1-mini",
)
graph = extractor.transform("An infection caused the patient to develop a fever.")
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
budget—not against the document token count alone.

For each selected node and edge, TextGraphicalizer generates every contiguous
one-, two-, and three-word span without removing stopwords. It sends the
ontology query and a marked sentence context to the same Sentence Transformers
cross-encoder, then attaches the highest-scoring span as `span`, `span_start`,
`span_end`, and `span_score`. The top five candidates are retained in
`grounding_candidates` for diagnostics. Single-word winners also populate the
backward-compatible `word`, `word_index`, and `word_score` fields. The STS
score is a ranking compatibility score, not a calibrated probability; set
`grounding_model_id=...` to try another cross-encoder checkpoint. Relation
spans are attached only when a relation term or relation-label phrase occurs
in the document; otherwise the edge retains its ontology label without a
misleading span.

Span selection also favors concise evidence and discounts stopword-heavy
context fragments, so a repeated phrase such as `Fox saw some` does not become
the grounding span for every concept in a story.
Node spans are then resolved with a linear assignment over distinct surface
mentions; weak competing assignments are left ungrounded rather than forced.
When `use_llm=True`, this span-assignment process is replaced by the LLM's
context-sensitive paraphrases, so multiple concepts may be described even
when the document does not contain distinct matching surface spans.

After grounding, empty nodes are collapsed into adjacent grounded nodes using
the strongest grounded node as the stable target; their incident edges are
redirected and duplicate redirected edges are merged. Empty components with
no grounded anchor are left unchanged.
In LLM mode, a second structured call checks adjacent grounded nodes for
story-level coreference (for example, `goose`, `animal`, and `entity`).
Confirmed duplicates are collapsed into the more specific concept; their
edges are redirected and merged in the same way.
After all node collapsing is complete, a final LLM review re-evaluates every
remaining edge against the ontology. It may relabel an edge, reverse its
direction so an agent/causer precedes its patient/recipient, or remove it when
no ontology relation is supported.

## Rendering graphs

`TextGraphicalizer.display()` renders any resulting graph and returns its
Matplotlib `(figure, axes)` pair. Rendering options can be tuned without
duplicating visualization code:

When reciprocal edges have the same relation label, the renderer displays
them as one undirected edge. Node and edge annotations include both the
ontology/relation label and the associated span or LLM paraphrase.
Node labels are rendered lowercase in normal-weight monospace; grounded spans
use normal-weight serif text.

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
    max_char=100,
    show=False,                  # useful when composing several plots
)
```

Pass `text="..."` instead of `graph` to transform and render in one call.

For an interactive notebook view, `display_d3()` returns a force-directed D3
HTML display with zooming and draggable nodes. It loads D3.js from its CDN:

```python
from IPython.display import display

display(extractor.display_d3(
    graph,
    title=title,
    document=document,
    max_char=100,
))
```

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
