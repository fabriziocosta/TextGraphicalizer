# Laya concepts in TextGraphicalizer

This document explains the Laya ideas used by TextGraphicalizer. It is written
for readers who do not need to know the internals of a language model, but who
do want to understand what the numbers mean and how they become a graph.

## The short version

Laya is a **decision model**, not a text-generation model. We give it a piece
of state—here, one paragraph—and a set of typed questions. It returns
structured answers such as probabilities, class labels, scores, and confidence
values.

TextGraphicalizer uses those answers in two stages by default:

1. Laya supplies local semantic evidence: which ontology concepts seem to be
   present, and which relations seem to hold between pairs of concepts.
2. A mixed-integer optimizer chooses a globally consistent subset of that
   evidence and returns a directed NetworkX graph.

After graph selection, the grounding pass uses a Sentence Transformers
cross-encoder to choose a textual span for each retained node and edge. It
constructs a concept query from the ontology label and description, generates
every contiguous one-, two-, and three-word span, marks each span in the
original paragraph, and scores the concept/context pair. The highest-scoring
span is attached to the graph item. This is a ranking score, not a calibrated
probability, and it does not use another optimization problem.

Set `use_milp=False` to skip the second-stage optimizer and retain nodes and
edges by applying `node_threshold` and `edge_threshold` directly. In that mode,
`connected` and `max_node_degree` are ignored.

In symbols, the overall idea is:

$$
Text → Laya questions → weighted candidates
→ constraints + optimization → final graph
→ selected-item span grounding
$$

Laya does not know that the output should be a graph. The graph structure is
provided by this project.

## Span grounding

The default grounding model is the STS cross-encoder
`cross-encoder/stsb-distilroberta-base`. For an ontology concept such as
`State`, the first input is:

```text
Concept: State
Description: The way something is with respect to its main attributes.
```

The second input keeps the complete paragraph and marks one candidate span:

```text
Sentence: The severe <<drought>> caused widespread crop failure.
Candidate expression: drought
```

The backend scores all concept/span pairs in batches. Candidate spans retain
their original word positions, with `start_word` inclusive and `end_word`
exclusive. For example, `crop failure` can be stored as `start_word=5` and
`end_word=7`. Stopwords are not removed from candidate generation because the
surrounding sentence is part of the evidence.

An ontology can optionally provide single-word `grounding_terms` for a concept
or relation. These are lexical anchors, not replacements for model scoring:
when a candidate exactly matches an anchor (including a simple inflection), it
is preferred over a generic longer phrase that the STS model may otherwise
rank highly because it contains more of the unchanged paragraph.

Selected graph nodes and edges expose:

```python
{
    "span": "crop failure",
    "span_start": 5,
    "span_end": 7,
    "span_score": 0.83,
    "grounding_candidates": [
        {"span": "crop failure", "score": 0.83},
        {"span": "failure", "score": 0.76},
    ],
    "grounding_score_distribution": [...],
}
```

`grounding_candidates` keeps the top five alternatives for inspection, while
`grounding_score_distribution` retains the complete ranked distribution. When
the winning span contains one word, the legacy `word`, `word_index`, and
`word_score` attributes are populated as well. The graph metadata records the
grounding method, model id, candidate spans, and diagnostic shortlist size.

## Laya's basic vocabulary

### State

The **state** is the information Laya reads. It can be text, a dictionary, an
email, a ticket, or another JSON-like document. In this project the state is
the paragraph passed to `transform()`:

```python
graph = extractor.transform(
    "An infection caused the patient to develop a fever."
)
```

We can call the paragraph $T$. Laya evaluates questions using $T$; it does
not need a separate prompt for each question.

### Question

A **question** is a small instruction describing the decision we want to make.
It has an identifier, a `type`, and instructions. A question can also provide
the possible criteria or labels.

For example, the node question generated for an ontology concept is roughly:

```python
{
    "type": "noul",
    "instructions": (
        'Is the concept "Fever" expressed in this paragraph? '
        "Concept description: An elevated body temperature."
    ),
}
```

The identifier, such as `node_1`, is only a key used to match Laya's answer to
the concept that generated the question.

### One prediction with many questions

Laya can evaluate many typed questions against the same state in one call:

```python
result = agent.predict(state, questions)
answers = result["answers"]
```

The returned mapping has one answer per question. This is useful here because
an ontology may contain many concepts and many possible pairs of concepts. The
questions are different, but the paragraph is the same.

The phrase **non-autoregressive** means that Laya makes these structured
decisions directly. It is not writing an answer token by token and then asking
another program to parse the text.

## The three decision primitives

Laya exposes three main question types: `noul`, `choice`, and `score`.

### `noul`: a calibrated yes/no probability

Use `noul` when the decision is naturally a proposition that can be true or
false. In this project, each ontology concept is tested this way:

> Is concept $i$ expressed in paragraph $T$?

Laya returns a value called `noul`, which TextGraphicalizer interprets as

$$
pᵢ = P(Nᵢ = 1 | T)
$$

where $N_i=1$ means that concept $i$ is present or expressed.

The value is between 0 and 1:

- $p_i=0.90$ means strong evidence that the concept is present.
- $p_i=0.20$ means weak evidence that it is present.
- It is not a count of how many times a word occurs.

The project keeps a concept when

$$
pᵢ ≥ τᴺ
$$

where $\tau_N$ is `node_threshold`. The default is $0.5$. A threshold is
an operating choice, not a law of language: increasing it usually produces
fewer, more conservative nodes.

### `choice`: one category from several alternatives

Use `choice` when the answer should be one of a known set of alternatives. For
each pair of candidate concepts, TextGraphicalizer asks Laya to choose between
the ontology's permitted relation types and a special `no_relation` option.

For a source concept $i$ and target concept $j$, Laya returns a probability
for each option. If the options are $R_{ij}$, then ideally:

$$
Σᵣ∈Rᵢⱼ P(r | i, j, T) = 1
$$

The most likely relation label is

$$
r*ᵢⱼ = arg max { P(r | i, j, T) : r ∈ Rᵢⱼ and r ≠ no_relation }
$$

This gives the label placed on the graph edge.

TextGraphicalizer keeps the `no_relation` complement as a diagnostic existence
score, but uses the winning relation probability as the edge-selection score:

$$
eᵢⱼ = 1 − P(no_relation | i, j, T)

qᵢⱼ = maxᵣ P(r | i, j, T), for r ≠ no_relation
$$

Thus, in the current implementation:

- the winning relation probability chooses the edge label;
- $e_{ij}$ measures how much Laya believes *some* relation exists;
- $q_{ij}$ is the score compared with `edge_threshold` by the optimizer.

For example, suppose Laya returns:

```text
causes       0.72
part_of      0.08
no_relation  0.20
```

Then the edge label is `causes`, and its edge probability is

$$
eᵢⱼ = 1 − 0.20 = 0.80, \qquad qᵢⱼ = 0.72
$$

The graph stores `probability=qᵢⱼ`, `relation_probability=qᵢⱼ`, and
`existence_probability=eᵢⱼ`.

### `score`: a position on an ordered scale

Use `score` when answers have an order, such as `low`, `medium`, and `high`, or
levels 0 through 2. Unlike `choice`, the distance between levels matters.

If Laya assigns probabilities $p_0, p_1, \ldots, p_K$ to ordered levels
$0,1,\ldots,K$, the expected score is commonly written as

$$
E[S | T] = Σₖ₌₀ᴷ k · pₖ
$$

For example, with probabilities $0.1, 0.3, 0.6$ over levels 0, 1, and 2:

$$
E[S | T] = 0(0.1) + 1(0.3) + 2(0.6) = 1.5
$$

TextGraphicalizer currently uses `noul` for concept presence and `choice` for
relations. `score` is part of Laya's general vocabulary and is useful if a
future graph extension needs graded properties such as severity or importance.

## Probability and confidence are different

It is useful to separate **probability** from **confidence**.

For a `noul` question, the probability is about the proposition itself:

$$
P(concept is expressed | T)
$$

For a `choice` question, the probabilities describe the alternatives. The
confidence value is Laya's separate estimate of how reliable the selected
decision is. It should not automatically be treated as another class
probability or substituted for `noul`/relation probability.

In a calibrated system, predictions with probability $p$ should be correct
about $p$ of the time over a sufficiently large group of similar cases. For
example, among 100 independent cases assigned probability $0.8$, roughly 80
should be true if the probabilities are well calibrated:

$$
(1 / |B|) · Σₙ∈B yₙ ≈ (1 / |B|) · Σₙ∈B p̂ₙ
$$

where B is a bucket of predictions, yₙ ∈ {0, 1} is the observed outcome, and
p̂ₙ is the predicted probability.

Calibration is empirical. A number such as $0.8$ is not a guarantee, and
calibration on Laya's training tasks does not prove calibration for a new
ontology or domain. The graph stores both `probability` and, when Laya
provides it, `confidence` so downstream code can make that distinction.

## How Laya becomes a graph here

Let the ontology contain concepts $1,\ldots,n$, each with an id, a readable
label, and a description. It also contains directed relation definitions and
optional source/target restrictions.

### 1. Concept detection

TextGraphicalizer asks one `noul` question for every ontology concept. The
result is a set of candidate nodes:

$$
C = { i : pᵢ ≥ τᴺ }
$$

The candidates still carry their original probabilities. They are evidence,
not yet a final commitment to include every node.

### 2. Pairwise relation classification

For every ordered pair i, j ∈ C, i ≠ j, the ontology is checked
first. A pair is only queried when at least one relation is allowed for that
source and target type.

The resulting candidate edge has:

$$
e = (i, j, r*ᵢⱼ, qᵢⱼ)
$$

The order matters: $i\to j$ is different from $j\to i$. This is why the
output is a `networkx.DiGraph`, not an undirected graph.

### 3. Global selection with binary variables (`use_milp=True`)

The optimizer introduces a binary variable $x_i$ for each candidate node and
a binary variable $y_{ij}$ for each candidate edge:

$$
xᵢ, yᵢⱼ ∈ {0, 1}
$$

The basic endpoint constraints are:

$$
yᵢⱼ ≤ xᵢ,   yᵢⱼ ≤ xⱼ
$$

An edge therefore cannot be selected unless both of its endpoint nodes are
selected.

The optimizer rewards evidence relative to the configured thresholds. Define
the logit function

$$
logit(p) = ln( p / (1 − p) )
$$

The score it maximizes is equivalent to:

$$
maximize  Σᵢ [logit(pᵢ) − logit(τᴺ)] xᵢ
        + Σᵢⱼ [logit(qᵢⱼ) − logit(τᴱ)] yᵢⱼ
$$

where $\tau_E$ is `edge_threshold`.

This explains the meaning of a threshold-relative score. A probability above
the threshold has a positive contribution; a probability below it has a
negative contribution. The solver can still choose a lower-scoring edge when
the `connected=True` constraint makes that edge necessary to connect the graph.

If `max_node_degree=D` is set, every selected node obeys:

$$
Σ incident edges e at i: yₑ ≤ D · xᵢ
$$

This limits the number of incoming plus outgoing edges touching a node.

If `connected=False`, the result may contain several disconnected components.
If `connected=True`, the optimizer chooses the highest-probability candidate
node as a root and adds flow constraints so every selected node is reachable
from that root when edge directions are treated as usable connections.

### 4. Direct threshold selection (`use_milp=False`)

In direct threshold mode, a node is retained exactly when:

$$
p_i ≥ τ_N
$$

An edge is retained exactly when:

$$
q_{ij} ≥ τ_E
$$

and both endpoint nodes are retained. No connectivity or degree constraints
are applied in this mode.

## Worked example

Suppose the paragraph is:

> An infection caused the patient to develop a fever.

The concept questions might produce:

```text
infection  p = 0.91
patient    p = 0.78
fever      p = 0.88
```

With $\tau_N=0.5$, all three are candidates. For the ordered pair
`infection -> fever`, the relation question might return:

```text
causes       0.74
part_of      0.04
no_relation  0.22
```

The selected label is `causes`, while the edge selection score is

$$
q = 0.74
$$

With $\tau_E=0.5$, both the node evidence and this edge evidence are above
their thresholds, so selecting them improves the optimizer's objective. The
final graph can therefore contain:

```text
infection ── causes ──▶ fever
```

The actual output may contain additional nodes or edges if their evidence and
the global constraints make them worthwhile.

## What Laya does not do

Laya does not replace the ontology, prove that a statement is factually true,
or decide whether a graph is globally coherent. It answers the questions it is
given. The surrounding project is responsible for:

- defining concepts and their descriptions;
- defining which relation types are legal;
- choosing thresholds;
- enforcing connectivity and degree limits;
- interpreting the final graph.

This separation is intentional: Laya is good at producing many small pieces
of semantic evidence, while the optimizer is good at applying exact structural
rules.

## Model loading and reproducibility

The default model is `convaiinnovations/laya`. TextGraphicalizer pins a model
revision when using the default model, and records the effective model revision
in `graph.graph["model_revision"]`. The graph metadata also records the model
id, thresholds, connectivity setting, degree limit, solver, and whether the
input was longer than the model's configured token limit.

For more details about Laya's public API and decision primitives, see the
[upstream Laya repository](https://github.com/NandhaKishorM/laya).
