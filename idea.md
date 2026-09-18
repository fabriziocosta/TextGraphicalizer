The approach is to treat graph construction as a sequence of probabilistic classification decisions, then impose global graph constraints with an optimization solver. Laya is well suited to the local decision stage because it accepts arbitrary state plus typed questions and returns structured probabilities rather than generated text. The repository supports `choice`, `score`, and `noul` decision primitives, with `choice` returning probabilities over labels and `noul` returning a calibrated \(P(\mathrm{true})\). ([GitHub][1])

Given a paragraph and an ontology of concepts, the first stage would score whether each concept is present in the text. Each concept could be posed as a `noul` question such as “Is concept X expressed in this paragraph?”, producing a probability between 0 and 1. Laya’s README explicitly presents `noul` as a calibrated boolean probability and demonstrates this pattern for questions such as churn risk and phishing detection. ([GitHub][1])

The concepts with sufficiently high probability become candidate nodes. It is preferable to retain their probabilities rather than immediately turning them into hard binary decisions. Laya is specifically designed to return confidence values alongside its predictions, and the repository describes confidence thresholding as a mechanism for deciding when to act automatically or defer a decision. ([GitHub][1])

The next stage examines pairs of candidate nodes. For each pair, one option is to ask a binary `noul` question such as “Is there a semantic relation between concept A and concept B?” and retain the resulting edge-existence probability. This use is an extension of the repository’s documented boolean decision primitive; the repository does not provide a graph extraction example specifically. ([GitHub][1])

For pairs that appear related, a `choice` question can then classify the relation using the relation ontology. For example, the criteria might contain labels such as `causes`, `part_of`, `supports`, `contradicts`, or whatever relations the ontology defines. Laya’s `choice` primitive accepts a set of candidate labels with short natural-language descriptions and returns the selected label, probabilities for the alternatives, and a confidence value. ([GitHub][1])

An alternative is to combine edge detection and relation classification into a single `choice` operation:

```text
causes
part_of
supports
contradicts
...
no_relation
```

This is consistent with Laya’s documented use of `choice` for categorical selection over user-specified criteria. The potential advantage of this formulation is methodological rather than something claimed by the Laya authors: it avoids a separate binary gate whose error could eliminate an edge before its relation type is evaluated. ([GitHub][1])

The output of these stages is therefore a weighted candidate graph:

```text
Paragraph
   ↓
Concept questions
   ↓
Candidate nodes
P(node_i | text)
   ↓
Pairwise relation questions
   ↓
Candidate edges
P(relation_r(i,j) | text)
   ↓
Mixed integer optimization
   ↓
Final graph
```

The language model supplies local probabilistic evidence. A mixed integer optimization model can then choose a globally coherent subset of those nodes and edges. Constraints can enforce properties such as connectivity, maximum degree, sparsity, permitted relation combinations, ontology rules, or other application-specific graph conditions. This optimization layer would be your own addition; it is not part of the Laya repository.

One attractive feature is that Laya can evaluate multiple typed questions over the same state. The repository states that multiple questions can be processed together and gives examples of jointly evaluating department, urgency, churn risk, and phishing status. ([GitHub][1]) For the graph application, that suggests batching many concept-presence or relation questions around the same paragraph rather than invoking a separate generative model for each decision.

The resulting architecture separates two kinds of reasoning. Laya handles **local semantic evidence**: whether a concept is present and how likely particular semantic relations are. The optimization model handles **global graph structure**. This preserves probabilistic uncertainty from the classifier while allowing exact structural constraints to determine the final graph. The repository’s emphasis on calibrated confidence scores makes this especially relevant, although the quality of calibration for a new ontology and domain would need to be empirically validated on your own data. The README reports low calibration error on several of its benchmark tasks and some zero-shot evaluation, but these are project-reported results rather than evidence specifically for ontology-based graph extraction. ([GitHub][1])

Repository: [NandhaKishorM/laya on GitHub](https://github.com/NandhaKishorM/laya?utm_source=chatgpt.com)

[1]: https://github.com/NandhaKishorM/laya "GitHub - NandhaKishorM/laya · GitHub"
