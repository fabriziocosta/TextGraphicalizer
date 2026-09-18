"""Public scikit-learn-style TextGraphicalizer estimator."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Mapping

import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .laya_backend import LayaBackend
from .nli_backend import NliGroundingBackend
from .ontology import Ontology, load_ontology
from .optimizer import EdgeEvidence, NodeEvidence, select_graph, select_graph_by_threshold
from .stopwords import DEFAULT_STOPWORDS_PATH, load_stopwords

logger = logging.getLogger(__name__)


_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


def _confidence(answer: Mapping[str, Any]) -> float | None:
    value = answer.get("confidence")
    return float(value) if value is not None else None


class TextGraphicalizer(BaseEstimator, TransformerMixin):
    """Convert one paragraph into an ontology-constrained directed graph."""

    def __init__(
        self,
        ontology: str | Path | Mapping[str, Any] | Ontology,
        model_id: str = "convaiinnovations/laya",
        model_path: str | Path | None = None,
        model_revision: str | None = None,
        device: str = "auto",
        node_threshold: float = 0.5,
        edge_threshold: float = 0.5,
        use_milp: bool = True,
        connected: bool = False,
        max_node_degree: int | None = None,
        stopwords_path: str | Path | None = None,
        grounding_model_id: str = "cross-encoder/nli-distilroberta-base",
    ) -> None:
        self.ontology = ontology
        self.model_id = model_id
        self.model_path = model_path
        self.model_revision = model_revision
        self.device = device
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold
        self.use_milp = use_milp
        self.connected = connected
        self.max_node_degree = max_node_degree
        self.stopwords_path = stopwords_path
        self.grounding_model_id = grounding_model_id
        self.load_model()

    def _validate_parameters(self) -> None:
        if not isinstance(self.grounding_model_id, str) or not self.grounding_model_id:
            raise TypeError("grounding_model_id must be a non-empty string")
        for name, value in (
            ("node_threshold", self.node_threshold),
            ("edge_threshold", self.edge_threshold),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        if not isinstance(self.use_milp, bool):
            raise TypeError("use_milp must be a bool")
        if not isinstance(self.connected, bool):
            raise TypeError("connected must be a bool")
        if self.max_node_degree is not None:
            if not isinstance(self.max_node_degree, int) or isinstance(self.max_node_degree, bool):
                raise TypeError("max_node_degree must be an integer or None")
            if self.max_node_degree < 0:
                raise ValueError("max_node_degree must be non-negative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")

    def fit(self, X: Any = None, y: Any = None) -> "TextGraphicalizer":
        del X, y
        self._validate_parameters()
        if not hasattr(self, "ontology_"):
            self.ontology_ = load_ontology(self.ontology)
        if not hasattr(self, "stopwords_"):
            self.stopwords_ = load_stopwords(self.stopwords_path)
        self.n_features_in_ = 1
        return self

    def load_model(self) -> "TextGraphicalizer":
        """Load Laya and return this estimator.

        Loading is explicit because it may download a large checkpoint and
        initialize a device-specific runtime. This method can be called before
        or after ``fit()``. Repeated calls are idempotent for this estimator
        instance.
        """
        self._validate_parameters()
        if not hasattr(self, "ontology_"):
            self.ontology_ = load_ontology(self.ontology)
            self.n_features_in_ = 1
        if not hasattr(self, "stopwords_"):
            self.stopwords_ = load_stopwords(self.stopwords_path)
        if getattr(self, "_model_loaded_", False):
            return self
        self.backend_: LayaBackend = LayaBackend(
            model_id=self.model_id,
            model_path=self.model_path,
            model_revision=self.model_revision,
            device=self.device,
        ).load()
        self.grounding_backend_: NliGroundingBackend = NliGroundingBackend(
            model_id=self.grounding_model_id,
            device=self.device,
        ).load()
        self._model_loaded_ = True
        return self

    def _node_questions(self) -> dict[str, dict[str, Any]]:
        return {
            f"node_{index}": {
                "type": "noul",
                "instructions": (
                    f'Is the concept "{concept.label}" expressed in this paragraph? '
                    f"Concept description: {concept.description}"
                ),
            }
            for index, concept in enumerate(self.ontology_.concepts)
        }

    def _content_words(self, text: str) -> list[tuple[int, str]]:
        """Return original token positions and non-stopword single words."""
        words = _WORD_RE.findall(text)
        return [
            (index, word)
            for index, word in enumerate(words)
            if word.casefold() not in self.stopwords_
        ]

    def _node_words_by_nli(
        self,
        text: str,
        words: Sequence[tuple[int, str]],
        graph: nx.DiGraph,
        concepts_by_node: Mapping[str, str],
    ) -> dict[str, dict[str, Any]]:
        """Ground selected nodes by NLI entailment probability."""
        if not words or graph.number_of_nodes() == 0 or not concepts_by_node:
            return {}
        scores = self.grounding_backend_.score_words_contrastive(
            text, words, concepts_by_node
        )
        return self._best_nli_words(scores)

    def _edge_words_by_nli(
        self,
        text: str,
        words: Sequence[tuple[int, str]],
        graph: nx.DiGraph,
        relations_by_edge: Mapping[tuple[str, str], str],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Ground selected edges by NLI entailment probability."""
        if not words or graph.number_of_edges() == 0 or not relations_by_edge:
            return {}
        scores = self.grounding_backend_.score_words_contrastive(
            text, words, relations_by_edge
        )
        return self._best_nli_words(scores)

    @staticmethod
    def _best_nli_words(
        scores: Mapping[Any, Sequence[tuple[int, str, float]]],
    ) -> dict[Any, dict[str, Any]]:
        """Choose the highest-entailment candidate for each target."""
        best: dict[Any, dict[str, Any]] = {}
        for target, candidates in scores.items():
            if not candidates:
                continue
            token_index, word, score = max(candidates, key=lambda item: item[2])
            best[target] = {
                "word": word,
                "word_index": token_index,
                "word_score": score,
            }
        return best

    @staticmethod
    def _attach_node_words(
        graph: nx.DiGraph,
        node_words: Mapping[str, Mapping[str, Any]],
    ) -> None:
        for node in graph.nodes:
            if str(node) in node_words:
                graph.nodes[node].update(node_words[str(node)])

    @staticmethod
    def _attach_edge_words(
        graph: nx.DiGraph,
        edge_words: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> None:
        for source, target in graph.edges:
            if (str(source), str(target)) in edge_words:
                graph.edges[source, target].update(edge_words[(str(source), str(target))])

    @staticmethod
    def _display_edges(
        graph: nx.DiGraph,
    ) -> tuple[list[tuple[Any, Any, dict[str, Any]]], list[tuple[Any, Any, dict[str, Any]]]]:
        """Split edges into collapsed reciprocal and remaining directed edges."""
        undirected: list[tuple[Any, Any, dict[str, Any]]] = []
        directed: list[tuple[Any, Any, dict[str, Any]]] = []
        collapsed_pairs: set[frozenset[Any]] = set()

        for source, target, data in graph.edges(data=True):
            reverse_data = graph.get_edge_data(target, source)
            same_label = (
                source != target
                and reverse_data is not None
                and data.get("label") == reverse_data.get("label")
            )
            pair = frozenset((source, target))
            if not same_label or pair in collapsed_pairs:
                if not same_label:
                    directed.append((source, target, dict(data)))
                continue

            collapsed_pairs.add(pair)
            merged = dict(data)
            reverse_word = reverse_data.get("word")
            word = data.get("word")
            words = list(dict.fromkeys(
                str(value) for value in (word, reverse_word) if value is not None
            ))
            if words:
                merged["word"] = " / ".join(words)
            for field in ("probability", "existence_probability", "relation_probability"):
                values = [
                    float(value)
                    for value in (data.get(field), reverse_data.get(field))
                    if value is not None
                ]
                if values:
                    merged[field] = max(values)
            undirected.append((source, target, merged))

        return undirected, directed

    @staticmethod
    def _node_display_label(node: Any, data: Mapping[str, Any], show_probabilities: bool) -> str:
        parts = [str(data.get("label", node))]
        if data.get("word") is not None:
            parts.append(str(data["word"]))
        if show_probabilities:
            parts.append(f"{float(data.get('probability', 0.0)):.2f}")
        return "\n".join(parts)

    @staticmethod
    def _edge_display_label(data: Mapping[str, Any], show_probabilities: bool) -> str:
        parts = []
        if data.get("label") is not None:
            parts.append(str(data["label"]))
        if data.get("word") is not None:
            parts.append(str(data["word"]))
        if show_probabilities:
            parts.append(f"{float(data.get('probability', 0.0)):.2f}")
        return "\n".join(parts)

    def _transform_one(self, text: str) -> nx.DiGraph:
        """Transform one paragraph after readiness checks have completed."""
        node_questions = self._node_questions()
        logger.info("Scoring ontology concepts question_count=%d", len(node_questions))
        node_result = self.backend_.predict(text, node_questions)
        node_answers = node_result.get("answers", {})
        node_evidence: list[NodeEvidence] = []
        for index, concept in enumerate(self.ontology_.concepts):
            answer = node_answers.get(f"node_{index}")
            if answer is None:
                raise ValueError(f"Laya did not return an answer for node_{index}")
            probability = float(answer.get("noul"))
            # Keep every scored concept for MILP selection. In threshold mode,
            # below-threshold concepts are filtered before relation questions.
            node_evidence.append(
                NodeEvidence(
                    concept_id=concept.id,
                    label=concept.label,
                    probability=probability,
                    confidence=_confidence(answer),
                )
            )

        relation_nodes = (
            node_evidence
            if self.use_milp
            else [node for node in node_evidence if node.probability >= self.node_threshold]
        )
        content_words = self._content_words(text)
        edge_questions: dict[str, dict[str, Any]] = {}
        pair_map: dict[str, tuple[str, str]] = {}
        for source in relation_nodes:
            for target in relation_nodes:
                if source.concept_id == target.concept_id:
                    continue
                relations = self.ontology_.valid_relations(source.concept_id, target.concept_id)
                if not relations:
                    continue
                question_id = f"edge_{len(pair_map)}"
                pair_map[question_id] = (source.concept_id, target.concept_id)
                criteria = {
                    relation.id: f"{relation.label}: {relation.description}"
                    for relation in relations
                }
                criteria["no_relation"] = "No semantic relation is expressed between the concepts."
                edge_questions[question_id] = {
                    "type": "choice",
                    "instructions": (
                        f'Which relation, if any, is expressed from "{source.label}" '
                        f'to "{target.label}" in this paragraph? Choose no_relation '
                        "unless the relation is explicitly stated."
                    ),
                    "criteria": criteria,
                }

        logger.info(
            "Scoring ontology relations question_count=%d candidate_node_count=%d",
            len(edge_questions),
            len(relation_nodes),
        )
        edge_result = self.backend_.predict(text, edge_questions) if edge_questions else {"answers": {}}
        edge_answers = edge_result.get("answers", {})
        relation_by_id = self.ontology_.relation_by_id
        candidate_edges: list[EdgeEvidence] = []
        for question_id, (source_id, target_id) in pair_map.items():
            answer = edge_answers.get(question_id)
            if answer is None:
                raise ValueError(f"Laya did not return an answer for {question_id}")
            probabilities = {
                str(key): float(value)
                for key, value in (answer.get("probabilities") or {}).items()
            }
            no_relation_probability = probabilities.get("no_relation")
            if no_relation_probability is None:
                raise ValueError(f"Laya response for {question_id} lacks no_relation probability")
            relation_probabilities = {
                key: value for key, value in probabilities.items() if key != "no_relation"
            }
            if not relation_probabilities:
                continue
            relation_id = max(
                relation_probabilities,
                key=lambda relation: relation_probabilities[relation],
            )
            relation_probability = relation_probabilities[relation_id]
            candidate_edges.append(
                EdgeEvidence(
                    source=source_id,
                    target=target_id,
                    label=relation_by_id[relation_id].label,
                    probability=relation_probability,
                    confidence=_confidence(answer),
                    relation_probability=relation_probability,
                )
            )

        if self.use_milp:
            graph = select_graph(
                node_evidence,
                candidate_edges,
                node_threshold=self.node_threshold,
                edge_threshold=self.edge_threshold,
                connected=self.connected,
                max_node_degree=self.max_node_degree,
            )
        else:
            graph = select_graph_by_threshold(
                node_evidence,
                candidate_edges,
                node_threshold=self.node_threshold,
                edge_threshold=self.edge_threshold,
            )

        concept_by_id = self.ontology_.concept_by_id
        node_descriptions = {
            str(node): (
                f'The paragraph expresses the concept '
                f'"{graph.nodes[node]["label"]}". '
                f"{concept_by_id[node].description}"
            )
            for node in graph.nodes
            if node in concept_by_id
        }
        node_words = self._node_words_by_nli(
            text,
            content_words,
            graph,
            node_descriptions,
        )
        self._attach_node_words(graph, node_words)

        relation_by_label = {relation.label: relation for relation in relation_by_id.values()}
        edge_descriptions = {
            (str(source), str(target)): (
                f'The paragraph explicitly expresses the relation '
                f'"{data["label"]}" from "{graph.nodes[source]["label"]}" '
                f'to "{graph.nodes[target]["label"]}". '
                f"{relation_by_label[data['label']].description}"
            )
            for source, target, data in graph.edges(data=True)
            if data.get("label") in relation_by_label
        }
        edge_words = self._edge_words_by_nli(
            text,
            content_words,
            graph,
            edge_descriptions,
        )
        self._attach_edge_words(graph, edge_words)
        graph.graph.update(
            {
                "ontology_version": self.ontology_.version,
                "model_id": self.model_id,
                "model_revision": getattr(
                    self.backend_, "effective_model_revision", self.model_revision
                ),
                "node_threshold": self.node_threshold,
                "edge_threshold": self.edge_threshold,
                "use_milp": self.use_milp,
                "connected": self.connected,
                "max_node_degree": self.max_node_degree,
                "solver": "scipy.optimize.milp" if self.use_milp else "thresholds",
                "grounding_stopwords_removed": True,
                "stopwords_path": str(
                    self.stopwords_path or DEFAULT_STOPWORDS_PATH
                ),
                "stopwords_count": len(self.stopwords_),
                "grounding_candidate_words": [word for _, word in content_words],
                "grounding_method": "nli_contrastive_entailment",
                "grounding_model_id": self.grounding_model_id,
                "grounding_candidate_context_radius": getattr(
                    self.grounding_backend_, "candidate_context_radius", None
                ),
                "input_truncated": self.backend_.was_truncated(
                    text,
                    {
                        **node_questions,
                        **edge_questions,
                    },
                ),
            }
        )
        return graph

    def transform(self, text: str | Sequence[str]) -> nx.DiGraph | list[nx.DiGraph]:
        """Transform one paragraph or a sequence of paragraphs.

        A single string returns one ``DiGraph``. A sequence of strings returns
        a list of graphs in the same order as the input.
        """
        check_is_fitted(self, ["ontology_"])
        if not getattr(self, "_model_loaded_", False):
            raise RuntimeError(
                "The Laya model is not loaded. Call load_model() before transform()."
            )
        if not isinstance(text, str):
            if not isinstance(text, Sequence):
                raise TypeError("transform() expects a string or a sequence of strings")
            if not all(isinstance(item, str) for item in text):
                raise TypeError("transform() sequences must contain only strings")
            return [self._transform_one(item) for item in text]
        return self._transform_one(text)

    def display(
        self,
        graph: nx.DiGraph | None = None,
        *,
        text: str | None = None,
        paragraph: str | None = None,
        title: str | None = None,
        ax: Any | None = None,
        figsize: tuple[float, float] = (13.0, 6.5),
        layout: str | Callable[..., Mapping[Any, Any]] = "kamada_kawai",
        seed: int | None = 17,
        node_size: float = 1200.0,
        node_size_min: float = 900.0,
        node_size_max: float = 2000.0,
        scale_node_size_by_probability: bool = False,
        node_color: str = "white",
        node_cmap: str | Any = "viridis",
        edge_cmap: str | Any = "plasma",
        color_by_probability: bool = False,
        node_alpha: float = 0.94,
        node_edge_color: str = "none",
        node_edge_width: float = 0.0,
        edge_color: str = "black",
        edge_width_min: float = 1.0,
        edge_width_max: float = 4.7,
        scale_edge_width_by_probability: bool = False,
        arrowsize: int = 20,
        connectionstyle: str = "arc3,rad=0.08",
        show_node_labels: bool = True,
        show_edge_labels: bool = True,
        show_probabilities: bool = False,
        show_paragraph: bool = True,
        show_legend: bool = False,
        show: bool = True,
    ) -> tuple[Any, Any]:
        """Render a graph and return ``(figure, axes)``.

        Pass an existing graph, or pass ``text=...`` to transform and render
        in one call. Rendering is optional at runtime; install matplotlib via
        the ``notebook`` extra to use this method.
        """
        if graph is None:
            if text is None:
                raise ValueError("Provide either graph or text")
            graph = self.transform(text)
            paragraph = paragraph or text
        elif text is not None:
            raise ValueError("Provide graph or text, not both")
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("graph must be a networkx.DiGraph")
        if node_size < 0:
            raise ValueError("node_size must be non-negative")
        if node_size_min < 0 or node_size_max < node_size_min:
            raise ValueError("node_size_max must be at least node_size_min >= 0")
        if edge_width_min < 0 or edge_width_max < edge_width_min:
            raise ValueError("edge_width_max must be at least edge_width_min >= 0")
        if not 0.0 <= node_alpha <= 1.0:
            raise ValueError("node_alpha must be between 0 and 1")
        if not isinstance(scale_node_size_by_probability, bool):
            raise TypeError("scale_node_size_by_probability must be a bool")
        if not isinstance(scale_edge_width_by_probability, bool):
            raise TypeError("scale_edge_width_by_probability must be a bool")
        if not isinstance(color_by_probability, bool):
            raise TypeError("color_by_probability must be a bool")

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Graph rendering requires matplotlib. Install with "
                "`pip install -e \".[notebook]\"`."
            ) from exc

        if ax is None:
            figure, axes = plt.subplots(figsize=figsize)
        else:
            axes = ax
            figure = ax.figure
        figure.patch.set_facecolor("#f7f8fa")
        axes.set_facecolor("#f7f8fa")

        if callable(layout):
            positions = layout(graph)
        elif layout == "spring":
            positions = nx.spring_layout(graph, seed=seed)
        elif layout == "kamada_kawai":
            positions = nx.kamada_kawai_layout(graph)
        elif layout == "circular":
            positions = nx.circular_layout(graph)
        elif layout == "shell":
            positions = nx.shell_layout(graph)
        else:
            raise ValueError(
                "layout must be 'spring', 'kamada_kawai', 'circular', 'shell', or a callable"
            )

        undirected_edges, directed_edges = self._display_edges(graph)
        display_edges = undirected_edges + directed_edges
        node_probabilities = [
            float(graph.nodes[node].get("probability", 0.0)) for node in graph.nodes
        ]
        node_sizes = (
            [
                node_size_min + (node_size_max - node_size_min) * probability
                for probability in node_probabilities
            ]
            if scale_node_size_by_probability
            else [node_size] * len(node_probabilities)
        )
        edge_probabilities = [
            float(data.get("probability", 0.0)) for _, _, data in display_edges
        ]
        edge_widths = (
            [
                edge_width_min + (edge_width_max - edge_width_min) * probability
                for probability in edge_probabilities
            ]
            if scale_edge_width_by_probability
            else [edge_width_min] * len(edge_probabilities)
        )
        node_color_map = plt.get_cmap(node_cmap) if isinstance(node_cmap, str) else node_cmap
        edge_color_map = plt.get_cmap(edge_cmap) if isinstance(edge_cmap, str) else edge_cmap

        if graph.number_of_nodes() > 0:
            nx.draw_networkx_nodes(
                graph,
                positions,
                ax=axes,
                node_size=node_sizes,
                node_color=node_probabilities if color_by_probability else node_color,
                cmap=node_color_map if color_by_probability else None,
                vmin=0.0 if color_by_probability else None,
                vmax=1.0 if color_by_probability else None,
                alpha=node_alpha,
                edgecolors=node_edge_color,
                linewidths=node_edge_width,
            )
        if display_edges:
            undirected_widths = edge_widths[:len(undirected_edges)]
            directed_widths = edge_widths[len(undirected_edges):]
            undirected_probabilities = edge_probabilities[:len(undirected_edges)]
            directed_probabilities = edge_probabilities[len(undirected_edges):]
        else:
            undirected_widths = []
            directed_widths = []
            undirected_probabilities = []
            directed_probabilities = []
        if undirected_edges:
            nx.draw_networkx_edges(
                graph,
                positions,
                ax=axes,
                edgelist=[(source, target) for source, target, _ in undirected_edges],
                arrows=False,
                width=undirected_widths,
                edge_color=undirected_probabilities if color_by_probability else edge_color,
                edge_cmap=edge_color_map if color_by_probability else None,
                edge_vmin=0.0 if color_by_probability else None,
                edge_vmax=1.0 if color_by_probability else None,
            )
        if directed_edges:
            nx.draw_networkx_edges(
                graph,
                positions,
                ax=axes,
                edgelist=[(source, target) for source, target, _ in directed_edges],
                arrows=True,
                arrowstyle="-|>",
                arrowsize=arrowsize,
                width=directed_widths,
                edge_color=directed_probabilities if color_by_probability else edge_color,
                edge_cmap=edge_color_map if color_by_probability else None,
                edge_vmin=0.0 if color_by_probability else None,
                edge_vmax=1.0 if color_by_probability else None,
                connectionstyle=connectionstyle,
                min_source_margin=14,
                min_target_margin=18,
            )

        if show_node_labels and graph.number_of_nodes() > 0:
            node_labels = {}
            for node, data in graph.nodes(data=True):
                node_labels[node] = self._node_display_label(
                    node, data, show_probabilities
                )
            nx.draw_networkx_labels(
                graph,
                positions,
                labels=node_labels,
                ax=axes,
                font_size=9,
                font_weight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
            )

        if show_edge_labels and display_edges:
            undirected_labels = {
                (source, target): self._edge_display_label(data, show_probabilities)
                for source, target, data in undirected_edges
            }
            directed_labels = {
                (source, target): self._edge_display_label(data, show_probabilities)
                for source, target, data in directed_edges
            }
            if undirected_labels:
                label_graph = nx.Graph()
                label_graph.add_edges_from(undirected_labels)
                nx.draw_networkx_edge_labels(
                    label_graph,
                    positions,
                    edge_labels=undirected_labels,
                    ax=axes,
                    font_size=8,
                    font_color="black",
                    rotate=False,
                    label_pos=0.52,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 2},
                )
            if directed_labels:
                nx.draw_networkx_edge_labels(
                    graph,
                    positions,
                    edge_labels=directed_labels,
                    ax=axes,
                    font_size=8,
                    font_color="black",
                    rotate=False,
                    label_pos=0.52,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 2},
                )

        graph_title = title or "TextGraphicalizer graph"
        axes.set_title(
            f"{graph_title}  ·  {graph.number_of_nodes()} nodes / {len(display_edges)} edges",
            pad=16,
        )
        if show_paragraph and paragraph:
            axes.text(
                0.5,
                -0.08,
                paragraph,
                transform=axes.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                color="#4b5563",
                wrap=True,
            )
        if show_legend:
            axes.text(
                0.01,
                0.01,
                "node color/size = node probability   ·   edge width/color = edge probability   ·   labels = label / word",
                transform=axes.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#64748b",
            )
        axes.axis("off")
        figure.tight_layout()
        if show:
            plt.show()
        return figure, axes

    def fit_transform(
        self, X: Any = None, y: Any = None, **fit_params: Any
    ) -> nx.DiGraph | list[nx.DiGraph]:
        del fit_params
        self.fit(X, y)
        self.load_model()
        return self.transform(X)
