"""Public scikit-learn-style TextGraphicalizer estimator."""

from __future__ import annotations

import json
import logging
import re
import textwrap
from collections import Counter
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Callable, Mapping, cast
from uuid import uuid4

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .laya_backend import LayaBackend
from .llm_backend import OpenAIGroundingBackend
from .ontology import Ontology, load_ontology
from .optimizer import EdgeEvidence, NodeEvidence, select_graph, select_graph_by_threshold
from .span_backend import ConceptDescription, SpanGroundingBackend, SpanScore
from .stopwords import DEFAULT_STOPWORDS_PATH, load_stopwords

# Kept as a module-level compatibility name for callers/tests that used the
# old grounding backend's patch point.  The implementation is now span-based.
NliGroundingBackend = SpanGroundingBackend

logger = logging.getLogger(__name__)


_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_D3_SOURCE_URLS = (
    "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js",
    "https://unpkg.com/d3@7.9.0/dist/d3.min.js",
)
_D3_SOURCE_CACHE: str | None = None


def _load_d3_source() -> str:
    """Fetch D3 once so notebook HTML can run without external script loading."""
    global _D3_SOURCE_CACHE
    if _D3_SOURCE_CACHE is not None:
        return _D3_SOURCE_CACHE
    from urllib.request import Request, urlopen

    for url in _D3_SOURCE_URLS:
        try:
            request = Request(url, headers={"User-Agent": "TextGraphicalizer/0.1"})
            with urlopen(request, timeout=10) as response:
                source = cast(str, response.read().decode("utf-8"))
            if source:
                _D3_SOURCE_CACHE = source
                return source
        except Exception as exc:
            logger.debug("Could not fetch D3 source from %s: %s", url, exc)
    return ""


def _confidence(answer: Mapping[str, Any]) -> float | None:
    value = answer.get("confidence")
    return float(value) if value is not None else None


class TextGraphicalizer(BaseEstimator, TransformerMixin):
    """Convert one document into an ontology-constrained directed graph."""

    _MIN_GROUNDING_SCORE = 0.2
    _MIN_GROUNDING_MARGIN = 0.1
    _GROUNDING_TOP_K = 5
    _SPAN_LENGTH_PENALTY = 0.04
    _SPAN_STOPWORD_PENALTY = 0.03
    _SPAN_FREQUENCY_BONUS = 0.003
    _SPAN_ASSIGNMENT_MARGIN = 0.015

    def __init__(
        self,
        ontology: str | Path | Mapping[str, Any] | Ontology | None = None,
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
        grounding_model_id: str = "cross-encoder/stsb-distilroberta-base",
        use_llm: bool = False,
        llm_model: str = "gpt-4.1-mini",
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
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.load_model()

    def _load_configuration(self) -> None:
        """Load the ontology and stopwords configured on the estimator."""
        if self.ontology is None:
            raise ValueError(
                "ontology must be set before fitting or transforming; "
                "assign estimator.ontology or pass it to the constructor"
            )
        self.ontology_ = load_ontology(self.ontology)
        self.stopwords_ = load_stopwords(self.stopwords_path)
        self._loaded_ontology_source = self.ontology
        self._loaded_stopwords_source = self.stopwords_path
        self.n_features_in_ = 1

    def _ensure_configuration(self) -> None:
        """Load configuration lazily so it can be assigned after init."""
        if (
            not hasattr(self, "ontology_")
            or getattr(self, "_loaded_ontology_source", None) is not self.ontology
            or getattr(self, "_loaded_stopwords_source", None) is not self.stopwords_path
        ):
            self._load_configuration()

    def _validate_parameters(self) -> None:
        if not isinstance(self.grounding_model_id, str) or not self.grounding_model_id:
            raise TypeError("grounding_model_id must be a non-empty string")
        if not isinstance(self.use_llm, bool):
            raise TypeError("use_llm must be a bool")
        if not isinstance(self.llm_model, str) or not self.llm_model:
            raise TypeError("llm_model must be a non-empty string")
        if not self.llm_model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
            raise ValueError("llm_model must be an OpenAI model ID")
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
        self._load_configuration()
        return self

    def load_model(self) -> "TextGraphicalizer":
        """Load Laya and return this estimator.

        Loading happens automatically during construction and may download a
        large checkpoint and initialize a device-specific runtime. This method
        can also be called explicitly and is idempotent for this estimator
        instance. Ontology and stopwords configuration is loaded when present,
        but may be assigned after construction.
        """
        self._validate_parameters()
        if self.ontology is not None:
            self._ensure_configuration()
        if not getattr(self, "_model_loaded_", False):
            self.backend_: LayaBackend = LayaBackend(
                model_id=self.model_id,
                model_path=self.model_path,
                model_revision=self.model_revision,
                device=self.device,
            ).load()
            self._model_loaded_ = True

        desired_grounding_signature = self._grounding_signature()
        if getattr(self, "_loaded_grounding_signature", None) != desired_grounding_signature:
            self._load_grounding_backend()
        return self

    def _grounding_signature(self) -> tuple[bool, str, str | None]:
        """Return the settings that determine the active grounding backend."""
        if self.use_llm:
            return True, self.llm_model, None
        return False, self.grounding_model_id, self.device

    def _load_grounding_backend(self) -> None:
        """Load the grounding backend selected by the current settings."""
        if self.use_llm:
            self.grounding_backend_: Any = OpenAIGroundingBackend(
                model_id=self.llm_model,
            ).load()
        else:
            self.grounding_backend_ = SpanGroundingBackend(
                model_id=self.grounding_model_id,
                device=self.device,
            ).load()
        self._loaded_grounding_signature = self._grounding_signature()

    def _node_questions(self) -> dict[str, dict[str, Any]]:
        return {
            f"node_{index}": {
                "type": "noul",
                "instructions": (
                    f'Is the concept "{concept.label}" expressed in this document? '
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
        concepts_by_node: Mapping[str, Any],
        grounding_terms_by_node: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Ground selected nodes with the span backend.

        The legacy word backend shape is still accepted for lightweight
        downstream test doubles, but production instances use ``score_spans``.
        """
        if graph.number_of_nodes() == 0 or not concepts_by_node:
            return {}
        if hasattr(self.grounding_backend_, "score_spans"):
            candidates = self.grounding_backend_.generate_candidates(text)
            scores = self.grounding_backend_.score_spans(text, candidates, concepts_by_node)
            return self._best_spans(
                scores,
                grounding_terms_by_node,
                stopwords=self.stopwords_,
                unique=True,
            )
        if not words:
            return {}
        legacy_targets = {
            target: (
                value
                if isinstance(value, str)
                else f"Concept: {value.label}\nDescription: {value.description}"
            )
            for target, value in concepts_by_node.items()
        }
        scores = self.grounding_backend_.score_words_contrastive(text, words, legacy_targets)
        return self._best_nli_words(scores, grounding_terms_by_node)

    def _edge_words_by_nli(
        self,
        text: str,
        words: Sequence[tuple[int, str]],
        graph: nx.DiGraph,
        relations_by_edge: Mapping[tuple[str, str], Any],
        grounding_terms_by_edge: Mapping[tuple[str, str], Sequence[str]] | None = None,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Ground selected edges with the span backend."""
        if graph.number_of_edges() == 0 or not relations_by_edge:
            return {}
        if hasattr(self.grounding_backend_, "score_spans"):
            candidates = self.grounding_backend_.generate_candidates(text)
            scores = self.grounding_backend_.score_spans(text, candidates, relations_by_edge)
            return self._best_spans(
                scores,
                grounding_terms_by_edge,
                require_anchor=True,
                stopwords=self.stopwords_,
            )
        if not words:
            return {}
        legacy_targets = {
            target: (
                value
                if isinstance(value, str)
                else f"Concept: {value.label}\nDescription: {value.description}"
            )
            for target, value in relations_by_edge.items()
        }
        scores = self.grounding_backend_.score_words_contrastive(text, words, legacy_targets)
        return self._best_nli_words(scores, grounding_terms_by_edge)

    @classmethod
    def _best_spans(
        cls,
        scores: Mapping[Any, Sequence[SpanScore]],
        preferred_terms: Mapping[Any, Sequence[str]] | None = None,
        *,
        require_anchor: bool = False,
        stopwords: Collection[str] | None = None,
        unique: bool = False,
    ) -> dict[Any, dict[str, Any]]:
        """Attach the best span and a ranked diagnostic shortlist.

        STS checkpoints tend to reward longer pieces of an otherwise identical
        document.  When an ontology provides optional grounding terms, use a
        matching one-word span as a lexical anchor; the cross-encoder still
        ranks all candidates and supplies the diagnostic distribution.
        """
        assignments = (
            cls._unique_span_assignments(
                scores,
                preferred_terms,
                require_anchor=require_anchor,
                stopwords=stopwords,
            )
            if unique
            else {}
        )
        best: dict[Any, dict[str, Any]] = {}
        for target, candidates in scores.items():
            mention_counts = Counter(
                candidate.text.casefold()
                for candidate in candidates
                if candidate.end_word - candidate.start_word == 1
            )
            score_ranked = sorted(
                candidates,
                key=lambda candidate: (
                    cls._span_selection_score(candidate, stopwords, mention_counts),
                    candidate.score,
                ),
                reverse=True,
            )
            if not score_ranked:
                continue
            terms = (preferred_terms or {}).get(target, ())
            anchored = [
                candidate
                for candidate in score_ranked
                if any(cls._grounding_span_matches(candidate, term) for term in terms)
            ]
            if require_anchor and not anchored:
                continue
            if anchored:
                term_order = {
                    candidate: min(
                        index
                        for index, term in enumerate(terms)
                        if cls._grounding_term_matches(candidate.text, term)
                    )
                    for candidate in anchored
                }
                ranked = sorted(
                    anchored,
                    key=lambda candidate: (term_order[candidate], -candidate.score),
                ) + [candidate for candidate in score_ranked if candidate not in anchored]
            else:
                ranked = score_ranked
            if unique:
                winner = assignments.get(target)
                if winner is None:
                    continue
                ranked = [winner] + [
                    candidate
                    for candidate in ranked
                    if candidate.text.casefold() != winner.text.casefold()
                ]
            winner = ranked[0]
            score_distribution = [
                {
                    "span": candidate.text,
                    "start_word": candidate.start_word,
                    "end_word": candidate.end_word,
                    "score": candidate.score,
                }
                for candidate in ranked
            ]
            best[target] = {
                "span": winner.text,
                "span_start": winner.start_word,
                "span_end": winner.end_word,
                "span_score": winner.score,
                "grounding_candidates": [
                    {"span": candidate.text, "score": candidate.score}
                    for candidate in ranked[:cls._GROUNDING_TOP_K]
                ],
                "grounding_score_distribution": score_distribution,
            }
            if winner.end_word - winner.start_word == 1:
                best[target].update(
                    {
                        "word": winner.text,
                        "word_index": winner.start_word,
                        "word_score": winner.score,
                    }
                )
        return best

    @classmethod
    def _unique_span_assignments(
        cls,
        scores: Mapping[Any, Sequence[SpanScore]],
        preferred_terms: Mapping[Any, Sequence[str]] | None,
        *,
        require_anchor: bool,
        stopwords: Collection[str] | None,
    ) -> dict[Any, SpanScore]:
        """Assign distinct surface spans to nodes when the evidence supports it."""
        targets = list(scores)
        options: dict[Any, dict[str, tuple[SpanScore, float]]] = {}
        dummy_scores: dict[Any, float] = {}
        surfaces: set[str] = set()

        for target, candidates in scores.items():
            terms = (preferred_terms or {}).get(target, ())
            anchored = [
                candidate
                for candidate in candidates
                if any(cls._grounding_span_matches(candidate, term) for term in terms)
            ]
            if require_anchor and not anchored:
                options[target] = {}
                dummy_scores[target] = 0.0
                continue
            mention_counts = Counter(
                candidate.text.casefold()
                for candidate in candidates
                if candidate.end_word - candidate.start_word == 1
            )
            target_options: dict[str, tuple[SpanScore, float]] = {}
            for candidate in candidates:
                utility = cls._span_selection_score(
                    candidate,
                    stopwords,
                    mention_counts,
                )
                matching_terms = [
                    index
                    for index, term in enumerate(terms)
                    if cls._grounding_span_matches(candidate, term)
                ]
                if matching_terms:
                    utility += 1.0 - 0.01 * min(matching_terms)
                surface = candidate.text.casefold()
                current = target_options.get(surface)
                if current is None or utility > current[1]:
                    target_options[surface] = (candidate, utility)
            options[target] = target_options
            surfaces.update(target_options)
            dummy_scores[target] = (
                max(utility for _, utility in target_options.values())
                - cls._SPAN_ASSIGNMENT_MARGIN
                if target_options
                else 0.0
            )

        surface_list = sorted(surfaces)
        surface_index = {surface: index for index, surface in enumerate(surface_list)}
        dummy_start = len(surface_list)
        invalid_score = -1.0e6
        utility_matrix = np.full(
            (len(targets), len(surface_list) + len(targets)),
            invalid_score,
            dtype=float,
        )
        for row, target in enumerate(targets):
            for surface, (_, utility) in options[target].items():
                utility_matrix[row, surface_index[surface]] = utility
            utility_matrix[row, dummy_start + row] = dummy_scores[target]

        rows, columns = linear_sum_assignment(-utility_matrix)
        assignments: dict[Any, SpanScore] = {}
        for row, column in zip(rows, columns):
            if column >= dummy_start:
                continue
            surface = surface_list[column]
            assignments[targets[row]] = options[targets[row]][surface][0]
        return assignments

    @staticmethod
    def _span_selection_score(
        candidate: SpanScore,
        stopwords: Collection[str] | None = None,
        mention_counts: Mapping[str, int] | None = None,
    ) -> float:
        """Prefer concise evidence over generic multi-word context fragments."""
        length = candidate.end_word - candidate.start_word
        stopword_count = sum(
            word.casefold() in (stopwords or ())
            for word in _WORD_RE.findall(candidate.text)
        )
        frequency_bonus = 0.0
        if length == 1 and mention_counts:
            frequency_bonus = TextGraphicalizer._SPAN_FREQUENCY_BONUS * min(
                max(0, mention_counts.get(candidate.text.casefold(), 1) - 1),
                3,
            )
        return (
            candidate.score
            - TextGraphicalizer._SPAN_LENGTH_PENALTY * max(0, length - 1)
            - TextGraphicalizer._SPAN_STOPWORD_PENALTY * stopword_count
            + frequency_bonus
        )

    @classmethod
    def _grounding_span_matches(cls, candidate: SpanScore, term: str) -> bool:
        """Match a one-word anchor or an explicitly configured phrase anchor."""
        if len(term.split()) > 1:
            return candidate.text.casefold() == term.casefold()
        return (
            candidate.end_word - candidate.start_word == 1
            and cls._grounding_term_matches(candidate.text, term)
        )

    @staticmethod
    def _grounding_term_matches(word: str, term: str) -> bool:
        word = word.casefold()
        term = term.casefold()
        if word == term:
            return True
        variants = {word}
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                base = word[: -len(suffix)]
                variants.add(base)
                variants.add(base + "e")
        return term in variants

    @staticmethod
    def _best_nli_words(
        scores: Mapping[Any, Sequence[tuple[int, str, float]]],
        preferred_terms: Mapping[Any, Sequence[str]] | None = None,
    ) -> dict[Any, dict[str, Any]]:
        """Choose a strong candidate, optionally restricted to ontology terms."""
        best: dict[Any, dict[str, Any]] = {}
        for target, candidates in scores.items():
            if not candidates:
                continue
            terms = (preferred_terms or {}).get(target, ())
            term_matches = {
                candidate: next(
                    (
                        index
                        for index, term in enumerate(terms)
                        if TextGraphicalizer._grounding_term_matches(candidate[1], term)
                    ),
                    None,
                )
                for candidate in candidates
            }
            preferred = [candidate for candidate in candidates if term_matches[candidate] is not None]
            if preferred:
                best_term_index = min(
                    index for index in term_matches.values() if index is not None
                )
                preferred = [
                    candidate
                    for candidate in preferred
                    if term_matches[candidate] == best_term_index
                ]
            ranked = sorted(preferred or candidates, key=lambda item: item[2], reverse=True)
            token_index, word, score = ranked[0]
            runner_up_score = ranked[1][2] if len(ranked) > 1 else float("-inf")
            if not preferred and (
                score < TextGraphicalizer._MIN_GROUNDING_SCORE
                or score - runner_up_score < TextGraphicalizer._MIN_GROUNDING_MARGIN
            ):
                continue
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
            for field in ("paraphrase", "span", "word"):
                display_values = list(dict.fromkeys(
                    str(value)
                    for value in (data.get(field), reverse_data.get(field))
                    if value is not None
                ))
                if display_values:
                    merged[field] = " / ".join(display_values)
            for field in ("probability", "existence_probability", "relation_probability"):
                numeric_values = [
                    float(value)
                    for value in (data.get(field), reverse_data.get(field))
                    if value is not None
                ]
                if numeric_values:
                    merged[field] = max(numeric_values)
            undirected.append((source, target, merged))

        return undirected, directed

    @staticmethod
    def _grounding_display_value(data: Mapping[str, Any]) -> str | None:
        """Return the human-readable grounding, whether span or paraphrase."""
        for field in ("paraphrase", "span", "word"):
            value = data.get(field)
            if value is not None and str(value):
                return str(value)
        return None

    @staticmethod
    def _node_display_parts(
        node: Any,
        data: Mapping[str, Any],
        show_probabilities: bool,
    ) -> list[tuple[str, str]]:
        parts = [(str(data.get("label", node)).lower(), "monospace")]
        grounding = TextGraphicalizer._grounding_display_value(data)
        if grounding is not None:
            parts.append((grounding, "serif"))
        if show_probabilities:
            parts.append((f"{float(data.get('probability', 0.0)):.2f}", "serif"))
        return parts

    @classmethod
    def _node_display_label(cls, node: Any, data: Mapping[str, Any], show_probabilities: bool) -> str:
        """Return the node label text without applying display styling."""
        return "\n".join(
            text for text, _ in cls._node_display_parts(node, data, show_probabilities)
        )

    @classmethod
    def _draw_node_labels(
        cls,
        graph: nx.DiGraph,
        positions: Mapping[Any, Any],
        axes: Any,
        show_probabilities: bool,
    ) -> None:
        """Draw lowercase monospace labels and normal serif grounding text."""
        for node, data in graph.nodes(data=True):
            parts = cls._node_display_parts(node, data, show_probabilities)
            line_spacing = 10.0
            center = (len(parts) - 1) / 2
            for index, (text, fontfamily) in enumerate(parts):
                axes.annotate(
                    text,
                    xy=positions[node],
                    xytext=(0.0, (center - index) * line_spacing),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontfamily=fontfamily,
                    fontweight="normal",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.72,
                        "pad": 1.5,
                    },
                )

    @staticmethod
    def _wrap_display_text(document: str, max_char: int) -> str:
        """Wrap annotation text at a predictable character width."""
        return "\n".join(
            textwrap.fill(
                line,
                width=max_char,
                break_long_words=False,
                break_on_hyphens=False,
            )
            for line in document.splitlines()
        )

    @staticmethod
    def _edge_display_label(data: Mapping[str, Any], show_probabilities: bool) -> str:
        relation_values = (
            data.get("relation_id"),
            data.get("relation"),
            data.get("label"),
        )
        normalized_relations = {
            re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
            for value in relation_values
            if value is not None
        }
        if "is_a" in normalized_relations:
            return ""
        parts = []
        if data.get("label") is not None:
            parts.append(str(data["label"]))
        grounding = TextGraphicalizer._grounding_display_value(data)
        if grounding is not None:
            parts.append(grounding)
        if show_probabilities:
            parts.append(f"{float(data.get('probability', 0.0)):.2f}")
        return "\n".join(parts)

    def _transform_one(self, text: str) -> nx.DiGraph:
        """Transform one document after readiness checks have completed."""
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
                        f'to "{target.label}" in this document? Choose no_relation '
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
                    relation_id=relation_id,
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
            str(node): ConceptDescription(
                label=concept_by_id[node].label,
                description=concept_by_id[node].description,
            )
            for node in graph.nodes
            if node in concept_by_id
        }
        node_grounding_terms = {
            str(node): concept_by_id[node].grounding_terms
            for node in graph.nodes
            if node in concept_by_id
        }

        relation_by_label = {relation.label: relation for relation in relation_by_id.values()}
        edge_descriptions = {
            (str(source), str(target)): ConceptDescription(
                label=relation_by_label[data["label"]].label,
                description=(
                    f"{relation_by_label[data['label']].description} "
                    f"This relation is from the concept "
                    f"\"{graph.nodes[source]['label']}\" to the concept "
                    f"\"{graph.nodes[target]['label']}\"."
                ),
            )
            for source, target, data in graph.edges(data=True)
            if data.get("label") in relation_by_label
        }
        edge_grounding_terms = {
            (str(source), str(target)): (
                relation_by_label[data["label"]].grounding_terms
                or (relation_by_label[data["label"]].label,)
            )
            for source, target, data in graph.edges(data=True)
            if data.get("label") in relation_by_label
        }
        if self.use_llm:
            node_words, edge_words = self.grounding_backend_.ground_graph(
                text,
                graph,
                node_descriptions,
                edge_descriptions,
            )
        else:
            node_words = self._node_words_by_nli(
                text,
                content_words,
                graph,
                node_descriptions,
                node_grounding_terms,
            )
            edge_words = self._edge_words_by_nli(
                text,
                content_words,
                graph,
                edge_descriptions,
                edge_grounding_terms,
            )
        self._attach_node_words(graph, node_words)
        self._attach_edge_words(graph, edge_words)
        span_backend = hasattr(self.grounding_backend_, "generate_candidates")
        candidate_spans = (
            self.grounding_backend_.generate_candidates(text) if span_backend else []
        )
        uses_word_grounding = not span_backend and not self.use_llm
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
                "grounding_stopwords_removed": uses_word_grounding,
                "stopwords_path": str(
                    self.stopwords_path or DEFAULT_STOPWORDS_PATH
                ),
                "stopwords_count": len(self.stopwords_),
                "grounding_candidate_words": [
                    word for _, word in content_words
                ] if uses_word_grounding or self.use_llm else _WORD_RE.findall(text),
                "grounding_candidate_spans": [
                    candidate.text for candidate in candidate_spans
                ],
                "grounding_method": (
                    "openai_llm_graph_assignment"
                    if self.use_llm
                    else (
                        "cross_encoder_span_similarity"
                        if span_backend
                        else "nli_contrastive_entailment"
                    )
                ),
                "grounding_model_id": (
                    self.llm_model if self.use_llm else self.grounding_model_id
                ),
                "llm_model": self.llm_model if self.use_llm else None,
                "grounding_value_type": "paraphrase" if self.use_llm else "span",
                "grounding_top_k": self._GROUNDING_TOP_K,
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
        """Transform one document or a sequence of documents.

        A single string returns one ``DiGraph``. A sequence of strings returns
        a list of graphs in the same order as the input.
        """
        self._ensure_configuration()
        check_is_fitted(self, ["ontology_", "stopwords_"])
        if not getattr(self, "_model_loaded_", False):
            raise RuntimeError(
                "The Laya model is not loaded. Call load_model() before transform()."
            )
        # Configuration options are intentionally mutable after construction.
        # Refresh the grounding backend if, for example, a notebook changes
        # ``use_llm`` from its default after automatic model loading.
        self.load_model()
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
        document: str | None = None,
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
        max_char: int = 100,
        show_legend: bool = False,
        show: bool = True,
    ) -> tuple[Any, Any]:
        """Render a graph and return ``(figure, axes)``.

        Pass an existing graph, or pass ``text=...`` to transform and render
        in one call. ``document=...`` annotates the graph with its source
        document; ``paragraph=...`` remains a backward-compatible alias.
        Rendering is optional at runtime; install matplotlib via the
        ``notebook`` extra to use this method.
        """
        if document is not None and paragraph is not None:
            raise ValueError("Provide document or paragraph, not both")
        if graph is None:
            if text is None:
                raise ValueError("Provide either graph or text")
            graph = self.transform(text)
            document = document or paragraph or text
        elif text is not None:
            raise ValueError("Provide graph or text, not both")
        else:
            document = document or paragraph
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
        if not isinstance(max_char, int) or isinstance(max_char, bool) or max_char < 1:
            raise ValueError("max_char must be a positive integer")

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
            self._draw_node_labels(
                graph,
                positions,
                axes,
                show_probabilities,
            )

        if show_edge_labels and display_edges:
            undirected_labels = {}
            for source, target, data in undirected_edges:
                label = self._edge_display_label(data, show_probabilities)
                if label:
                    undirected_labels[(source, target)] = label
            directed_labels = {}
            for source, target, data in directed_edges:
                label = self._edge_display_label(data, show_probabilities)
                if label:
                    directed_labels[(source, target)] = label
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
        if show_paragraph and document:
            display_document = self._wrap_display_text(document, max_char)
            axes.text(
                0.5,
                -0.08,
                display_document,
                transform=axes.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                color="#4b5563",
                wrap=False,
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

    def display_d3(
        self,
        graph: nx.DiGraph | None = None,
        *,
        text: str | None = None,
        document: str | None = None,
        paragraph: str | None = None,
        title: str | None = None,
        width: int = 1000,
        height: int = 620,
        show_node_labels: bool = True,
        show_edge_labels: bool = True,
        show_probabilities: bool = False,
        show_document: bool = True,
        max_char: int = 100,
        inline_d3: bool = True,
    ) -> Any:
        """Return an interactive D3 force-directed graph for notebook display."""
        if document is not None and paragraph is not None:
            raise ValueError("Provide document or paragraph, not both")
        if graph is None:
            if text is None:
                raise ValueError("Provide either graph or text")
            graph = self.transform(text)
            document = document or paragraph or text
        elif text is not None:
            raise ValueError("Provide graph or text, not both")
        else:
            document = document or paragraph
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("graph must be a networkx.DiGraph")
        for name, value in (("width", width), ("height", height), ("max_char", max_char)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("show_node_labels", show_node_labels),
            ("show_edge_labels", show_edge_labels),
            ("show_probabilities", show_probabilities),
            ("show_document", show_document),
            ("inline_d3", inline_d3),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")

        try:
            from IPython.display import HTML
        except ImportError as exc:
            raise ImportError(
                "D3 graph rendering requires IPython. Install with "
                "`pip install -e \".[notebook]\"`."
            ) from exc

        graph_id = f"textgraphicalizer-d3-{uuid4().hex}"
        nodes = [
            {
                "id": str(node),
                "label": str(data.get("label", node)).lower(),
                "evidence": self._grounding_display_value(data) or "",
                "probability": float(data.get("probability", 0.0)),
            }
            for node, data in graph.nodes(data=True)
        ]
        links = [
            {
                "source": str(source),
                "target": str(target),
                "label": (
                    self._edge_display_label(data, show_probabilities)
                    if show_edge_labels
                    else ""
                ),
                "probability": float(data.get("probability", 0.0)),
            }
            for source, target, data in graph.edges(data=True)
        ]
        payload = {
            "title": title or "TextGraphicalizer graph",
            "document": (
                self._wrap_display_text(document, max_char)
                if show_document and document
                else ""
            ),
            "nodes": nodes,
            "links": links,
        }
        data_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
        inline_d3_source = (
            json.dumps(_load_d3_source()).replace("<", "\\u003c")
            if inline_d3
            else '""'
        )
        html = """
<div id="__GRAPH_ID__" class="textgraphicalizer-d3"></div>
<script>
(function () {
  const container = document.getElementById("__GRAPH_ID__");
  const data = __GRAPH_DATA__;
  const width = __WIDTH__;
  const height = __HEIGHT__;
  const inlineD3Source = __INLINE_D3_SOURCE__;

  function loadD3() {
    if (window.d3) return Promise.resolve(window.d3);
    if (window.__textGraphicalizerD3PromiseV2) {
      return window.__textGraphicalizerD3PromiseV2;
    }
    let inlineError = null;
    if (inlineD3Source) {
      try {
        const script = document.createElement("script");
        // Jupyter exposes RequireJS' AMD `define`; hide it so D3 creates its
        // global `d3` object instead of registering an inaccessible module.
        script.textContent = "(function () { const define = undefined;\\n"
          + inlineD3Source + "\\n}).call(window);";
        document.head.appendChild(script);
        if (window.d3) return Promise.resolve(window.d3);
        inlineError = new Error("Inline D3.js source did not create window.d3");
      } catch (error) {
        inlineError = error;
      }
    }
    const sources = [
      "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js",
      "https://unpkg.com/d3@7.9.0/dist/d3.min.js",
      "https://d3js.org/d3.v7.min.js",
    ];
    const promise = new Promise((resolve, reject) => {
      function trySource(index, lastError) {
        if (window.d3) {
          resolve(window.d3);
          return;
        }
        if (index >= sources.length) {
          reject(lastError || new Error("No D3.js source could be loaded"));
          return;
        }
        const script = document.createElement("script");
        script.src = sources[index];
        script.onload = () => {
          if (window.d3) resolve(window.d3);
          else trySource(index + 1, new Error("D3.js loaded without a global d3"));
        };
        script.onerror = (error) => trySource(index + 1, error);
        document.head.appendChild(script);
      }
      trySource(0, inlineError);
    });
    window.__textGraphicalizerD3PromiseV2 = promise;
    promise.catch(() => {
      if (window.__textGraphicalizerD3PromiseV2 === promise) {
        window.__textGraphicalizerD3PromiseV2 = null;
      }
    });
    return promise;
  }

  function render(d3) {
    const root = d3.select(container);
    root.html("");
    root.append("div").attr("class", "textgraphicalizer-d3-title").text(data.title);
    const svg = root.append("svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("width", "100%")
      .attr("height", height)
      .style("background", "#f7f8fa")
      .style("border-radius", "6px")
      .style("cursor", "grab");
    const defs = svg.append("defs");
    defs.append("marker")
      .attr("id", "textgraphicalizer-arrow-__GRAPH_ID__")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 18)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#94a3b8");

    const layer = svg.append("g");
    svg.call(d3.zoom().scaleExtent([0.25, 4]).on("zoom", (event) => {
      layer.attr("transform", event.transform);
    }));
    const simulation = d3.forceSimulation(data.nodes)
      .force("link", d3.forceLink(data.links).id((d) => d.id).distance(150))
      .force("charge", d3.forceManyBody().strength(-480))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(42));
    const links = layer.append("g")
      .attr("stroke", "#94a3b8")
      .attr("stroke-opacity", 0.7)
      .selectAll("line")
      .data(data.links)
      .join("line")
      .attr("stroke-width", (d) => 1 + 3 * d.probability)
      .attr("marker-end", "url(#textgraphicalizer-arrow-__GRAPH_ID__)");
    const linkLabels = layer.append("g")
      .attr("font-family", "serif")
      .attr("font-size", 11)
      .selectAll("text")
      .data(data.links.filter((d) => d.label))
      .join("text")
      .attr("text-anchor", "middle")
      .attr("fill", "#334155")
      .text((d) => d.label);
    const nodes = layer.append("g")
      .selectAll("g")
      .data(data.nodes)
      .join("g")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));
    if (__SHOW_NODE_LABELS__) {
      nodes.append("text")
        .attr("text-anchor", "middle")
        .attr("font-family", "monospace")
        .attr("font-size", 11)
        .attr("fill", "#111827")
        .each(function (d) {
          const text = d3.select(this);
          text.append("tspan").attr("x", 0).attr("dy", "-1.1em").text(d.label);
          if (d.evidence) {
            text.append("tspan").attr("x", 0).attr("dy", "1.2em")
              .attr("font-family", "serif").text(d.evidence);
          }
        });
    }
    if (data.document) {
      root.append("pre").attr("class", "textgraphicalizer-d3-document").text(data.document);
    }
    simulation.on("tick", () => {
      links
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);
      linkLabels
        .attr("x", (d) => (d.source.x + d.target.x) / 2)
        .attr("y", (d) => (d.source.y + d.target.y) / 2);
      nodes.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  }

  loadD3().then(render).catch((error) => {
    console.error("Could not load D3.js for the interactive graph", error);
    container.textContent = "Could not load D3.js. Check notebook network access and rerun this cell.";
  });
})();
</script>
<style>
.textgraphicalizer-d3 { max-width: 1000px; margin: 0.5rem auto 1rem; }
.textgraphicalizer-d3-title { font: 16px sans-serif; margin: 0.4rem 0; color: #111827; }
.textgraphicalizer-d3-document {
  white-space: pre-wrap; font: 14px serif; color: #4b5563;
  margin: 0.7rem 0 0; text-align: center;
}
</style>
"""
        html = (
            html.replace("__GRAPH_ID__", graph_id)
            .replace("__GRAPH_DATA__", data_json)
            .replace("__WIDTH__", str(width))
            .replace("__HEIGHT__", str(height))
            .replace("__INLINE_D3_SOURCE__", inline_d3_source)
            .replace("__SHOW_NODE_LABELS__", "true" if show_node_labels else "false")
        )
        return HTML(html)

    def fit_transform(
        self, X: Any = None, y: Any = None, **fit_params: Any
    ) -> nx.DiGraph | list[nx.DiGraph]:
        del fit_params
        self.fit(X, y)
        self.load_model()
        return self.transform(X)
