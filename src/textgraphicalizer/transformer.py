"""Public scikit-learn-style TextGraphicalizer estimator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .laya_backend import LayaBackend
from .ontology import Ontology, load_ontology
from .optimizer import EdgeEvidence, NodeEvidence, select_graph


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
        connected: bool = False,
        max_node_degree: int | None = None,
    ) -> None:
        self.ontology = ontology
        self.model_id = model_id
        self.model_path = model_path
        self.model_revision = model_revision
        self.device = device
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold
        self.connected = connected
        self.max_node_degree = max_node_degree

    def _validate_parameters(self) -> None:
        for name, value in (
            ("node_threshold", self.node_threshold),
            ("edge_threshold", self.edge_threshold),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1")
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
        self.ontology_ = load_ontology(self.ontology)
        self.backend_ = LayaBackend(
            model_id=self.model_id,
            model_path=self.model_path,
            model_revision=self.model_revision,
            device=self.device,
        ).load()
        self.n_features_in_ = 1
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

    def transform(self, text: str) -> nx.DiGraph:
        check_is_fitted(self, ["ontology_", "backend_"])
        if not isinstance(text, str):
            raise TypeError("transform() expects one paragraph as a string")

        node_questions = self._node_questions()
        node_result = self.backend_.predict(text, node_questions)
        node_answers = node_result.get("answers", {})
        candidate_nodes: list[NodeEvidence] = []
        for index, concept in enumerate(self.ontology_.concepts):
            answer = node_answers.get(f"node_{index}")
            if answer is None:
                raise ValueError(f"Laya did not return an answer for node_{index}")
            probability = float(answer.get("noul"))
            if probability >= self.node_threshold:
                candidate_nodes.append(
                    NodeEvidence(
                        concept_id=concept.id,
                        label=concept.label,
                        probability=probability,
                        confidence=_confidence(answer),
                    )
                )

        edge_questions: dict[str, dict[str, Any]] = {}
        pair_map: dict[str, tuple[str, str]] = {}
        for pair_index, source in enumerate(candidate_nodes):
            for target in candidate_nodes:
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
                        f'to "{target.label}" in this paragraph?'
                    ),
                    "criteria": criteria,
                }

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
            relation_id = max(relation_probabilities, key=relation_probabilities.get)
            edge_probability = max(0.0, min(1.0, 1.0 - no_relation_probability))
            candidate_edges.append(
                EdgeEvidence(
                    source=source_id,
                    target=target_id,
                    label=relation_by_id[relation_id].label,
                    probability=edge_probability,
                    confidence=_confidence(answer),
                )
            )

        graph = select_graph(
            candidate_nodes,
            candidate_edges,
            node_threshold=self.node_threshold,
            edge_threshold=self.edge_threshold,
            connected=self.connected,
            max_node_degree=self.max_node_degree,
        )
        graph.graph.update(
            {
                "ontology_version": self.ontology_.version,
                "model_id": self.model_id,
                "model_revision": getattr(
                    self.backend_, "effective_model_revision", self.model_revision
                ),
                "node_threshold": self.node_threshold,
                "edge_threshold": self.edge_threshold,
                "connected": self.connected,
                "max_node_degree": self.max_node_degree,
                "solver": "scipy.optimize.milp",
                "input_truncated": self.backend_.was_truncated(text),
            }
        )
        return graph

    def fit_transform(self, X: Any = None, y: Any = None, **fit_params: Any) -> nx.DiGraph:
        del fit_params
        self.fit(X, y)
        if not isinstance(X, str):
            raise TypeError("fit_transform() expects one paragraph as a string")
        return self.transform(X)
