"""OpenAI-backed graph paraphrase assignment."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

GROUNDING_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "paraphrase": {"type": "string"},
                },
                "required": ["node_id", "paraphrase"],
                "additionalProperties": False,
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "relation_label": {"type": "string"},
                    "paraphrase": {"type": "string"},
                },
                "required": [
                    "source_id",
                    "target_id",
                    "relation_label",
                    "paraphrase",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["nodes", "edges"],
    "additionalProperties": False,
}

SEMANTIC_MERGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "merges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pair_id": {"type": "string"},
                    "node_a_id": {"type": "string"},
                    "node_b_id": {"type": "string"},
                    "same_entity": {"type": "boolean"},
                    "keep_node_id": {"type": "string"},
                },
                "required": [
                    "pair_id",
                    "node_a_id",
                    "node_b_id",
                    "same_entity",
                    "keep_node_id",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["merges"],
    "additionalProperties": False,
}


class OpenAIGroundingBackend:
    """Use an OpenAI model to assign document paraphrases to a selected graph."""

    _MAX_PARAPHRASE_WORDS = 12

    def __init__(
        self,
        model_id: str = "gpt-4.1-mini",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.api_key = api_key
        self.client = client

    def load(self) -> "OpenAIGroundingBackend":
        """Create the OpenAI client using ``OPENAI_API_KEY`` when needed."""
        if self.client is not None:
            return self
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LLM grounding requires the openai package to be installed."
            ) from exc
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM grounding requires the OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=api_key)
        return self

    @staticmethod
    def _concept_parts(concept: Any) -> tuple[str, str]:
        if isinstance(concept, Mapping):
            return str(concept["label"]), str(concept["description"])
        return str(concept.label), str(concept.description)

    @classmethod
    def _graph_description(
        cls,
        graph: Any,
        concepts: Mapping[str, Any],
    ) -> str:
        lines = ["NODES:"]
        for node_id, concept in concepts.items():
            label, description = cls._concept_parts(concept)
            lines.append(f'- node_id="{node_id}" label="{label}"')
            lines.append(f"  concept_description: {description}")
            connections: list[str] = []
            for source, target, data in graph.edges(data=True):
                relation = str(data.get("label", "related to"))
                if str(source) == node_id:
                    target_label = graph.nodes[target].get("label", target)
                    connections.append(f'to node_id="{target}" ({target_label}) via {relation}')
                elif str(target) == node_id:
                    source_label = graph.nodes[source].get("label", source)
                    connections.append(f'from node_id="{source}" ({source_label}) via {relation}')
            if connections:
                lines.append("  connections: " + "; ".join(connections))
        lines.append("EDGES:")
        for source, target, data in graph.edges(data=True):
            lines.append(
                f'- source_id="{source}" target_id="{target}" '
                f'relation="{data.get("label", "related to")}"'
            )
        return "\n".join(lines)

    def _request_assignments(
        self,
        text: str,
        graph: Any,
        concepts: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self.client is None:
            self.load()
        if self.client is None:
            raise RuntimeError("OpenAI client is not loaded")
        system_prompt = (
            "You assign concise, context-sensitive paraphrases from a document to "
            "a selected concept graph. Return only the requested structured output. "
            "A paraphrase is a newly worded expression of how the document conveys "
            "the concept or relation; it does not have to be a verbatim substring. "
            "Use a meaningful short phrase, normally two to eight words and never "
            "more than twelve words, "
            "not a generic function word or the ontology label by itself. Give each "
            "node the best supported paraphrase, including when the concept is "
            "implicit or abstract. For entity and category concepts, return one "
            "focused noun phrase for the salient concrete story referent that "
            "instantiates the concept "
            "rather than a broad summary of the whole story. Avoid phrases such as "
            "the story involves ... when a specific referent can be identified. "
            "For example, for entity, physical entity, and animal nodes all "
            "expressed by a goose, prefer the focused phrase the goose. "
            "Use an empty string only when the document gives no meaningful support "
            "at all."
        )
        user_prompt = (
            "DOCUMENT:\n"
            f"{text}\n\n"
            "SELECTED CONCEPT GRAPH:\n"
            f"{self._graph_description(graph, concepts)}\n\n"
            "TASK:\n"
            "For every listed node, write the short paraphrase that best explains "
            "how the document expresses that concept, using its connected nodes "
            "and relations as context. For every listed edge, write a short "
            "paraphrase of how that relation is expressed in the document. Prefer "
            "different paraphrases when the concepts are distinct, but do not force "
            "a distinction that the document does not support."
        )
        response = self.client.responses.create(
            model=self.model_id,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "graph_grounding",
                    "strict": True,
                    "schema": GROUNDING_RESPONSE_SCHEMA,
                }
            },
            store=False,
        )
        raw_output = getattr(response, "output_text", "")
        if not raw_output:
            raise ValueError("OpenAI grounding response did not contain output_text")
        payload = json.loads(raw_output)
        if not isinstance(payload, Mapping):
            raise ValueError("OpenAI grounding response must be a JSON object")
        return payload

    @classmethod
    def _normalize_paraphrase(cls, paraphrase: Any) -> str | None:
        if not isinstance(paraphrase, str):
            return None
        normalized = " ".join(paraphrase.split())
        tokens = _WORD_RE.findall(normalized)
        if not tokens or len(tokens) > cls._MAX_PARAPHRASE_WORDS:
            return None
        return normalized

    @staticmethod
    def _paraphrase_data(paraphrase: str) -> dict[str, Any]:
        return {
            "paraphrase": paraphrase,
            "grounding_score": 1.0,
            "grounding_method": "openai_llm_paraphrase",
        }

    def ground_graph(
        self,
        text: str,
        graph: Any,
        concepts: Mapping[str, Any],
        edges: Mapping[tuple[str, str], Any],
    ) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
        """Assign node and edge paraphrases with one structured model call."""
        payload = self._request_assignments(text, graph, concepts)
        node_result: dict[str, dict[str, Any]] = {}
        raw_nodes = payload.get("nodes", [])
        if isinstance(raw_nodes, list):
            for item in raw_nodes:
                if not isinstance(item, Mapping):
                    continue
                node_id = str(item.get("node_id", ""))
                if node_id not in concepts or node_id in node_result:
                    continue
                paraphrase = self._normalize_paraphrase(
                    item.get("paraphrase", item.get("evidence", ""))
                )
                if paraphrase is None:
                    continue
                node_result[node_id] = self._paraphrase_data(paraphrase)

        edge_result: dict[tuple[str, str], dict[str, Any]] = {}
        raw_edges = payload.get("edges", [])
        if isinstance(raw_edges, list):
            for item in raw_edges:
                if not isinstance(item, Mapping):
                    continue
                key = (str(item.get("source_id", "")), str(item.get("target_id", "")))
                if key not in edges or key in edge_result:
                    continue
                paraphrase = self._normalize_paraphrase(
                    item.get("paraphrase", item.get("evidence", ""))
                )
                if paraphrase is not None:
                    edge_result[key] = self._paraphrase_data(paraphrase)
        return node_result, edge_result

    @staticmethod
    def _node_expression(data: Mapping[str, Any]) -> str:
        for field in ("paraphrase", "span", "word"):
            value = data.get(field)
            if value is not None and str(value):
                return str(value)
        return "(no grounded expression)"

    @classmethod
    def _semantic_pair_description(
        cls,
        pair_id: str,
        node_a: str,
        node_b: str,
        graph: Any,
        concepts: Mapping[str, Any],
    ) -> str:
        lines = [f'PAIR {pair_id}: node_a_id="{node_a}" node_b_id="{node_b}"']
        for node_id in (node_a, node_b):
            concept = concepts[node_id]
            label, description = cls._concept_parts(concept)
            expression = cls._node_expression(graph.nodes[node_id])
            lines.append(
                f'  node_id="{node_id}" label="{label}" '
                f'concept_description="{description}" '
                f'document_expression="{expression}"'
            )
        connections: list[str] = []
        for source, target, data in graph.edges(data=True):
            if {str(source), str(target)} != {node_a, node_b}:
                continue
            connections.append(
                f'{source} -> {target} via {data.get("label", "related to")}'
            )
        if connections:
            lines.append("  connecting_relations: " + "; ".join(connections))
        return "\n".join(lines)

    def _request_semantic_merges(
        self,
        text: str,
        graph: Any,
        concepts: Mapping[str, Any],
        pairs: list[tuple[str, str, str]],
    ) -> Mapping[str, Any]:
        if self.client is None:
            self.load()
        if self.client is None:
            raise RuntimeError("OpenAI client is not loaded")
        pair_descriptions = "\n".join(
            self._semantic_pair_description(pair_id, node_a, node_b, graph, concepts)
            for pair_id, node_a, node_b in pairs
        )
        system_prompt = (
            "You decide whether adjacent ontology nodes refer to the same story "
            "referent. A referent can be a thing, person, animal, object, event, "
            "or other story element. Same entity means coreference, not merely "
            "semantic relatedness or a type/subtype relationship. For example, "
            "a goose, an animal, and a physical entity can refer to the same goose "
            "when their document expressions support that reading. If they are "
            "the same referent, even if one node uses a generic ontology concept, "
            "compare the concrete story referents expressed by the nodes rather "
            "than treating the generic label as a separate entity. A broad node "
            "that names several different participants should not be merged with "
            "all of them. Choose the more specific ontology concept to keep; "
            "never keep a generic supertype when the other concept is its specific "
            "instance or subtype. An explicit is_a edge points from the more "
            "specific source to the more general target. Return only the requested "
            "structured output."
        )
        user_prompt = (
            "DOCUMENT:\n"
            f"{text}\n\n"
            "ADJACENT NODE PAIRS:\n"
            f"{pair_descriptions}\n\n"
            "TASK:\n"
            "For every pair, set same_entity to true only when both nodes refer "
            "to the same concrete story referent. A generic concept and a specific "
            "concept can still be the same referent: if both document expressions "
            "explicitly identify the same goose, person, object, or event, set it "
            "to true even when their ontology labels differ. If true, set keep_node_id to the more "
            "specific concept; otherwise set keep_node_id to an empty string. "
            "Do not merge nodes only because one is related to, connected to, or "
            "a superclass of the other."
        )
        response = self.client.responses.create(
            model=self.model_id,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "semantic_graph_merges",
                    "strict": True,
                    "schema": SEMANTIC_MERGE_RESPONSE_SCHEMA,
                }
            },
            store=False,
        )
        raw_output = getattr(response, "output_text", "")
        if not raw_output:
            raise ValueError("OpenAI semantic merge response did not contain output_text")
        payload = json.loads(raw_output)
        if not isinstance(payload, Mapping):
            raise ValueError("OpenAI semantic merge response must be a JSON object")
        return payload

    def find_semantic_merges(
        self,
        text: str,
        graph: Any,
        concepts: Mapping[str, Any],
    ) -> list[tuple[str, str, str]]:
        """Ask the model which adjacent node pairs co-refer and which to keep."""
        pairs: list[tuple[str, str, str]] = []
        seen: set[frozenset[str]] = set()
        for source, target in graph.edges:
            node_a, node_b = str(source), str(target)
            if node_a == node_b or node_a not in concepts or node_b not in concepts:
                continue
            if not all(
                any(
                    graph.nodes[node_id].get(field) is not None
                    and str(graph.nodes[node_id].get(field))
                    for field in ("paraphrase", "span", "word")
                )
                for node_id in (source, target)
            ):
                continue
            key = frozenset((node_a, node_b))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((f"pair_{len(pairs)}", node_a, node_b))
        if not pairs:
            return []

        payload = self._request_semantic_merges(text, graph, concepts, pairs)
        pair_by_id = {
            pair_id: (node_a, node_b)
            for pair_id, node_a, node_b in pairs
        }
        pair_by_nodes = {
            frozenset((node_a, node_b)): (node_a, node_b)
            for _, node_a, node_b in pairs
        }
        results: list[tuple[str, str, str]] = []
        raw_merges = payload.get("merges", [])
        if not isinstance(raw_merges, list):
            return results
        for item in raw_merges:
            if not isinstance(item, Mapping) or item.get("same_entity") is not True:
                continue
            pair = pair_by_id.get(str(item.get("pair_id", "")))
            if pair is None:
                pair = pair_by_nodes.get(
                    frozenset((str(item.get("node_a_id", "")), str(item.get("node_b_id", ""))))
                )
            if pair is None:
                continue
            keep_node_id = str(item.get("keep_node_id", ""))
            if keep_node_id in pair:
                results.append((pair[0], pair[1], keep_node_id))
        return results
