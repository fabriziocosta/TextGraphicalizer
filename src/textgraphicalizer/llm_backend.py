"""Provider-backed graph paraphrase assignment."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

DEFAULT_OPENAI_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_OLLAMA_LLM_MODEL = "gemma4:12b-mlx"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 600.0
DEFAULT_MLX_LM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MLX_LM_MODEL = "local-model"
DEFAULT_MLX_LM_TIMEOUT = 600.0
DEFAULT_MLX_LM_TEMPERATURE = 0.0
DEFAULT_MLX_LM_MAX_TOKENS = 2048
DEFAULT_MLX_LM_MODEL_PATH = (
    "/Users/f.costa/Documents/Codex/2026-09-19/referenced-chatgpt-conversation-this-is-an/"
    "models/GLM-4.7-Flash-4bit"
)
DEFAULT_MLX_LM_PYTHON = "/Users/f.costa/.venvs/py312/bin/python"
DEFAULT_MLX_LM_SERVER_HOST = "127.0.0.1"
DEFAULT_MLX_LM_SERVER_PORT = 8080
DEFAULT_MLX_LM_SERVER_LOG_LEVEL = "INFO"

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

RELATION_REVIEW_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "edge_id": {"type": "string"},
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "keep_edge": {"type": "boolean"},
                    "relation_id": {"type": "string"},
                },
                "required": [
                    "edge_id",
                    "source_id",
                    "target_id",
                    "keep_edge",
                    "relation_id",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["edges"],
    "additionalProperties": False,
}


class OpenAIGroundingBackend:
    """Use an OpenAI model to assign document paraphrases to a selected graph."""

    _MAX_PARAPHRASE_WORDS = 12
    provider = "openai"

    def __init__(
        self,
        model_id: str = DEFAULT_OPENAI_LLM_MODEL,
        api_key: str | None = None,
        client: Any | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        provider_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.api_key = api_key
        self.client = client
        self.api_key_env = api_key_env
        self.provider_config = dict(provider_config or {})

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
        api_key = self.api_key or os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"LLM grounding requires the {self.api_key_env} environment variable."
            )
        self.client = OpenAI(api_key=api_key)
        return self

    @staticmethod
    def _parse_structured_output(raw_output: str, provider: str) -> Mapping[str, Any]:
        """Parse JSON objects with optional Markdown fences or short preambles."""
        cleaned = raw_output.strip()
        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced is not None:
            cleaned = fenced.group(1).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            decoder = json.JSONDecoder()
            payload = None
            start = cleaned.find("{")
            while start >= 0:
                try:
                    candidate, _ = decoder.raw_decode(cleaned[start:])
                except json.JSONDecodeError:
                    start = cleaned.find("{", start + 1)
                    continue
                payload = candidate
                break
            if payload is None:
                raise ValueError(
                    f"{provider} grounding response was not valid JSON"
                ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"{provider} grounding response must be a JSON object")
        return payload

    def _structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        schema_name: str,
    ) -> Mapping[str, Any]:
        """Request one schema-constrained response from the active provider."""
        if self.client is None:
            self.load()
        if self.client is None:
            raise RuntimeError("OpenAI client is not loaded")
        response = self.client.responses.create(
            model=self.model_id,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
            store=False,
        )
        raw_output = getattr(response, "output_text", "")
        if not raw_output:
            raise ValueError("OpenAI grounding response did not contain output_text")
        return self._parse_structured_output(raw_output, "OpenAI")

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
        return self._structured_completion(
            system_prompt,
            user_prompt,
            GROUNDING_RESPONSE_SCHEMA,
            "graph_grounding",
        )

    @classmethod
    def _normalize_paraphrase(cls, paraphrase: Any) -> str | None:
        if not isinstance(paraphrase, str):
            return None
        normalized = " ".join(paraphrase.split())
        tokens = _WORD_RE.findall(normalized)
        if not tokens or len(tokens) > cls._MAX_PARAPHRASE_WORDS:
            return None
        return normalized

    def _paraphrase_data(self, paraphrase: str) -> dict[str, Any]:
        return {
            "paraphrase": paraphrase,
            "grounding_score": 1.0,
            "grounding_method": f"{self.provider}_llm_paraphrase",
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
        return self._structured_completion(
            system_prompt,
            user_prompt,
            SEMANTIC_MERGE_RESPONSE_SCHEMA,
            "semantic_graph_merges",
        )

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

    def _request_relation_review(
        self,
        text: str,
        graph: Any,
        concepts: Mapping[str, Any],
        relations: Mapping[str, Any],
        allowed_relations: Mapping[tuple[str, str], tuple[str, ...]],
    ) -> Mapping[str, Any]:
        edge_descriptions: list[str] = []
        for edge_index, (source, target, data) in enumerate(graph.edges(data=True)):
            source_id, target_id = str(source), str(target)
            edge_id = f"edge_{edge_index}"
            source_label = graph.nodes[source].get("label", source)
            target_label = graph.nodes[target].get("label", target)
            source_concept = self._concept_parts(concepts[source_id])
            target_concept = self._concept_parts(concepts[target_id])
            orientations = [(source_id, target_id)]
            if source_id != target_id:
                orientations.append((target_id, source_id))
            orientation_options = []
            for orientation_source, orientation_target in orientations:
                options = allowed_relations.get(
                    (orientation_source, orientation_target),
                    (),
                )
                option_text = "; ".join(
                    f'{relation_id}={self._concept_parts(relations[relation_id])[0]}: '
                    f'{self._concept_parts(relations[relation_id])[1]}'
                    for relation_id in options
                    if relation_id in relations
                )
                orientation_options.append(
                    f'{orientation_source} -> {orientation_target}: '
                    f'{option_text or "(none)"}'
                )
            edge_descriptions.append(
                f'EDGE edge_id="{edge_id}" current_source_id="{source_id}" '
                f'current_target_id="{target_id}"\n'
                f'  source_label="{source_label}" source_concept="{source_concept[0]}: '
                f'{source_concept[1]}" expression="{self._node_expression(graph.nodes[source])}"\n'
                f'  target_label="{target_label}" target_concept="{target_concept[0]}: '
                f'{target_concept[1]}" expression="{self._node_expression(graph.nodes[target])}"\n'
                f'  current_relation="{data.get("label", "")}" '
                f'current_relation_expression="{self._node_expression(data)}"\n'
                f'  allowed_ontology_relations_by_direction: '
                f'{" | ".join(orientation_options)}'
            )
        system_prompt = (
            "You revise the relationships in a story graph. For every listed edge, "
            "choose the source and target direction and exactly one relation_id from "
            "that direction's allowed ontology relations, using the document and the "
            "source/target concepts as context. For agent-patient relations, put the "
            "agent, actor, causer, or giver in source_id and the patient, affected "
            "entity, recipient, or result in target_id. The current relation may be "
            "wrong or may be a merged placeholder. "
            "keep_edge to false when none of the allowed ontology relations is "
            "supported. Never invent relation IDs, labels, or relations outside the "
            "provided ontology. Return only the requested structured output."
        )
        user_prompt = (
            "DOCUMENT:\n"
            f"{text}\n\n"
            "ONTOLOGY RELATION REVIEW:\n"
            f"{chr(10).join(edge_descriptions)}\n\n"
            "TASK:\n"
            "Return one decision for every edge. If keep_edge is true, relation_id "
            "must be allowed for the returned source_id -> target_id direction. "
            "The returned endpoints must be the two endpoints of the listed edge, "
            "possibly reversed. If keep_edge is false, use an empty relation_id."
        )
        return self._structured_completion(
            system_prompt,
            user_prompt,
            RELATION_REVIEW_RESPONSE_SCHEMA,
            "ontology_relation_review",
        )

    def revise_relationships(
        self,
        text: str,
        graph: Any,
        concepts: Mapping[str, Any],
        relations: Mapping[str, Any],
        allowed_relations: Mapping[tuple[str, str], tuple[str, ...]],
    ) -> dict[tuple[str, str], tuple[str, str, str, bool]]:
        """Relabel and orient surviving edges with ontology relations."""
        if not graph.edges:
            return {}
        payload = self._request_relation_review(
            text,
            graph,
            concepts,
            relations,
            allowed_relations,
        )
        edge_result: dict[tuple[str, str], tuple[str, str, str, bool]] = {}
        edge_by_id = {
            f"edge_{edge_index}": (str(source), str(target))
            for edge_index, (source, target) in enumerate(graph.edges)
        }
        graph_edges = set(edge_by_id.values())
        raw_edges = payload.get("edges", [])
        if not isinstance(raw_edges, list):
            return edge_result
        for item in raw_edges:
            if not isinstance(item, Mapping):
                continue
            current_key = edge_by_id.get(str(item.get("edge_id", "")))
            if current_key is None or current_key in edge_result:
                continue
            keep_edge = item.get("keep_edge") is True
            source_id = str(item.get("source_id", ""))
            target_id = str(item.get("target_id", ""))
            relation_id = str(item.get("relation_id", ""))
            if not keep_edge:
                edge_result[current_key] = (*current_key, "", False)
                continue
            oriented_key = (source_id, target_id)
            if oriented_key not in graph_edges and oriented_key != current_key[::-1]:
                continue
            if relation_id not in allowed_relations.get(oriented_key, ()):
                continue
            edge_result[current_key] = (source_id, target_id, relation_id, keep_edge)
        return edge_result


class OllamaGroundingBackend(OpenAIGroundingBackend):
    """Use a locally hosted Ollama chat model for the same graph tasks."""

    provider = "ollama"

    def __init__(
        self,
        model_id: str = DEFAULT_OLLAMA_LLM_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
        client: Any | None = None,
    ) -> None:
        # ``client`` is retained as a small testing/integration escape hatch;
        # normal operation talks to Ollama's local HTTP API directly.
        super().__init__(model_id=model_id, client=client)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def load(self) -> "OllamaGroundingBackend":
        """Validate the endpoint configuration without starting a server."""
        if not self.base_url:
            raise ValueError("Ollama base_url must be a non-empty URL")
        if self.timeout <= 0:
            raise ValueError("Ollama timeout must be positive")
        return self

    def _structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        schema_name: str,
    ) -> Mapping[str, Any]:
        del schema_name
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            # Gemma 4 advertises thinking support. The graph task needs the
            # constrained JSON answer, not a separate reasoning trace; leaving
            # thinking enabled can make a local 12B request exceed its timeout.
            "think": False,
            "format": schema,
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"Ollama request failed with HTTP {exc.code} at {self.base_url}; "
                f"check that model {self.model_id!r} is available"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama did not finish the request within {self.timeout:g} seconds "
                f"at {self.base_url}. The model may still be loading; increase "
                "ollama_timeout or use a smaller model."
            ) from exc
        except (URLError, OSError) as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. Start Ollama "
                f"with `ollama serve` and make sure model {self.model_id!r} is available."
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError("Ollama response was not valid JSON") from exc

        message = response_payload.get("message")
        raw_output = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise ValueError("Ollama response did not contain message.content")
        return self._parse_structured_output(raw_output, "Ollama")
