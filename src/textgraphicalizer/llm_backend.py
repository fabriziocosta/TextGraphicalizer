"""OpenAI-backed graph evidence assignment."""

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
                    "evidence": {"type": "string"},
                },
                "required": ["node_id", "evidence"],
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
                    "evidence": {"type": "string"},
                },
                "required": [
                    "source_id",
                    "target_id",
                    "relation_label",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["nodes", "edges"],
    "additionalProperties": False,
}


class OpenAIGroundingBackend:
    """Use an OpenAI model to assign document evidence to a selected graph."""

    _MAX_EVIDENCE_WORDS = 8

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
            "You assign exact evidence from a document to a selected concept graph. "
            "Return only the requested structured output. Evidence must be copied "
            "verbatim as a contiguous substring of the document, preserving its "
            "capitalization. Prefer a meaningful word or short phrase, not generic "
            "function words or surrounding context. Assign distinct evidence to "
            "different nodes when the document supports that; use an empty string "
            "when no clear evidence exists."
        )
        user_prompt = (
            "DOCUMENT:\n"
            f"{text}\n\n"
            "SELECTED CONCEPT GRAPH:\n"
            f"{self._graph_description(graph, concepts)}\n\n"
            "TASK:\n"
            "For every listed node, choose the word or short phrase from the "
            "document that best expresses that concept, using its connected nodes "
            "and relations as context. For every listed edge, choose the exact "
            "word or short phrase that expresses the relation, or use an empty "
            "string if no relation expression is present. Do not invent text."
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
    def _resolve_evidence(
        cls,
        text: str,
        evidence: Any,
    ) -> tuple[str, int, int] | None:
        if not isinstance(evidence, str):
            return None
        tokens = [token.casefold() for token in _WORD_RE.findall(evidence)]
        if not tokens or len(tokens) > cls._MAX_EVIDENCE_WORDS:
            return None
        matches = list(_WORD_RE.finditer(text))
        for start in range(len(matches) - len(tokens) + 1):
            candidate_tokens = [
                match.group(0).casefold()
                for match in matches[start:start + len(tokens)]
            ]
            if candidate_tokens != tokens:
                continue
            end = start + len(tokens)
            return text[matches[start].start():matches[end - 1].end()], start, end
        return None

    @staticmethod
    def _span_data(span: tuple[str, int, int]) -> dict[str, Any]:
        value, start, end = span
        data: dict[str, Any] = {
            "span": value,
            "span_start": start,
            "span_end": end,
            "span_score": 1.0,
            "grounding_candidates": [{"span": value, "score": 1.0}],
            "grounding_score_distribution": [
                {
                    "span": value,
                    "start_word": start,
                    "end_word": end,
                    "score": 1.0,
                }
            ],
        }
        if end - start == 1:
            data.update({"word": value, "word_index": start, "word_score": 1.0})
        return data

    def ground_graph(
        self,
        text: str,
        graph: Any,
        concepts: Mapping[str, Any],
        edges: Mapping[tuple[str, str], Any],
    ) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
        """Assign node and edge evidence with one structured model call."""
        payload = self._request_assignments(text, graph, concepts)
        node_result: dict[str, dict[str, Any]] = {}
        used_node_surfaces: set[str] = set()
        raw_nodes = payload.get("nodes", [])
        if isinstance(raw_nodes, list):
            for item in raw_nodes:
                if not isinstance(item, Mapping):
                    continue
                node_id = str(item.get("node_id", ""))
                if node_id not in concepts or node_id in node_result:
                    continue
                resolved = self._resolve_evidence(text, item.get("evidence", ""))
                if resolved is None or resolved[0].casefold() in used_node_surfaces:
                    continue
                used_node_surfaces.add(resolved[0].casefold())
                node_result[node_id] = self._span_data(resolved)

        edge_result: dict[tuple[str, str], dict[str, Any]] = {}
        raw_edges = payload.get("edges", [])
        if isinstance(raw_edges, list):
            for item in raw_edges:
                if not isinstance(item, Mapping):
                    continue
                key = (str(item.get("source_id", "")), str(item.get("target_id", "")))
                if key not in edges or key in edge_result:
                    continue
                resolved = self._resolve_evidence(text, item.get("evidence", ""))
                if resolved is not None:
                    edge_result[key] = self._span_data(resolved)
        return node_result, edge_result
