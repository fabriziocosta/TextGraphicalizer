import json

import networkx as nx

from textgraphicalizer.llm_backend import OpenAIGroundingBackend
from textgraphicalizer.span_backend import ConceptDescription


class FakeResponse:
    def __init__(self, payload):
        self.output_text = json.dumps(payload)


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse(self.payload)


class FakeClient:
    def __init__(self, payload):
        self.responses = FakeResponses(payload)


def test_llm_grounding_assigns_exact_node_and_edge_evidence():
    graph = nx.DiGraph()
    graph.add_node("animal", label="Animal")
    graph.add_node("food", label="Food")
    graph.add_edge("animal", "food", label="causes")
    client = FakeClient(
        {
            "nodes": [
                {"node_id": "animal", "evidence": "fox"},
                {"node_id": "food", "evidence": "grapes"},
            ],
            "edges": [
                {
                    "source_id": "animal",
                    "target_id": "food",
                    "relation_label": "causes",
                    "evidence": "caused",
                }
            ],
        }
    )
    backend = OpenAIGroundingBackend(client=client)

    nodes, edges = backend.ground_graph(
        "The Fox caused the Grapes to fall.",
        graph,
        {
            "animal": ConceptDescription("Animal", "A living organism."),
            "food": ConceptDescription("Food", "A source of nourishment."),
        },
        {
            ("animal", "food"): ConceptDescription("causes", "Produces or leads to."),
        },
    )

    assert nodes["animal"]["span"] == "Fox"
    assert nodes["animal"]["span_start"] == 1
    assert nodes["food"]["span"] == "Grapes"
    assert edges[("animal", "food")]["span"] == "caused"
    assert client.responses.kwargs["model"] == "gpt-4.1-mini"
    assert client.responses.kwargs["text"]["format"]["type"] == "json_schema"
    prompt = client.responses.kwargs["input"][1]["content"]
    assert 'node_id="animal"' in prompt
    assert "The Fox caused the Grapes" in prompt


def test_llm_grounding_ignores_duplicate_and_non_verbatim_evidence():
    graph = nx.DiGraph()
    graph.add_node("a", label="A")
    graph.add_node("b", label="B")
    client = FakeClient(
        {
            "nodes": [
                {"node_id": "a", "evidence": "same phrase"},
                {"node_id": "b", "evidence": "invented phrase"},
            ],
            "edges": [],
        }
    )
    backend = OpenAIGroundingBackend(client=client)

    nodes, _ = backend.ground_graph(
        "The same phrase appears here.",
        graph,
        {
            "a": ConceptDescription("A", "First concept."),
            "b": ConceptDescription("B", "Second concept."),
        },
        {},
    )

    assert set(nodes) == {"a"}


def test_llm_backend_requires_system_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAIGroundingBackend()

    try:
        backend.load()
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("expected missing API key to fail")
