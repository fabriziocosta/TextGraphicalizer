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


def test_llm_grounding_assigns_node_and_edge_paraphrases():
    graph = nx.DiGraph()
    graph.add_node("animal", label="Animal")
    graph.add_node("food", label="Food")
    graph.add_edge("animal", "food", label="causes")
    client = FakeClient(
        {
            "nodes": [
                {"node_id": "animal", "paraphrase": "the hungry fox"},
                {"node_id": "food", "paraphrase": "fruit hanging overhead"},
            ],
            "edges": [
                {
                    "source_id": "animal",
                    "target_id": "food",
                    "relation_label": "causes",
                    "paraphrase": "the fox's attempt to reach the fruit",
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

    assert nodes["animal"]["paraphrase"] == "the hungry fox"
    assert "span" not in nodes["animal"]
    assert nodes["food"]["paraphrase"] == "fruit hanging overhead"
    assert edges[("animal", "food")]["paraphrase"] == (
        "the fox's attempt to reach the fruit"
    )
    assert client.responses.kwargs["model"] == "gpt-4.1-mini"
    assert client.responses.kwargs["text"]["format"]["type"] == "json_schema"
    prompt = client.responses.kwargs["input"][1]["content"]
    assert 'node_id="animal"' in prompt
    assert "The Fox caused the Grapes" in prompt
    assert "does not have to be a verbatim substring" in client.responses.kwargs["input"][0]["content"]


def test_llm_grounding_accepts_repeated_and_non_verbatim_paraphrases():
    graph = nx.DiGraph()
    graph.add_node("a", label="A")
    graph.add_node("b", label="B")
    client = FakeClient(
        {
            "nodes": [
                {"node_id": "a", "paraphrase": "the first idea"},
                {"node_id": "b", "paraphrase": "the first idea"},
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

    assert set(nodes) == {"a", "b"}


def test_llm_grounding_rejects_empty_or_overlong_paraphrases():
    graph = nx.DiGraph()
    graph.add_node("a", label="A")
    client = FakeClient(
        {
            "nodes": [
                {"node_id": "a", "paraphrase": ""},
            ],
            "edges": [],
        }
    )
    backend = OpenAIGroundingBackend(client=client)

    nodes, _ = backend.ground_graph(
        "A document.",
        graph,
        {"a": ConceptDescription("A", "A concept.")},
        {},
    )

    assert nodes == {}


def test_llm_backend_finds_same_entity_merges_for_adjacent_nodes():
    graph = nx.DiGraph()
    graph.add_node("entity", label="Entity", paraphrase="the bird")
    graph.add_node("animal", label="Animal", paraphrase="the goose")
    graph.add_edge("animal", "entity", relation_id="is_a", label="is a")
    client = FakeClient(
        {
            "merges": [
                {
                    "pair_id": "pair_0",
                    "node_a_id": "animal",
                    "node_b_id": "entity",
                    "same_entity": True,
                    "keep_node_id": "animal",
                }
            ]
        }
    )
    backend = OpenAIGroundingBackend(client=client)

    merges = backend.find_semantic_merges(
        "A goose walked by.",
        graph,
        {
            "entity": ConceptDescription("Entity", "Something that exists."),
            "animal": ConceptDescription("Animal", "A living organism."),
        },
    )

    assert merges == [("animal", "entity", "animal")]
    assert client.responses.kwargs["text"]["format"]["name"] == "semantic_graph_merges"
    prompt = client.responses.kwargs["input"][1]["content"]
    assert 'node_a_id="animal"' in prompt
    assert "same story referent" in client.responses.kwargs["input"][0]["content"]
    assert "concrete story referent" in client.responses.kwargs["input"][0]["content"]


def test_llm_backend_requires_system_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAIGroundingBackend()

    try:
        backend.load()
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("expected missing API key to fail")
