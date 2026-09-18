import pytest

from textgraphicalizer import TextGraphicalizer


ONTOLOGY = {
    "version": 1,
    "concepts": [
        {"id": "a", "label": "A", "description": "A concept."},
        {"id": "b", "label": "B", "description": "B concept."},
        {"id": "c", "label": "C", "description": "C concept."},
    ],
    "relations": [
        {"id": "causes", "label": "causes", "description": "causes"},
    ],
}


class FakeBackend:
    def predict(self, text, questions):
        answers = {}
        for question_id, question in questions.items():
            if question_id.startswith("node_"):
                index = int(question_id.split("_")[1])
                answers[question_id] = {
                    "type": "noul",
                    "noul": [0.9, 0.8, 0.1][index],
                    "confidence": 0.7,
                }
            else:
                answers[question_id] = {
                    "type": "choice",
                    "choice": "causes",
                    "probabilities": {"causes": 0.8, "no_relation": 0.2},
                    "confidence": 0.6,
                }
        return {"answers": answers}

    def was_truncated(self, text):
        return False


def fitted(monkeypatch, **params):
    monkeypatch.setattr(
        "textgraphicalizer.transformer.LayaBackend.load",
        lambda self: FakeBackend(),
    )
    return TextGraphicalizer(ONTOLOGY, **params).fit()


def test_transform_returns_graph_with_evidence(monkeypatch):
    estimator = fitted(monkeypatch)
    graph = estimator.transform("A causes B.")
    assert set(graph.nodes) == {"a", "b"}
    assert graph.nodes["a"]["probability"] == 0.9
    assert graph.edges["a", "b"]["label"] == "causes"
    assert graph.edges["a", "b"]["probability"] == pytest.approx(0.8)
    assert graph.graph["input_truncated"] is False


def test_relation_questions_are_domain_filtered(monkeypatch):
    ontology = {
        **ONTOLOGY,
        "relations": [
            {
                "id": "causes",
                "label": "causes",
                "description": "causes",
                "source_concepts": ["a"],
                "target_concepts": ["b"],
            }
        ],
    }
    monkeypatch.setattr(
        "textgraphicalizer.transformer.LayaBackend.load",
        lambda self: FakeBackend(),
    )
    estimator = TextGraphicalizer(ontology).fit()
    graph = estimator.transform("A causes B.")
    assert set(graph.edges) == {("a", "b")}


def test_transform_requires_string(monkeypatch):
    estimator = fitted(monkeypatch)
    with pytest.raises(TypeError):
        estimator.transform(["not a paragraph"])
