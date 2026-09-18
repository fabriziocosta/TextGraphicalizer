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
    return TextGraphicalizer(ONTOLOGY, **params).fit().load_model()


def test_fit_does_not_load_model(monkeypatch):
    calls = []

    def load(self):
        calls.append(self)
        return FakeBackend()

    monkeypatch.setattr("textgraphicalizer.transformer.LayaBackend.load", load)
    estimator = TextGraphicalizer(ONTOLOGY).fit()
    assert calls == []
    with pytest.raises(RuntimeError, match="Call load_model\(\)"):
        estimator.transform("A causes B.")


def test_load_model_is_explicit_and_idempotent(monkeypatch):
    calls = []

    def load(self):
        calls.append(self)
        return FakeBackend()

    monkeypatch.setattr("textgraphicalizer.transformer.LayaBackend.load", load)
    estimator = TextGraphicalizer(ONTOLOGY).fit()
    assert estimator.load_model() is estimator
    assert estimator.load_model() is estimator
    assert len(calls) == 1


def test_load_model_can_be_called_without_fit(monkeypatch):
    monkeypatch.setattr(
        "textgraphicalizer.transformer.LayaBackend.load",
        lambda self: FakeBackend(),
    )
    estimator = TextGraphicalizer(ONTOLOGY).load_model()
    graph = estimator.transform("A causes B.")
    assert set(graph.nodes) == {"a", "b"}


def test_transform_returns_graph_with_evidence(monkeypatch):
    estimator = fitted(monkeypatch)
    graph = estimator.transform("A causes B.")
    assert set(graph.nodes) == {"a", "b"}
    assert graph.nodes["a"]["probability"] == 0.9
    assert graph.edges["a", "b"]["label"] == "causes"
    assert graph.edges["a", "b"]["probability"] == pytest.approx(0.8)
    assert graph.edges["a", "b"]["existence_probability"] == pytest.approx(0.8)
    assert graph.edges["a", "b"]["relation_probability"] == pytest.approx(0.8)
    assert graph.graph["input_truncated"] is False


def test_transform_sequence_returns_graphs(monkeypatch):
    estimator = fitted(monkeypatch)
    graphs = estimator.transform(["A causes B.", "B causes A."])
    assert isinstance(graphs, list)
    assert len(graphs) == 2
    assert all(graph.is_directed() for graph in graphs)


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


def test_display_is_parameterized_and_returns_matplotlib_objects(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimator = fitted(monkeypatch)
    graph = estimator.transform("A causes B.")
    figure, axes = estimator.display(
        graph,
        title="Custom graph",
        layout="circular",
        node_size_min=300,
        node_size_max=800,
        scale_node_size_by_probability=True,
        color_by_probability=False,
        node_color="#2563eb",
        edge_color="#64748b",
        scale_edge_width_by_probability=False,
        show_probabilities=False,
        show_paragraph=False,
        show_legend=False,
        show=False,
    )
    assert figure is axes.figure
    assert axes.get_title().startswith("Custom graph")
    plt.close(figure)
