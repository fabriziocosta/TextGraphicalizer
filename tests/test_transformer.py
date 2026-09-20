import networkx as nx
import pytest

from textgraphicalizer import Relation, TextGraphicalizer
from textgraphicalizer.span_backend import Span, SpanScore

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
    def __init__(self):
        self.predicted_states = []

    def predict(self, text, questions):
        self.predicted_states.append(text)
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
                is_a_to_b = 'from "A" to "B"' in question["instructions"]
                answers[question_id] = {
                    "type": "choice",
                    "choice": "causes",
                    "probabilities": (
                        {"causes": 0.8, "no_relation": 0.2}
                        if is_a_to_b
                        else {"causes": 0.05, "no_relation": 0.95}
                    ),
                    "confidence": 0.6,
                }
        return {"answers": answers}

    def was_truncated(self, text, questions=None):
        return False


class FakeNliBackend:
    def __init__(self, model_id="cross-encoder/nli-distilroberta-base", device="auto"):
        self.model_id = model_id
        self.device = device

    def load(self):
        return self

    def score_words(self, text, words, targets):
        del text
        result = {}
        for target in targets:
            result[target] = [
                (token_index, word, 0.9 if word == "infection" else 0.1)
                for token_index, word in words
            ]
        return result

    def score_words_contrastive(self, text, words, targets):
        return self.score_words(text, words, targets)


def fitted(monkeypatch, **params):
    monkeypatch.setattr(
        "textgraphicalizer.transformer.LayaBackend.load",
        lambda self: FakeBackend(),
    )
    monkeypatch.setattr(
        "textgraphicalizer.transformer.NliGroundingBackend.load",
        lambda self: FakeNliBackend(self.model_id, self.device),
    )
    return TextGraphicalizer(ONTOLOGY, **params).load_model()


def test_init_loads_model_automatically(monkeypatch):
    calls = []

    def load(self):
        calls.append(self)
        return FakeBackend()

    monkeypatch.setattr("textgraphicalizer.transformer.LayaBackend.load", load)
    monkeypatch.setattr(
        "textgraphicalizer.transformer.NliGroundingBackend.load",
        lambda self: FakeNliBackend(self.model_id, self.device),
    )
    estimator = TextGraphicalizer(ONTOLOGY).fit()
    assert len(calls) == 1
    assert estimator.transform("A causes B.").number_of_nodes() == 2


def test_init_without_configuration_loads_model_and_accepts_late_configuration(
    monkeypatch, tmp_path
):
    calls = []

    def load(self):
        calls.append(self)
        return FakeBackend()

    monkeypatch.setattr("textgraphicalizer.transformer.LayaBackend.load", load)
    monkeypatch.setattr(
        "textgraphicalizer.transformer.NliGroundingBackend.load",
        lambda self: FakeNliBackend(self.model_id, self.device),
    )
    stopwords_path = tmp_path / "stopwords.yaml"
    stopwords_path.write_text("stopwords: [the, caused]\n", encoding="utf-8")

    estimator = TextGraphicalizer()
    estimator.ontology = ONTOLOGY
    estimator.stopwords_path = stopwords_path

    graph = estimator.transform("The infection caused a fever.")

    assert len(calls) == 1
    assert set(graph.nodes) == {"a", "b"}
    assert graph.graph["stopwords_path"] == str(stopwords_path)


def test_load_model_remains_idempotent_after_automatic_loading(monkeypatch):
    calls = []

    def load(self):
        calls.append(self)
        return FakeBackend()

    monkeypatch.setattr("textgraphicalizer.transformer.LayaBackend.load", load)
    monkeypatch.setattr(
        "textgraphicalizer.transformer.NliGroundingBackend.load",
        lambda self: FakeNliBackend(self.model_id, self.device),
    )
    estimator = TextGraphicalizer(ONTOLOGY)
    assert estimator.load_model() is estimator
    assert estimator.load_model() is estimator
    assert len(calls) == 1


def test_use_llm_selects_openai_grounding_backend(monkeypatch):
    monkeypatch.setattr(
        "textgraphicalizer.transformer.LayaBackend.load",
        lambda self: FakeBackend(),
    )

    class FakeLlmBackend:
        model_id = "gpt-4.1-mini"

        def ground_graph(self, text, graph, concepts, edges):
            assert text == "A causes B."
            assert set(concepts) == {"a", "b"}
            assert set(edges) == {("a", "b")}
            return (
                {
                    "a": {"paraphrase": "the first concept"},
                    "b": {"paraphrase": "the second concept"},
                },
                {},
            )

    monkeypatch.setattr(
        "textgraphicalizer.transformer.OpenAIGroundingBackend.load",
        lambda self: FakeLlmBackend(),
    )
    monkeypatch.setattr(
        "textgraphicalizer.transformer.SpanGroundingBackend.load",
        lambda self: pytest.fail("cross-encoder should not load in LLM mode"),
    )

    estimator = TextGraphicalizer(ONTOLOGY, use_llm=True)
    graph = estimator.transform("A causes B.")

    assert graph.nodes["a"]["paraphrase"] == "the first concept"
    assert graph.nodes["b"]["paraphrase"] == "the second concept"
    assert graph.graph["grounding_method"] == "openai_llm_graph_assignment"
    assert graph.graph["grounding_model_id"] == "gpt-4.1-mini"
    assert graph.graph["grounding_value_type"] == "paraphrase"


def test_use_llm_selects_ollama_with_local_default_model(monkeypatch):
    monkeypatch.setattr(
        "textgraphicalizer.transformer.LayaBackend.load",
        lambda self: FakeBackend(),
    )

    class FakeLlmBackend:
        model_id = "gemma4:12b-mlx"

        def ground_graph(self, text, graph, concepts, edges):
            del text, graph, concepts, edges
            return {}, {}

    loaded = []

    def load_ollama(self):
        loaded.append((self.model_id, self.base_url, self.timeout))
        return FakeLlmBackend()

    monkeypatch.setattr(
        "textgraphicalizer.transformer.OllamaGroundingBackend.load",
        load_ollama,
    )
    monkeypatch.setattr(
        "textgraphicalizer.transformer.SpanGroundingBackend.load",
        lambda self: pytest.fail("cross-encoder should not load in LLM mode"),
    )

    estimator = TextGraphicalizer(ONTOLOGY, use_llm=True, llm_provider="ollama")
    graph = estimator.transform("A causes B.")

    assert loaded == [("gemma4:12b-mlx", "http://localhost:11434", 600.0)]
    assert graph.graph["grounding_method"] == "ollama_llm_graph_assignment"
    assert graph.graph["llm_provider"] == "ollama"
    assert graph.graph["grounding_model_id"] == "gemma4:12b-mlx"


def test_use_llm_selects_mlx_lm_and_reports_provider_metadata(monkeypatch):
    monkeypatch.setattr(
        "textgraphicalizer.transformer.LayaBackend.load",
        lambda self: FakeBackend(),
    )
    loaded = []

    class FakeLlmBackend:
        def ground_graph(self, text, graph, concepts, edges):
            del text, graph, concepts, edges
            return {}, {}

    def load_mlx(self):
        loaded.append(self)
        return FakeLlmBackend()

    monkeypatch.setattr(
        "textgraphicalizer.transformer.MlxLmGroundingBackend.load",
        load_mlx,
    )
    monkeypatch.setattr(
        "textgraphicalizer.transformer.SpanGroundingBackend.load",
        lambda self: pytest.fail("cross-encoder should not load in LLM mode"),
    )

    estimator = TextGraphicalizer(
        ONTOLOGY,
        use_llm=True,
        llm_provider="mlx-lm",
        llm_model="GLM-4.7-Flash-4bit",
        mlx_lm_base_url="http://127.0.0.1:8080/v1",
    )
    graph = estimator.transform("A causes B.")

    assert len(loaded) == 1
    assert loaded[0].model_id == "GLM-4.7-Flash-4bit"
    assert loaded[0].base_url == "http://127.0.0.1:8080/v1"
    assert graph.graph["llm_provider"] == "mlx-lm"
    assert graph.graph["grounding_model_id"] == "GLM-4.7-Flash-4bit"
    assert graph.graph["llm_provider_config"]["model_path"]


def test_llm_yaml_configuration_is_authoritative(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "textgraphicalizer.transformer.LayaBackend.load",
        lambda self: FakeBackend(),
    )
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        """version: 1
providers:
  mlx-lm:
    model: yaml-model
    base_url: http://yaml-server/v1
    timeout: 12
    temperature: 0.25
    max_tokens: 64
    model_path: /yaml/model
    python_executable: /yaml/python
    auto_start: false
    server:
      host: 127.0.0.2
      port: 9090
      log_level: DEBUG
""",
        encoding="utf-8",
    )
    loaded = []

    class FakeLlmBackend:
        def ground_graph(self, text, graph, concepts, edges):
            del text, graph, concepts, edges
            return {}, {}

    def load_mlx(self):
        loaded.append(self)
        return FakeLlmBackend()

    monkeypatch.setattr(
        "textgraphicalizer.transformer.MlxLmGroundingBackend.load",
        load_mlx,
    )
    estimator = TextGraphicalizer(
        ONTOLOGY,
        use_llm=True,
        llm_provider="mlx-lm",
        llm_model="constructor-model",
        mlx_lm_base_url="http://constructor/v1",
        llm_config_path=config_path,
    )
    estimator.transform("A causes B.")

    assert loaded[0].model_id == "yaml-model"
    assert loaded[0].base_url == "http://yaml-server/v1"
    assert loaded[0].timeout == 12.0
    assert loaded[0].temperature == 0.25
    assert loaded[0].max_tokens == 64
    assert loaded[0].model_path == "/yaml/model"
    assert loaded[0].python_executable == "/yaml/python"
    assert loaded[0].server_host == "127.0.0.2"
    assert loaded[0].server_port == 9090


def test_transform_refreshes_mlx_backend_when_base_url_changes(monkeypatch):
    monkeypatch.setattr(
        "textgraphicalizer.transformer.LayaBackend.load",
        lambda self: FakeBackend(),
    )
    class FakeLlmBackend:
        def __init__(self, base_url):
            self.base_url = base_url

        def ground_graph(self, text, graph, concepts, edges):
            del text, graph, concepts, edges
            return {}, {}

    monkeypatch.setattr(
        "textgraphicalizer.transformer.MlxLmGroundingBackend.load",
        lambda self: FakeLlmBackend(self.base_url),
    )
    estimator = TextGraphicalizer(
        ONTOLOGY,
        use_llm=True,
        llm_provider="mlx-lm",
        mlx_lm_auto_start=False,
    )
    first_backend = estimator.grounding_backend_
    estimator.mlx_lm_base_url = "http://127.0.0.1:9090/v1"
    estimator.transform("A causes B.")

    assert estimator.grounding_backend_ is not first_backend
    assert estimator.grounding_backend_.base_url == "http://127.0.0.1:9090/v1"


def test_transform_refreshes_grounding_backend_when_use_llm_changes(monkeypatch):
    monkeypatch.setattr(
        "textgraphicalizer.transformer.LayaBackend.load",
        lambda self: FakeBackend(),
    )
    monkeypatch.setattr(
        "textgraphicalizer.transformer.SpanGroundingBackend.load",
        lambda self: FakeNliBackend(self.model_id, self.device),
    )
    llm_calls = []

    class FakeLlmBackend:
        model_id = "gpt-4.1-mini"

        def ground_graph(self, text, graph, concepts, edges):
            del text, graph, concepts, edges
            return {}, {}

    def load_llm(self):
        llm_calls.append(self)
        return FakeLlmBackend()

    monkeypatch.setattr(
        "textgraphicalizer.transformer.OpenAIGroundingBackend.load",
        load_llm,
    )

    estimator = TextGraphicalizer(ONTOLOGY)
    assert estimator.use_llm is False
    estimator.use_llm = True

    graph = estimator.transform("A causes B.")

    assert len(llm_calls) == 1
    assert graph.graph["grounding_method"] == "openai_llm_graph_assignment"
    assert isinstance(estimator.grounding_backend_, FakeLlmBackend)


def test_grounding_model_loads_automatically_and_remains_idempotent(monkeypatch):
    laya_calls = []
    nli_calls = []

    def load_laya(self):
        laya_calls.append(self)
        return FakeBackend()

    def load_nli(self):
        nli_calls.append(self)
        return FakeNliBackend(self.model_id, self.device)

    monkeypatch.setattr("textgraphicalizer.transformer.LayaBackend.load", load_laya)
    monkeypatch.setattr("textgraphicalizer.transformer.NliGroundingBackend.load", load_nli)
    estimator = TextGraphicalizer(ONTOLOGY)

    assert estimator.load_model() is estimator
    assert estimator.load_model() is estimator
    assert len(laya_calls) == 1
    assert len(nli_calls) == 1


def test_fit_preserves_an_already_loaded_model(monkeypatch):
    calls = []

    def load(self):
        calls.append(self)
        return FakeBackend()

    monkeypatch.setattr("textgraphicalizer.transformer.LayaBackend.load", load)
    monkeypatch.setattr(
        "textgraphicalizer.transformer.NliGroundingBackend.load",
        lambda self: FakeNliBackend(self.model_id, self.device),
    )
    estimator = TextGraphicalizer(ONTOLOGY).load_model()
    backend = estimator.backend_

    assert estimator.fit() is estimator
    assert estimator.backend_ is backend
    assert len(calls) == 1


def test_fit_transform_loads_model_once(monkeypatch):
    calls = []

    def load(self):
        calls.append(self)
        return FakeBackend()

    monkeypatch.setattr("textgraphicalizer.transformer.LayaBackend.load", load)
    monkeypatch.setattr(
        "textgraphicalizer.transformer.NliGroundingBackend.load",
        lambda self: FakeNliBackend(self.model_id, self.device),
    )
    estimator = TextGraphicalizer(ONTOLOGY)

    graph = estimator.fit_transform("A causes B.")

    assert set(graph.nodes) == {"a", "b"}
    assert len(calls) == 1


def test_threshold_mode_uses_thresholds_without_milp(monkeypatch):
    estimator = fitted(
        monkeypatch,
        use_milp=False,
        node_threshold=0.85,
        connected=True,
    )

    graph = estimator.transform("A causes B.")

    assert set(graph.nodes) == {"a"}
    assert graph.number_of_edges() == 0
    assert graph.graph["use_milp"] is False
    assert graph.graph["solver"] == "thresholds"


def test_load_model_can_be_called_without_fit(monkeypatch):
    monkeypatch.setattr(
        "textgraphicalizer.transformer.LayaBackend.load",
        lambda self: FakeBackend(),
    )
    monkeypatch.setattr(
        "textgraphicalizer.transformer.NliGroundingBackend.load",
        lambda self: FakeNliBackend(self.model_id, self.device),
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


def test_cleanup_absorbs_empty_nodes_and_redirects_their_edges():
    graph = nx.DiGraph()
    graph.add_node("source", label="Source", paraphrase="the source", probability=0.8)
    graph.add_node("empty", label="Empty", probability=0.2)
    graph.add_node("target", label="Target", word="the target", probability=0.9)
    graph.add_node("sink", label="Sink", paraphrase="the sink", probability=0.7)
    graph.add_edge("source", "empty", label="supports", probability=0.4)
    graph.add_edge("empty", "target", label="points to", probability=0.6)
    graph.add_edge("empty", "sink", label="leads to", probability=0.6)
    graph.add_edge("source", "target", label="already exists", probability=0.7)
    graph.add_edge("target", "sink", label="already exists", probability=0.5)

    collapsed, uncollapsed = TextGraphicalizer._collapse_empty_nodes(graph)

    assert collapsed == {"empty": "target"}
    assert uncollapsed == []
    assert set(graph.nodes) == {"source", "target", "sink"}
    assert graph.has_edge("source", "target")
    assert graph.edges["source", "target"]["probability"] == pytest.approx(0.7)
    assert graph.edges["target", "sink"]["probability"] == pytest.approx(0.6)
    assert "leads to" in graph.edges["target", "sink"]["label"]


def test_cleanup_repeats_through_empty_node_chains_and_leaves_unanchored_nodes():
    graph = nx.DiGraph()
    graph.add_node("grounded", label="Grounded", paraphrase="evidence", probability=0.8)
    graph.add_node("first", label="First")
    graph.add_node("second", label="Second")
    graph.add_node("unanchored", label="Unanchored")
    graph.add_edge("grounded", "first", label="r1")
    graph.add_edge("first", "second", label="r2")

    collapsed, uncollapsed = TextGraphicalizer._collapse_empty_nodes(graph)

    assert collapsed == {"first": "grounded", "second": "grounded"}
    assert uncollapsed == ["unanchored"]
    assert set(graph.nodes) == {"grounded", "unanchored"}


def test_semantic_cleanup_keeps_the_specific_is_a_concept_and_redirects_edges():
    graph = nx.DiGraph()
    graph.add_node("entity", label="Entity", paraphrase="the goose", probability=0.9)
    graph.add_node("animal", label="Animal", paraphrase="the goose", probability=0.8)
    graph.add_node("food", label="Food", paraphrase="the egg", probability=0.7)
    graph.add_edge("animal", "entity", relation_id="is_a", label="is a", probability=0.8)
    graph.add_edge("entity", "food", label="related to", probability=0.5)

    collapsed = TextGraphicalizer._collapse_semantic_nodes(
        graph,
        [("animal", "entity", "entity")],
    )

    assert collapsed == {"entity": "animal"}
    assert set(graph.nodes) == {"animal", "food"}
    assert graph.has_edge("animal", "food")


def test_final_relation_review_can_reverse_edges_and_remove_unsupported_edges():
    graph = nx.DiGraph()
    graph.add_node("agent", label="Agent")
    graph.add_node("patient", label="Patient")
    graph.add_node("other", label="Other")
    graph.add_edge("agent", "patient", label="old")
    graph.add_edge("agent", "other", label="unsupported")

    revised, removed = TextGraphicalizer._apply_relation_revisions(
        graph,
        {
            ("agent", "patient"): ("patient", "agent", "helps", True),
            ("agent", "other"): ("agent", "other", "", False),
        },
        {
            "helps": Relation("helps", "helps", "Assists another participant."),
        },
    )

    assert revised == {"agent->patient": "patient->agent:helps"}
    assert removed == ["agent->other"]
    assert not graph.has_edge("agent", "patient")
    assert graph.edges["patient", "agent"]["label"] == "helps"
    assert not graph.has_edge("agent", "other")


def test_grounding_uses_nli_entailment_scoring(monkeypatch):
    estimator = fitted(monkeypatch)
    graph = estimator.transform("The infection caused a fever.")

    assert graph.nodes["a"]["word"] == "infection"
    assert graph.nodes["b"]["word"] == "infection"
    assert graph.edges["a", "b"]["word"] == "infection"
    assert graph.graph["grounding_candidate_words"] == ["infection", "caused", "fever"]
    assert graph.graph["grounding_method"] == "nli_contrastive_entailment"
    assert graph.graph["grounding_model_id"] == "cross-encoder/stsb-distilroberta-base"
    assert graph.nodes["a"]["word_score"] == pytest.approx(0.9)


def test_grounding_hypotheses_do_not_copy_candidate_words(monkeypatch):
    estimator = fitted(monkeypatch)
    captured = []

    def score_words_contrastive(text, words, targets):
        del text, words
        captured.append(targets)
        return {target: [] for target in targets}

    estimator.grounding_backend_.score_words_contrastive = score_words_contrastive
    estimator.transform("The infection caused a fever.")

    assert captured
    assert all(
        "{word}" not in hypothesis
        for targets in captured
        for hypothesis in targets.values()
    )


def test_span_grounding_attaches_offsets_and_top_candidates(monkeypatch):
    estimator = fitted(monkeypatch)
    captured = []

    class FakeSpanBackend:
        @staticmethod
        def generate_candidates(text):
            del text
            return [Span("infection", 1, 2), Span("high fever", 4, 6)]

        def score_spans(self, text, candidates, concepts):
            captured.append((text, candidates, concepts))
            result = {}
            for target in concepts:
                winner = (
                    SpanScore("high fever", 4, 6, 0.83)
                    if target == "b"
                    else SpanScore("infection", 1, 2, 0.83)
                )
                result[target] = [
                    winner,
                    SpanScore("infection", 1, 2, 0.76)
                    if target == "b"
                    else SpanScore("high fever", 4, 6, 0.76),
                ]
            return result

    estimator.grounding_backend_ = FakeSpanBackend()
    graph = estimator.transform("The infection caused a high fever.")

    assert graph.nodes["a"]["span"] == "infection"
    assert graph.nodes["a"]["span_start"] == 1
    assert graph.nodes["a"]["span_end"] == 2
    assert graph.nodes["a"]["word"] == "infection"
    assert graph.nodes["a"]["grounding_candidates"][1]["span"] == "high fever"
    assert graph.graph["grounding_method"] == "cross_encoder_span_similarity"
    assert captured[0][2]["a"].label == "A"


def test_span_grounding_uses_ontology_terms_to_break_generic_score_ties():
    scores = {
        "event": [
            SpanScore("to move residents", 10, 13, 0.90),
            SpanScore("storm", 3, 4, 0.21),
        ]
    }

    result = TextGraphicalizer._best_spans(scores, {"event": ("storm",)})

    assert result["event"]["span"] == "storm"
    assert result["event"]["span_score"] == pytest.approx(0.21)


def test_span_grounding_prefers_concise_evidence_over_generic_context():
    scores = {
        "animal": [
            SpanScore("Fox saw some", 7, 10, 0.39),
            SpanScore("Fox", 7, 8, 0.326),
        ],
        "food": [
            SpanScore("Fox saw some", 7, 10, 0.408),
            SpanScore("Grapes", 4, 5, 0.322),
            SpanScore("Grapes", 74, 75, 0.312),
            SpanScore("bunches", 11, 12, 0.323),
        ],
    }

    result = TextGraphicalizer._best_spans(scores, stopwords={"some"})

    assert result["animal"]["span"] == "Fox"
    assert result["food"]["span"] == "Grapes"


def test_span_grounding_assigns_distinct_node_evidence():
    scores = {
        "first": [
            SpanScore("shared", 0, 1, 0.90),
            SpanScore("first-specific", 2, 3, 0.895),
        ],
        "second": [
            SpanScore("shared", 0, 1, 0.89),
            SpanScore("second-specific", 4, 5, 0.20),
        ],
    }

    result = TextGraphicalizer._best_spans(scores, unique=True)

    assert result["first"]["span"] == "first-specific"
    assert result["second"]["span"] == "shared"
    assert len({data["span"] for data in result.values()}) == len(result)


def test_relation_span_requires_a_relation_anchor():
    scores = {
        ("a", "b"): [
            SpanScore("repeated drought reduced", 0, 3, 0.90),
            SpanScore("reduced", 2, 3, 0.20),
        ]
    }

    assert TextGraphicalizer._best_spans(
        scores,
        {("a", "b"): ()},
        require_anchor=True,
    ) == {}
    result = TextGraphicalizer._best_spans(
        scores,
        {("a", "b"): ("reduce",)},
        require_anchor=True,
    )

    assert result[("a", "b")]["span"] == "reduced"


def test_weak_or_ambiguous_grounding_is_left_unattached():
    scores = {
        "node": [
            (0, "river", 0.21),
            (1, "shelter", 0.18),
        ],
        "clear_node": [
            (0, "storm", 0.9),
            (1, "river", 0.1),
        ],
    }

    result = TextGraphicalizer._best_nli_words(scores)

    assert "node" not in result
    assert result["clear_node"]["word"] == "storm"


def test_grounding_terms_can_resolve_a_weak_nli_match():
    scores = {"event": [(0, "shelter", 0.12), (1, "storm", 0.04)]}

    result = TextGraphicalizer._best_nli_words(scores, {"event": ("storm",)})

    assert result["event"]["word"] == "storm"


def test_grounding_loads_stopwords_from_external_yaml(monkeypatch, tmp_path):
    stopwords_path = tmp_path / "stopwords.yaml"
    stopwords_path.write_text("stopwords: [the, caused]\n", encoding="utf-8")
    estimator = fitted(monkeypatch, stopwords_path=stopwords_path)

    graph = estimator.transform("The infection caused a fever.")

    assert graph.graph["grounding_candidate_words"] == ["infection", "a", "fever"]
    assert graph.graph["stopwords_path"] == str(stopwords_path)


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
    monkeypatch.setattr(
        "textgraphicalizer.transformer.NliGroundingBackend.load",
        lambda self: FakeNliBackend(self.model_id, self.device),
    )
    estimator = TextGraphicalizer(ontology).load_model()
    graph = estimator.transform("A causes B.")
    assert set(graph.edges) == {("a", "b")}


def test_transform_requires_string(monkeypatch):
    estimator = fitted(monkeypatch)
    with pytest.raises(TypeError):
        estimator.transform(["not a document", 42])


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


def test_display_wraps_long_document_text(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimator = fitted(monkeypatch)
    graph = estimator.transform("A causes B.")
    document = "A very long document that should be wrapped across multiple lines."
    figure, axes = estimator.display(
        graph,
        document=document,
        max_char=20,
        show=False,
    )

    displayed = [text.get_text() for text in axes.texts if text.get_text().startswith("A very")]
    assert len(displayed) == 1
    assert "\n" in displayed[0]
    assert all(len(line) <= 20 for line in displayed[0].splitlines())
    plt.close(figure)


def test_display_d3_returns_force_directed_html(monkeypatch):
    pytest.importorskip("IPython")

    estimator = fitted(monkeypatch)
    monkeypatch.setattr(
        "textgraphicalizer.transformer._load_d3_source",
        lambda: "window.d3 = {};",
    )
    graph = nx.DiGraph()
    graph.add_node("a", label="Animal", span="Fox", probability=0.8)
    graph.add_node("b", label="Entity", probability=0.7)
    graph.add_edge("a", "b", relation_id="is_a", label="is a", word="is")

    rendered = estimator.display_d3(
        graph,
        title="Example",
        document="A short document.",
    )

    assert "d3@7" in rendered.data
    assert "window.d3 = {};" in rendered.data
    assert "const define = undefined" in rendered.data
    assert "unpkg.com/d3@7.9.0/dist/d3.min.js" in rendered.data
    assert "window.__textGraphicalizerD3PromiseV2 = null" in rendered.data
    assert "forceSimulation" in rendered.data
    assert "positionLinkLabels" in rendered.data
    assert "getComputedTextLength" in rendered.data
    assert 'attr("paint-order", "stroke")' in rendered.data
    assert '"#94a3b8"' in rendered.data
    assert '"#64748b"' not in rendered.data
    assert 'append("circle")' not in rendered.data
    assert "animal" in rendered.data
    assert "is a" not in rendered.data


def test_display_defaults_to_kamada_kawai(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimator = fitted(monkeypatch)
    graph = estimator.transform("A causes B.")
    calls = []

    def fake_kamada_kawai_layout(received_graph):
        calls.append(received_graph)
        return {"a": (0.0, 0.0), "b": (1.0, 1.0)}

    monkeypatch.setattr(
        "textgraphicalizer.transformer.nx.kamada_kawai_layout",
        fake_kamada_kawai_layout,
    )
    figure, _ = estimator.display(graph, show=False)

    assert calls == [graph]
    plt.close(figure)


def test_display_collapses_reciprocal_edges_with_the_same_label():
    graph = nx.DiGraph()
    graph.add_edge(
        "a",
        "b",
        label="supports",
        word="supports",
        probability=0.7,
    )
    graph.add_edge(
        "b",
        "a",
        label="supports",
        word="supports",
        probability=0.9,
    )

    undirected, directed = TextGraphicalizer._display_edges(graph)

    assert len(undirected) == 1
    assert directed == []
    assert undirected[0][2]["probability"] == 0.9


def test_display_labels_include_associated_words():
    assert TextGraphicalizer._node_display_label(
        "infection",
        {"label": "Infection", "word": "infection"},
        False,
    ) == "infection\ninfection"
    assert TextGraphicalizer._edge_display_label(
        {"label": "causes", "word": "caused"},
        False,
    ) == "causes\ncaused"


def test_display_hides_is_a_relation_label():
    assert TextGraphicalizer._edge_display_label(
        {"relation_id": "is_a", "label": "is a", "word": "is"},
        True,
    ) == ""
    assert TextGraphicalizer._edge_display_label(
        {"label": "is a"},
        False,
    ) == ""
    assert TextGraphicalizer._edge_display_label(
        {"label": "IS-A"},
        False,
    ) == ""


def test_display_does_not_render_is_a_edge_text(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimator = fitted(monkeypatch)
    graph = nx.DiGraph()
    graph.add_edge("a", "b", relation_id="is_a", label="is a", word="is")
    figure, axes = estimator.display(
        graph,
        layout="circular",
        show_node_labels=False,
        show_paragraph=False,
        show=False,
    )

    assert all(text.get_text() not in {"is", "is a", "is_a"} for text in axes.texts)
    plt.close(figure)


def test_display_node_labels_use_distinct_normal_fonts(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Annotation

    estimator = fitted(monkeypatch)
    graph = nx.DiGraph()
    graph.add_node("animal", label="Animal", paraphrase="a hungry fox")
    figure, axes = estimator.display(
        graph,
        layout="circular",
        show_edge_labels=False,
        show=False,
    )

    annotations = {
        annotation.get_text(): annotation
        for annotation in axes.get_children()
        if isinstance(annotation, Annotation)
    }
    assert annotations["animal"].get_fontfamily() == ["monospace"]
    assert annotations["animal"].get_fontweight() == "normal"
    assert annotations["a hungry fox"].get_fontfamily() == ["serif"]
    assert annotations["a hungry fox"].get_fontweight() == "normal"
    plt.close(figure)
