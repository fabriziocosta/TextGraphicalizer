from textgraphicalizer.span_backend import (
    ConceptDescription,
    Span,
    SpanGroundingBackend,
)


def test_generate_candidates_preserves_word_positions_and_supports_three_tokens():
    text = "The severe drought caused widespread crop failure."

    candidates = SpanGroundingBackend.generate_candidates(text)

    assert Span("severe drought", 1, 3) in candidates
    assert Span("crop failure", 5, 7) in candidates
    assert Span("widespread crop failure", 4, 7) in candidates


def test_candidate_context_marks_only_the_selected_span():
    text = "They sat on the bank of the river."
    span = Span("bank", 4, 5)

    assert SpanGroundingBackend.candidate_context(text, span) == (
        "Sentence: They sat on the <<bank>> of the river.\n"
        "Candidate expression: bank"
    )


class FakeCrossEncoder:
    def __init__(self):
        self.pairs = None

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        del kwargs
        return [index / 10 for index in range(len(pairs))]


def test_score_spans_batches_concept_queries_and_returns_all_scores():
    backend = SpanGroundingBackend("fake")
    backend.model = FakeCrossEncoder()
    candidates = [Span("infection", 1, 2), Span("fever", 5, 6)]
    concepts = {
        "infection": ConceptDescription(
            "Infection", "A disease-causing invasion by pathogens."
        ),
        "fever": ConceptDescription("Fever", "An elevated body temperature."),
    }

    scores = backend.score_spans("The infection caused a high fever.", candidates, concepts)

    assert len(scores["infection"]) == 2
    assert scores["infection"][0].text == "infection"
    assert scores["fever"][1].score == 0.3
    model = backend.model
    assert model is not None
    assert model.pairs[0] == (
        "Concept: Infection\nDescription: A disease-causing invasion by pathogens.",
        "Sentence: The <<infection>> caused a high fever.\n"
        "Candidate expression: infection",
    )
