import torch

from textgraphicalizer.nli_backend import NliGroundingBackend


def test_candidate_windows_keep_the_window_containing_each_token():
    backend = NliGroundingBackend("fake")
    backend._premise_windows = lambda text: [
        ("alpha storm", 0, 11),
        ("beta harvest", 12, 24),
    ]

    result = backend._candidate_windows(
        "alpha storm beta harvest",
        [(1, "storm"), (2, "beta")],
    )

    assert result[1] == ["alpha storm"]
    assert result[2] == ["beta harvest"]


def test_candidate_contexts_are_local_to_each_word():
    backend = NliGroundingBackend("fake")
    text = "zero one two three four five six seven eight nine ten eleven twelve"

    result = backend._candidate_contexts(text, [(6, "six")])

    context, relative_index = result[6][0]
    assert context == "one two three four five six seven eight nine ten eleven"
    assert relative_index == 5


class PairTokenizer:
    def __call__(self, premises, hypotheses, **kwargs):
        del premises, kwargs
        ids = [[2 if "infection" in hypothesis else 0, 1] for hypothesis in hypotheses]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.ones(len(ids), 2, dtype=torch.long),
        }


class PairModel:
    def __call__(self, **inputs):
        input_ids = inputs["input_ids"]
        logits = torch.zeros(input_ids.shape[0], 3)
        logits[:, 1] = input_ids[:, 0].float()
        return type("Output", (), {"logits": logits})()


def test_score_words_uses_entailment_probability():
    backend = NliGroundingBackend("fake")
    backend.tokenizer = PairTokenizer()
    backend.model = PairModel()
    backend.device_ = torch.device("cpu")
    backend.max_length_ = 512
    backend.entailment_index_ = 1
    backend._candidate_windows = lambda text, words: {
        token_index: [text] for token_index, _ in words
    }

    scores = backend.score_words(
        "The infection caused fever.",
        [(0, "infection"), (1, "fever")],
        {"node": 'The word "{word}" refers to Infection.'},
    )

    assert scores["node"][0][1] == "infection"
    assert scores["node"][0][2] > scores["node"][1][2]


def test_contrastive_scoring_prefers_words_that_change_entailment(monkeypatch):
    backend = NliGroundingBackend("fake")
    monkeypatch.setattr(
        backend,
        "score_words",
        lambda text, words, targets: {
            "node": [(0, "repeated", 0.9), (1, "drought", 0.8)]
        },
    )
    monkeypatch.setattr(
        backend,
        "_premise_windows",
        lambda text: [(text, 0, len(text))],
    )
    monkeypatch.setattr(
        backend,
        "_score_pairs",
        lambda pairs: {
            "node": {
                0: ("repeated", 0.8),
                1: ("drought", 0.1),
            }
        },
    )

    scores = backend.score_words_contrastive(
        "repeated drought",
        [(0, "repeated"), (1, "drought")],
        {"node": 'The word "{word}" refers to the concept.'},
    )

    assert scores["node"][1][1] == "drought"
    assert scores["node"][1][2] > scores["node"][0][2]
