from types import SimpleNamespace

import numpy as np
import torch

from textgraphicalizer.bert_backend import BertGroundingBackend


class WindowTokenizer:
    model_max_length = 6

    def __call__(
        self,
        value,
        *,
        return_offsets_mapping=False,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_tensors=None,
        truncation=False,
        max_length=None,
        stride=0,
        padding=False,
    ):
        del return_overflowing_tokens, truncation, stride, padding, max_length
        if isinstance(value, str):
            import re

            matches = list(re.finditer(r"[A-Za-z]+", value))
            pieces = []
            for match in matches:
                start, end = match.span()
                if match.group(0) == "storm":
                    middle = start + 2
                    pieces.extend([(start, middle), (middle, end)])
                else:
                    pieces.append((start, end))
            content_size = 4
            chunks = [pieces[start:start + content_size] for start in range(0, len(pieces), 2)]
            chunks = [chunk for chunk in chunks if chunk]
            input_ids = []
            attention = []
            offsets = []
            for chunk in chunks:
                ids = [101] + [index + 1 for index, _ in enumerate(chunk)] + [102]
                pad = 6 - len(ids)
                input_ids.append(ids + [0] * pad)
                attention.append([1] * len(ids) + [0] * pad)
                offsets.append([(0, 0)] + chunk + [(0, 0)] + [(0, 0)] * pad)
            result = {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention),
                "offset_mapping": torch.tensor(offsets),
            }
            if return_offsets_mapping:
                return result
            return result
        raise AssertionError("This test tokenizer only exercises paragraph encoding")


class HiddenStateModel:
    def __call__(self, **inputs):
        input_ids = inputs["input_ids"].float()
        hidden = torch.stack((input_ids, torch.ones_like(input_ids)), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_word_embeddings_map_wordpieces_across_overlapping_windows():
    backend = BertGroundingBackend("fake", window_stride=2)
    backend.tokenizer = WindowTokenizer()
    backend.model = HiddenStateModel()
    backend.device_ = torch.device("cpu")
    backend.max_length_ = 6

    result = backend._word_embeddings(
        "alpha storm beta gamma delta",
        [(0, "alpha"), (1, "storm"), (4, "delta")],
    )

    assert set(result) == {0, 1, 4}
    assert all(vector.shape == (2,) for vector in result.values())


def test_word_embeddings_skip_missing_candidate_tokens():
    backend = BertGroundingBackend("fake")
    backend.tokenizer = WindowTokenizer()
    backend.model = HiddenStateModel()
    backend.device_ = torch.device("cpu")
    backend.max_length_ = 6

    result = backend._word_embeddings("alpha beta", [(99, "missing")])

    assert result == {}


def test_score_words_chooses_highest_cosine_candidate(monkeypatch):
    backend = BertGroundingBackend("fake")
    monkeypatch.setattr(
        backend,
        "_word_embeddings",
        lambda text, words: {
            0: np.array([1.0, 0.0]),
            1: np.array([0.0, 1.0]),
        },
    )
    monkeypatch.setattr(
        backend,
        "_text_embeddings",
        lambda texts: np.array([[0.0, 1.0] for _ in texts]),
    )

    scores = backend.score_words(
        "alpha beta",
        [(0, "alpha"), (1, "beta")],
        {"node": "B concept"},
    )

    assert scores["node"][0][1] == "alpha"
    assert scores["node"][1][1] == "beta"
    assert scores["node"][1][2] > scores["node"][0][2]
