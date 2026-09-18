import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

from textgraphicalizer import LayaInferenceError, LayaResponseError
from textgraphicalizer.laya_backend import LayaBackend


class WordTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": text.split()}


def backend_with_limits(max_len=512, head_max_len=192):
    backend = LayaBackend("test-model")
    backend.agent = SimpleNamespace(
        tok=WordTokenizer(),
        cfg={"max_len": max_len, "head_max_len": head_max_len},
    )
    return backend


def test_load_uses_cached_snapshot_without_online_download(monkeypatch):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/cached/laya"

    fake_huggingface_hub = ModuleType("huggingface_hub")
    fake_huggingface_hub.snapshot_download = snapshot_download
    fake_laya = ModuleType("laya")
    fake_laya.load = lambda source, device: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)
    monkeypatch.setitem(sys.modules, "laya", fake_laya)

    LayaBackend(
        "convaiinnovations/laya",
        model_revision="revision",
    ).load()

    assert calls == [
        {
            "repo_id": "convaiinnovations/laya",
            "revision": "revision",
            "local_files_only": True,
        }
    ]


def test_load_downloads_when_snapshot_is_not_cached(monkeypatch):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        return "/downloaded/laya"

    fake_huggingface_hub = ModuleType("huggingface_hub")
    fake_huggingface_hub.snapshot_download = snapshot_download
    fake_laya = ModuleType("laya")
    fake_laya.load = lambda source, device: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)
    monkeypatch.setitem(sys.modules, "laya", fake_laya)

    LayaBackend(
        "convaiinnovations/laya",
        model_revision="revision",
    ).load()

    assert calls == [
        {
            "repo_id": "convaiinnovations/laya",
            "revision": "revision",
            "local_files_only": True,
        },
        {
            "repo_id": "convaiinnovations/laya",
            "revision": "revision",
            "token": None,
        },
    ]


def test_truncation_accounts_for_question_overhead():
    backend = backend_with_limits(max_len=10)
    question = {
        "type": "noul",
        "instructions": "Is this concept expressed?",
    }

    assert backend.was_truncated("short text", {"node_0": question}) is True


def test_truncation_is_false_when_full_sequence_fits():
    backend = backend_with_limits(max_len=100)
    question = {
        "type": "noul",
        "instructions": "Is this concept expressed?",
    }

    assert backend.was_truncated("short text", {"node_0": question}) is False


def test_truncation_checks_all_question_types():
    backend = backend_with_limits(max_len=20)
    questions = {
        "node_0": {
            "type": "noul",
            "instructions": "Short?",
        },
        "edge_0": {
            "type": "choice",
            "instructions": "Which relation?",
            "criteria": {
                "causes": "The source produces the target.",
                "no_relation": "No relation.",
            },
        },
    }

    assert backend.was_truncated("a tiny state", questions) is True


def test_truncation_returns_none_when_question_serialization_fails():
    backend = backend_with_limits()
    question = {"type": "unsupported"}

    assert backend.was_truncated("text", {"bad": question}) is None


def test_predict_rejects_malformed_noul_response():
    backend = backend_with_limits()
    backend.agent.predict = lambda state, questions: {
        "answers": {
            "node_0": {"type": "noul", "noul": "not-a-number"},
        }
    }
    question = {"type": "noul", "instructions": "Is it present?"}

    with pytest.raises(LayaResponseError, match="numeric probability"):
        backend.predict("text", {"node_0": question})


def test_predict_wraps_raw_inference_failure():
    backend = backend_with_limits()

    def fail(state, questions):
        raise ValueError("model exploded")

    backend.agent.predict = fail
    question = {"type": "noul", "instructions": "Is it present?"}

    with pytest.raises(LayaInferenceError, match="inference failed"):
        backend.predict("text", {"node_0": question})
