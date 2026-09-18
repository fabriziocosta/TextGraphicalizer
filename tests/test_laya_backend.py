from types import SimpleNamespace

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
