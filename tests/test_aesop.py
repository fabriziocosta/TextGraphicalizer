import json

import pytest

import textgraphicalizer.aesop as aesop
from textgraphicalizer import load_aesop_fables, parse_aesop_fables

SOURCE_TEXT = """\
The Project Gutenberg eBook of Aesop's Fables

INTRODUCTION
This is header matter.

AESOP'S FABLES

CONTENTS
THE FOX AND THE GRAPES
THE GOOSE THAT LAID THE GOLDEN EGGS

AESOP'S FABLES

THE FOX AND THE GRAPES

A hungry Fox saw some Grapes hanging high above him, but he could not reach
them. He walked away saying they were sour.

    It is easy to despise what you cannot get.


THE GOOSE THAT LAID THE GOLDEN EGGS

A Man owned a Goose which laid a Golden Egg every day.


ILLUSTRATIONS

[Illustration: THE FOX AND THE GRAPES]

*** END OF THE PROJECT GUTENBERG EBOOK AESOP'S FABLES ***
"""


def test_parse_aesop_fables_removes_header_and_splits_stories():
    stories = parse_aesop_fables(SOURCE_TEXT)

    assert len(stories) == 2
    assert stories[0].startswith("THE FOX AND THE GRAPES\n\nA hungry Fox")
    assert "INTRODUCTION" not in stories[0]
    assert "CONTENTS" not in stories[0]
    assert "ILLUSTRATIONS" not in stories[-1]
    assert "sour.\n\nIt is easy" in stories[0]
    assert stories[1].startswith("THE GOOSE THAT LAID THE GOLDEN EGGS\n\nA Man")


def test_load_aesop_fables_caches_cleaned_stories(monkeypatch, tmp_path):
    calls = []

    def download(source_url):
        calls.append(source_url)
        return SOURCE_TEXT

    monkeypatch.setattr(aesop, "_download_text", download)
    cache_path = tmp_path / "nested" / "aesop.json"
    source_url = "https://example.test/aesop.txt"

    first = load_aesop_fables(cache_path, source_url=source_url)
    second = load_aesop_fables(cache_path, source_url=source_url)

    assert first == second
    assert calls == [source_url]
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["stories"] == first
    assert "INTRODUCTION" not in payload["stories"][0]


def test_load_aesop_fables_refreshes_cache(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        aesop,
        "_download_text",
        lambda source_url: calls.append(source_url) or SOURCE_TEXT,
    )
    cache_path = tmp_path / "aesop.json"

    load_aesop_fables(cache_path)
    load_aesop_fables(cache_path, refresh=True)

    assert len(calls) == 2


def test_parse_aesop_fables_rejects_empty_source():
    with pytest.raises(ValueError, match="Could not find any Aesop fables"):
        parse_aesop_fables("no stories here")
