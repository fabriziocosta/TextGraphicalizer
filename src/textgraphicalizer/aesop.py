"""Load and clean the public-domain Aesop fables corpus."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError
from urllib.request import Request, urlopen

AESOP_SOURCE_URL = "https://www.gutenberg.org/ebooks/11339.txt.utf-8"
_CACHE_FORMAT_VERSION = 1
_END_MARKER = re.compile(
    r"^\s*\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*$", re.IGNORECASE | re.MULTILINE
)
_BOOK_TITLE = re.compile(r"^(?:ÆSOP|AESOP)'S FABLES\s*$", re.IGNORECASE | re.MULTILINE)
_STORY_TITLE = re.compile(r"^[A-Z0-9][A-Z0-9 ,.'’&()\-]+$")


def _default_cache_path() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home) if cache_home else Path.home() / ".cache"
    return root / "textgraphicalizer" / "aesop-fables-11339.json"


DEFAULT_AESOP_CACHE_PATH = _default_cache_path()


def _is_story_title(line: str) -> bool:
    line = line.strip()
    return (
        bool(line)
        and len(line) <= 120
        and bool(_STORY_TITLE.fullmatch(line))
        and any(character.isalpha() for character in line)
    )


def _book_body(source_text: str) -> str:
    source_text = source_text.replace("\r\n", "\n").replace("\r", "\n")
    source_text = source_text.lstrip("\ufeff")
    title_matches = list(_BOOK_TITLE.finditer(source_text))
    if title_matches:
        # Gutenberg repeats the book title in the contents/front matter. The
        # second occurrence is the title immediately before the fables.
        source_text = source_text[title_matches[min(1, len(title_matches) - 1)].end() :]

    illustrations = re.search(
        r"^\s*ILLUSTRATIONS\s*$", source_text, re.IGNORECASE | re.MULTILINE
    )
    if illustrations:
        source_text = source_text[: illustrations.start()]
    end_marker = _END_MARKER.search(source_text)
    if end_marker:
        source_text = source_text[: end_marker.start()]
    return source_text


def _normalize_story(title: str, lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in [title, *lines]:
        value = line.strip()
        if value:
            current.append(value)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def parse_aesop_fables(source_text: str) -> list[str]:
    """Return one cleaned string per fable from a Gutenberg text export."""
    if not isinstance(source_text, str):
        raise TypeError("source_text must be a string")

    stories: list[str] = []
    current_title: str | None = None
    current_lines: list[str] = []
    for line in _book_body(source_text).splitlines():
        stripped = line.strip()
        if _is_story_title(stripped):
            if current_title is not None:
                stories.append(_normalize_story(current_title, current_lines))
            current_title = stripped
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        stories.append(_normalize_story(current_title, current_lines))
    if not stories:
        raise ValueError("Could not find any Aesop fables in the source text")
    return stories


def _download_text(source_url: str) -> str:
    request = Request(source_url, headers={"User-Agent": "TextGraphicalizer/0.1"})
    try:
        with urlopen(request, timeout=30) as response:
            return cast(bytes, response.read()).decode("utf-8-sig")
    except (OSError, UnicodeDecodeError, URLError) as exc:
        raise RuntimeError(f"Could not download Aesop fables from {source_url}") from exc


def _read_cache(cache_path: Path, source_url: str) -> list[str] | None:
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload: Any = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read Aesop cache {cache_path}; use refresh=True to rebuild it"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Aesop cache {cache_path} must contain a JSON object")
    if payload.get("format_version") != _CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported Aesop cache format in {cache_path}")
    if payload.get("source_url") != source_url:
        return None
    stories = payload.get("stories")
    if (
        not isinstance(stories, list)
        or not stories
        or not all(isinstance(story, str) and story.strip() for story in stories)
    ):
        raise ValueError(f"Aesop cache {cache_path} contains invalid stories")
    return stories


def _write_cache(cache_path: Path, source_url: str, stories: list[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": _CACHE_FORMAT_VERSION,
        "source_url": source_url,
        "stories": stories,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_path.replace(cache_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_aesop_fables(
    cache_path: str | Path | None = None,
    *,
    source_url: str = AESOP_SOURCE_URL,
    refresh: bool = False,
) -> list[str]:
    """Load one cleaned string per Aesop fable, using a disk cache when possible.

    The default cache is ``~/.cache/textgraphicalizer/aesop-fables-11339.json``
    (or ``$XDG_CACHE_HOME/textgraphicalizer/...``). Set ``refresh=True`` to
    download and parse the public source again.
    """
    path = Path(cache_path) if cache_path is not None else DEFAULT_AESOP_CACHE_PATH
    if path.exists() and not refresh:
        cached = _read_cache(path, source_url)
        if cached is not None:
            return cached

    stories = parse_aesop_fables(_download_text(source_url))
    _write_cache(path, source_url, stories)
    return stories
