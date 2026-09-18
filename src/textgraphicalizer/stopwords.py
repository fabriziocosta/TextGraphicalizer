"""Load the external stopword list used for word grounding."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_STOPWORDS_PATH = Path(__file__).resolve().parents[2] / "stopwords.yaml"


def load_stopwords(source: str | Path | None = None) -> frozenset[str]:
    """Load and normalize stopwords from a YAML file.

    The file may contain either a top-level list or a mapping with a
    ``stopwords`` list.
    """
    path = DEFAULT_STOPWORDS_PATH if source is None else Path(source)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data: Any = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read stopwords file {path}: {exc}") from exc

    values: Any = data.get("stopwords") if isinstance(data, Mapping) else data
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
    ):
        raise ValueError(
            f"Stopwords file {path} must contain a non-empty 'stopwords' list"
        )

    stopwords: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Stopwords file {path} contains an invalid word at index {index}"
            )
        stopwords.add(value.strip().casefold())
    return frozenset(stopwords)
