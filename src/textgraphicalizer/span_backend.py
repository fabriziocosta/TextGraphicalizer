"""Cross-encoder backend for grounding ontology concepts in text spans."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .hf_quiet import silence_model_download_output

_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


@dataclass(frozen=True)
class ConceptDescription:
    """The ontology information presented to the grounding cross-encoder."""

    label: str
    description: str


@dataclass(frozen=True)
class Span:
    """A contiguous word span in a document.

    ``start_word`` is inclusive and ``end_word`` is exclusive, using the
    document's word-token positions.
    """

    text: str
    start_word: int
    end_word: int


@dataclass(frozen=True)
class SpanScore:
    """A candidate span and its cross-encoder compatibility score."""

    text: str
    start_word: int
    end_word: int
    score: float


class SpanGroundingBackend:
    """Score ontology concepts against marked candidate spans in context."""

    def __init__(
        self,
        model_id: str = "cross-encoder/stsb-distilroberta-base",
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.model: Any | None = None

    def load(self) -> "SpanGroundingBackend":
        """Load the Sentence Transformers cross-encoder lazily."""
        if self.model is not None:
            return self
        silence_model_download_output()
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Span grounding requires sentence-transformers to be installed."
            ) from exc

        # ``None`` lets Sentence Transformers select the best available device.
        selected_device = None if self.device == "auto" else self.device
        self.model = CrossEncoder(self.model_id, device=selected_device)
        return self

    @staticmethod
    def generate_candidates(text: str) -> list[Span]:
        """Generate all contiguous one-, two-, and three-word spans."""
        matches = list(_WORD_RE.finditer(text))
        candidates: list[Span] = []
        for start in range(len(matches)):
            for length in (1, 2, 3):
                end = start + length
                if end > len(matches):
                    break
                candidates.append(
                    Span(
                        text=text[matches[start].start():matches[end - 1].end()],
                        start_word=start,
                        end_word=end,
                    )
                )
        return candidates

    @staticmethod
    def mark_candidate(text: str, span: Span) -> str:
        """Return ``text`` with the selected span delimited for the model."""
        matches = list(_WORD_RE.finditer(text))
        if not 0 <= span.start_word < span.end_word <= len(matches):
            raise ValueError(f"Span word bounds are invalid: {span}")
        start = matches[span.start_word].start()
        end = matches[span.end_word - 1].end()
        return f"{text[:start]}<<{text[start:end]}>>{text[end:]}"

    @classmethod
    def candidate_context(cls, text: str, span: Span) -> str:
        marked = cls.mark_candidate(text, span)
        return f"Sentence: {marked}\nCandidate expression: {span.text}"

    @staticmethod
    def _concept_text(concept: ConceptDescription | Mapping[str, Any] | Any) -> str:
        if isinstance(concept, ConceptDescription):
            label, description = concept.label, concept.description
        elif isinstance(concept, Mapping):
            label, description = concept["label"], concept["description"]
        else:
            label = getattr(concept, "label")
            description = getattr(concept, "description")
        return f"Concept: {label}\nDescription: {description}"

    @staticmethod
    def _scalar_scores(values: Any) -> list[float]:
        if hasattr(values, "tolist"):
            values = values.tolist()
        result: list[float] = []
        for value in values:
            while isinstance(value, (list, tuple)):
                if not value:
                    raise ValueError("Cross-encoder returned an empty score")
                value = value[0]
            result.append(float(value))
        return result

    def score_spans(
        self,
        text: str,
        candidates: Sequence[Span],
        concepts: Mapping[Any, ConceptDescription | Mapping[str, Any] | Any],
    ) -> dict[Any, list[SpanScore]]:
        """Score every concept/candidate pair in one batched model call."""
        result: dict[Any, list[SpanScore]] = {key: [] for key in concepts}
        if not candidates or not concepts:
            return result
        if self.model is None:
            self.load()
        if self.model is None:
            raise RuntimeError("Span grounding model is not loaded")

        pairs: list[tuple[str, str]] = []
        pair_keys: list[Any] = []
        for key, concept in concepts.items():
            concept_text = self._concept_text(concept)
            for candidate in candidates:
                pairs.append((concept_text, self.candidate_context(text, candidate)))
                pair_keys.append(key)

        try:
            predicted = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except TypeError:
            # Small test doubles and older Sentence Transformers versions may
            # only accept the pair list.
            predicted = self.model.predict(pairs)
        scores = self._scalar_scores(predicted)
        if len(scores) != len(pairs):
            raise ValueError(
                "Span grounding model returned a different number of scores "
                f"({len(scores)}) than input pairs ({len(pairs)})"
            )
        for key, candidate, score in zip(
            pair_keys,
            (candidate for _ in concepts for candidate in candidates),
            scores,
        ):
            result[key].append(
                SpanScore(
                    text=candidate.text,
                    start_word=candidate.start_word,
                    end_word=candidate.end_word,
                    score=score,
                )
            )
        return result
