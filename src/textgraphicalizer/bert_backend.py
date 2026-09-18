"""BERT-based contextual word grounding."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class BertGroundingBackend:
    """Encode paragraph words and ontology descriptions in a shared BERT space."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        window_stride: int = 128,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.window_stride = window_stride
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.device_: Any | None = None
        self.max_length_: int | None = None

    def load(self) -> "BertGroundingBackend":
        if self.model is not None:
            return self
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BERT grounding requires torch and transformers to be installed."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        if self.device == "auto":
            if torch.cuda.is_available():
                selected_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                selected_device = "mps"
            else:
                selected_device = "cpu"
        else:
            selected_device = self.device
        self.device_ = torch.device(selected_device)
        self.model.to(self.device_)
        self.model.eval()

        configured_max_length = getattr(self.tokenizer, "model_max_length", 512)
        model_max_length = getattr(self.model.config, "max_position_embeddings", 512)
        if configured_max_length > 100_000:
            configured_max_length = model_max_length
        self.max_length_ = max(2, min(int(configured_max_length), int(model_max_length)))
        if self.max_length_ <= 2:
            raise ValueError("BERT grounding model must support at least two tokens")
        if not 0 <= self.window_stride < self.max_length_ - 1:
            raise ValueError(
                "window_stride must be non-negative and smaller than the BERT context window"
            )
        return self

    def _require_loaded(self) -> tuple[Any, Any, Any, int]:
        if self.model is None or self.tokenizer is None or self.device_ is None:
            self.load()
        if self.model is None or self.tokenizer is None or self.device_ is None:
            raise RuntimeError("BERT grounding model is not loaded")
        if self.max_length_ is None:
            raise RuntimeError("BERT grounding context length is not initialized")
        return self.model, self.tokenizer, self.device_, self.max_length_

    @staticmethod
    def _mean_pool(hidden: Any, attention_mask: Any, special_tokens_mask: Any | None = None) -> Any:
        mask = attention_mask.bool()
        if special_tokens_mask is not None:
            mask = mask & ~special_tokens_mask.bool()
            empty_rows = mask.sum(dim=1) == 0
            if bool(empty_rows.any()):
                mask = mask.clone()
                mask[empty_rows] = attention_mask[empty_rows].bool()
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _word_embeddings(
        self,
        text: str,
        words: Sequence[tuple[int, str]],
    ) -> dict[int, np.ndarray]:
        model, tokenizer, device, max_length = self._require_loaded()
        import torch

        matches = list(re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text))
        if not words:
            return {}
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            stride=min(self.window_stride, max_length - 2),
            padding=True,
        )
        offsets = encoded.pop("offset_mapping").tolist()
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.no_grad():
            hidden = model(**model_inputs).last_hidden_state.detach().cpu()
        attention = model_inputs["attention_mask"].detach().cpu()

        result: dict[int, np.ndarray] = {}
        for token_index, _ in words:
            if token_index >= len(matches):
                continue
            word_start, word_end = matches[token_index].span()
            vectors = []
            for window_index, window_offsets in enumerate(offsets):
                token_positions = [
                    position
                    for position, (start, end) in enumerate(window_offsets)
                    if end > start
                    and start < word_end
                    and end > word_start
                    and bool(attention[window_index, position])
                ]
                if token_positions:
                    vectors.append(hidden[window_index, token_positions].mean(dim=0))
            if vectors:
                result[token_index] = torch.stack(vectors).mean(dim=0).numpy()
        return result

    def _text_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        model, tokenizer, device, max_length = self._require_loaded()
        import torch

        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        encoded = tokenizer(
            list(texts),
            return_special_tokens_mask=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        special_tokens_mask = encoded.pop("special_tokens_mask").to(device)
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.no_grad():
            hidden = model(**model_inputs).last_hidden_state
            pooled = self._mean_pool(
                hidden,
                model_inputs["attention_mask"],
                special_tokens_mask,
            )
        return pooled.detach().cpu().numpy()

    def score_words(
        self,
        text: str,
        words: Sequence[tuple[int, str]],
        targets: Mapping[Any, str],
    ) -> dict[Any, list[tuple[int, str, float]]]:
        """Return cosine scores between candidate words and target descriptions."""
        if not words or not targets:
            return {target: [] for target in targets}
        word_vectors = self._word_embeddings(text, words)
        target_keys = list(targets)
        target_vectors = self._text_embeddings([targets[key] for key in target_keys])
        target_norms = np.linalg.norm(target_vectors, axis=1)

        result: dict[Any, list[tuple[int, str, float]]] = {}
        for target_index, target_key in enumerate(target_keys):
            target_vector = target_vectors[target_index]
            denominator = target_norms[target_index]
            candidates = []
            for token_index, word in words:
                word_vector = word_vectors.get(token_index)
                word_norm = np.linalg.norm(word_vector) if word_vector is not None else 0.0
                if word_vector is None or word_norm == 0.0 or denominator == 0.0:
                    continue
                score = float(np.dot(word_vector, target_vector) / (word_norm * denominator))
                candidates.append((token_index, word, score))
            result[target_key] = candidates
        return result
