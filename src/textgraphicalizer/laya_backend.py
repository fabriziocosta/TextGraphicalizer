"""Small adapter around the Laya package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


class LayaBackend:
    """Load one Laya agent and expose the operations needed by the transformer."""

    def __init__(
        self,
        model_id: str,
        model_path: str | os.PathLike[str] | None = None,
        model_revision: str | None = None,
        device: str = "auto",
    ) -> None:
        if model_path is not None and model_revision is not None:
            raise ValueError("model_path and model_revision cannot be combined")
        self.model_id = model_id
        self.model_path = str(model_path) if model_path is not None else None
        self.model_revision = model_revision
        self.device = device
        self.agent: Any | None = None
        self.resolved_model_path: str | None = self.model_path

    def load(self) -> "LayaBackend":
        try:
            import laya
        except ImportError as exc:
            raise ImportError(
                "Laya is not installed. Install TextGraphicalizer with its project "
                "dependencies or run `pip install laya==0.1.6`."
            ) from exc

        model_source = self.model_path
        if self.model_revision is not None:
            from huggingface_hub import snapshot_download

            model_source = snapshot_download(
                repo_id=self.model_id,
                revision=self.model_revision,
                token=os.environ.get("HF_TOKEN"),
            )
            self.resolved_model_path = model_source
        device = None if self.device == "auto" else self.device
        self.agent = laya.load(model_source or self.model_id, device=device)
        if model_source is None:
            self.resolved_model_path = None
        return self

    def predict(self, state: str, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        if self.agent is None:
            self.load()
        return self.agent.predict(state, dict(questions))

    def was_truncated(self, text: str) -> bool | None:
        if self.agent is None:
            return None
        tokenizer = getattr(self.agent, "tok", None)
        if tokenizer is None:
            return None
        config = getattr(self.agent, "cfg", {}) or {}
        max_len = int(config.get("max_len", 512))
        try:
            token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:
            return None
        return token_count > max_len
