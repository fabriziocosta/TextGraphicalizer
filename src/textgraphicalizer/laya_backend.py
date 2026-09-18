"""Small adapter around the Laya package."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_LAYA_MODEL_REVISION = "7c76b622dfc5cac71b2dc1c29873efe2ce509a05"


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
        self.effective_model_revision: str | None = model_revision

    def load(self) -> "LayaBackend":
        if self.agent is not None:
            return self
        try:
            import laya
        except ImportError as exc:
            raise ImportError(
                "Laya is not installed. Install TextGraphicalizer with its project "
                "dependencies or run `pip install laya==0.1.6`."
            ) from exc

        model_source = self.model_path
        revision = self.model_revision
        if model_source is None and self.model_id == "convaiinnovations/laya" and revision is None:
            revision = DEFAULT_LAYA_MODEL_REVISION
        if revision is not None:
            from huggingface_hub import snapshot_download

            model_source = snapshot_download(
                repo_id=self.model_id,
                revision=revision,
                token=os.environ.get("HF_TOKEN"),
            )
            self.resolved_model_path = model_source
            self.effective_model_revision = revision
        device = None if self.device == "auto" else self.device
        self.agent = laya.load(model_source or self.model_id, device=device)
        if model_source is None:
            self.resolved_model_path = None
        return self

    def predict(self, state: str, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        if self.agent is None:
            self.load()
        return self.agent.predict(state, dict(questions))

    @staticmethod
    def _question_truncates(
        tokenizer: Any,
        state: str,
        question: Mapping[str, Any],
        *,
        max_len: int,
        head_max_len: int,
    ) -> bool:
        """Mirror Laya's sequence builder and detect state truncation.

        Laya reserves part of the sequence for the question head and answer
        options before placing the state text. Measuring the raw state alone
        therefore cannot tell us whether the actual model input was clipped.
        This deliberately mirrors ``laya.common.build_sequence`` for the
        pinned Laya version.
        """
        from laya.common import render_options, serialize_state

        question_type = question["type"]
        criteria = question.get("criteria")
        if question_type == "choice" and isinstance(criteria, list):
            criteria = {criterion: None for criterion in criteria}
        instructions = question["instructions"]
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions)
        internal_question = {"t": question_type, "ins": instructions, "crit": criteria}

        mask_token = tokenizer.mask_token
        options = render_options(internal_question)
        head_ids = tokenizer(
            "%s question: %s" % (question_type, instructions.replace(mask_token, " ")),
            add_special_tokens=False,
        )["input_ids"]
        option_ids = []
        for option in options:
            option_ids.append(
                [tokenizer.mask_token_id]
                + tokenizer(
                    " " + option.replace(mask_token, " "),
                    add_special_tokens=False,
                )["input_ids"][:48]
            )

        option_budget = head_max_len - sum(len(option) for option in option_ids)
        if option_budget < 16:
            per_option = max(4, (head_max_len - 16) // max(1, len(option_ids)))
            option_ids = [option[:per_option] for option in option_ids]
            option_budget = head_max_len - sum(len(option) for option in option_ids)
        head_ids = head_ids[: max(8, option_budget)]

        prefix_length = 1 + len(head_ids) + 1
        prefix_length += sum(len(option) for option in option_ids) + 1
        state_ids = tokenizer(
            serialize_state(state).replace(mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]
        return prefix_length + len(state_ids) + 1 > max_len

    def was_truncated(
        self,
        text: str,
        questions: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> bool | None:
        """Report whether Laya clipped the state in any supplied question.

        ``questions`` should contain the question definitions passed to Laya.
        If omitted, the method falls back to the raw-text check for backwards
        compatibility, but callers should provide questions for accurate
        detection.
        """
        if self.agent is None:
            return None
        tokenizer = getattr(self.agent, "tok", None)
        if tokenizer is None:
            return None
        config = getattr(self.agent, "cfg", {}) or {}
        max_len = int(config.get("max_len", 512))
        if questions is not None:
            head_max_len = int(config.get("head_max_len", 192))
            if not questions:
                return False
            try:
                return any(
                    self._question_truncates(
                        tokenizer,
                        text,
                        question,
                        max_len=max_len,
                        head_max_len=head_max_len,
                    )
                    for question in questions.values()
                )
            except Exception:
                return None
        try:
            token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:
            return None
        return token_count > max_len
