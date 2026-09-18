"""Quiet Hugging Face model loading without hiding actual failures."""

from __future__ import annotations


def silence_model_download_output() -> None:
    """Hide Hub warnings, Transformers load reports, and progress bars."""
    try:
        from huggingface_hub import logging as hub_logging
        from huggingface_hub.utils import disable_progress_bars

        hub_logging.set_verbosity_error()
        disable_progress_bars()
    except ImportError:
        pass

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except ImportError:
        pass
