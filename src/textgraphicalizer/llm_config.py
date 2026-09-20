"""Versioned YAML configuration for provider-backed LLM grounding."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

LLM_CONFIG_VERSION = 1
SUPPORTED_LLM_PROVIDERS = frozenset({"openai", "ollama", "mlx-lm"})

_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {"model": "gpt-4.1-mini", "api_key_env": "OPENAI_API_KEY"},
    "ollama": {
        "model": "gemma4:12b-mlx",
        "base_url": "http://localhost:11434",
        "timeout": 600.0,
    },
    "mlx-lm": {
        "model": "GLM-4.7-Flash-4bit",
        "base_url": "http://127.0.0.1:8080/v1",
        "timeout": 600.0,
        "temperature": 0.0,
        "max_tokens": 128,
        "model_path": (
            "/Users/f.costa/Documents/Codex/2026-09-19/"
            "referenced-chatgpt-conversation-this-is-an/models/GLM-4.7-Flash-4bit"
        ),
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "log_level": "INFO",
        },
    },
}

_ALLOWED_FIELDS = frozenset(
    {
        "model",
        "api_key_env",
        "base_url",
        "timeout",
        "temperature",
        "max_tokens",
        "model_path",
        "weights_path",
        "server",
        "metadata",
        "auto_start",
        "python_executable",
        "chat_template_kwargs",
    }
)


@dataclass(frozen=True)
class LLMConfig:
    """Validated provider configuration loaded from one YAML file."""

    version: int
    providers: Mapping[str, Mapping[str, Any]]
    source_path: Path
    fingerprint: str

    def provider(self, name: str) -> Mapping[str, Any]:
        """Return one provider section, or an empty section when absent."""
        return self.providers.get(name, MappingProxyType({}))


def _error(path: Path, field: str, message: str) -> ValueError:
    return ValueError(f"Invalid LLM configuration {path} ({field}): {message}")


def _validate_text(value: Any, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, field, "must be a non-empty string")
    return value.strip()


def _validate_number(value: Any, path: Path, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, field, "must be a number")
    return float(value)


def _validate_provider(name: str, raw: Any, path: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _error(path, f"providers.{name}", "must be a mapping")

    unknown = set(raw) - _ALLOWED_FIELDS
    if unknown:
        raise _error(
            path,
            f"providers.{name}",
            f"contains unsupported fields: {sorted(str(item) for item in unknown)}",
        )

    result: dict[str, Any] = {}
    for field, value in raw.items():
        field_name = f"providers.{name}.{field}"
        if field in {"model", "api_key_env", "base_url", "model_path", "weights_path"}:
            result[field] = _validate_text(value, path, field_name)
        elif field == "timeout":
            timeout = _validate_number(value, path, field_name)
            if timeout <= 0:
                raise _error(path, field_name, "must be positive")
            result[field] = timeout
        elif field == "temperature":
            temperature = _validate_number(value, path, field_name)
            if not 0.0 <= temperature <= 2.0:
                raise _error(path, field_name, "must be between 0 and 2")
            result[field] = temperature
        elif field == "max_tokens":
            if isinstance(value, bool) or not isinstance(value, int):
                raise _error(path, field_name, "must be an integer")
            if not 1 <= value <= 32768:
                raise _error(path, field_name, "must be between 1 and 32768")
            result[field] = value
        elif field == "server":
            if not isinstance(value, Mapping):
                raise _error(path, field_name, "must be a mapping")
            server = dict(value)
            unknown_server = set(server) - {
                "host",
                "port",
                "log_level",
            }
            if unknown_server:
                raise _error(
                    path,
                    field_name,
                    f"contains unsupported fields: {sorted(str(item) for item in unknown_server)}",
                )
            if "host" in server:
                server["host"] = _validate_text(server["host"], path, f"{field_name}.host")
            if "port" in server:
                port = server["port"]
                if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                    raise _error(path, f"{field_name}.port", "must be an integer from 1 to 65535")
            if "log_level" in server:
                server["log_level"] = _validate_text(
                    server["log_level"], path, f"{field_name}.log_level"
                )
            result[field] = server
        elif field == "metadata":
            if not isinstance(value, Mapping):
                raise _error(path, field_name, "must be a mapping")
            result[field] = dict(value)
        elif field == "auto_start":
            if not isinstance(value, bool):
                raise _error(path, field_name, "must be a boolean")
            result[field] = value
        elif field == "python_executable":
            result[field] = _validate_text(value, path, field_name)
        elif field == "chat_template_kwargs":
            if not isinstance(value, Mapping):
                raise _error(path, field_name, "must be a mapping")
            result[field] = dict(value)

    return result


def load_llm_config(source: str | Path) -> LLMConfig:
    """Load and validate a versioned provider YAML configuration."""
    path = Path(source)
    try:
        raw_bytes = path.read_bytes()
        data: Any = yaml.safe_load(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read LLM configuration {path}: {exc}") from exc

    if not isinstance(data, Mapping):
        raise _error(path, "root", "must be a mapping")
    if data.get("version") != LLM_CONFIG_VERSION:
        raise _error(path, "version", f"must be {LLM_CONFIG_VERSION}")

    raw_providers = data.get("providers")
    if not isinstance(raw_providers, Mapping):
        raise _error(path, "providers", "must be a mapping")

    providers: dict[str, Mapping[str, Any]] = {}
    for name, raw_provider in raw_providers.items():
        if name not in SUPPORTED_LLM_PROVIDERS:
            raise _error(path, f"providers.{name}", "is not a supported LLM provider")
        providers[name] = MappingProxyType(_validate_provider(name, raw_provider, path))

    fingerprint = hashlib.sha256(raw_bytes).hexdigest()
    return LLMConfig(
        version=LLM_CONFIG_VERSION,
        providers=MappingProxyType(providers),
        source_path=path,
        fingerprint=fingerprint,
    )


def default_provider_settings(provider: str) -> Mapping[str, Any]:
    """Return immutable defaults for one supported provider."""
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return MappingProxyType(dict(_DEFAULTS[provider]))


def provider_settings_for_config(
    config: LLMConfig | None,
    provider: str,
) -> Mapping[str, Any]:
    """Return YAML settings, or provider defaults when its section is absent."""
    if config is None:
        return default_provider_settings(provider)
    return config.provider(provider)


def config_fingerprint(config: LLMConfig | None) -> str | None:
    """Return a stable fingerprint suitable for estimator cache signatures."""
    if config is None:
        return None
    return config.fingerprint
