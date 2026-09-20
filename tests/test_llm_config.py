import pytest

from textgraphicalizer import load_llm_config


def test_load_llm_config_validates_and_preserves_provider_metadata(tmp_path):
    path = tmp_path / "llm.yaml"
    path.write_text(
        """version: 1
providers:
  mlx-lm:
    model: test-model
    base_url: http://127.0.0.1:8080/v1
    model_path: /models/test-model
    server:
      host: 127.0.0.1
      port: 8080
      log_level: INFO
""",
        encoding="utf-8",
    )

    config = load_llm_config(path)

    assert config.provider("mlx-lm")["model"] == "test-model"
    assert config.provider("mlx-lm")["model_path"] == "/models/test-model"
    assert config.provider("mlx-lm")["server"]["port"] == 8080
    assert config.fingerprint


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("[]", "root"),
        ("version: 2\nproviders: {}\n", "version"),
        ("version: 1\nproviders: []\n", "providers"),
        ("version: 1\nproviders:\n  unknown: {}\n", "providers.unknown"),
        (
            "version: 1\nproviders:\n  mlx-lm:\n    temperature: 3\n",
            "temperature",
        ),
        (
            "version: 1\nproviders:\n  mlx-lm:\n    server:\n      port: 70000\n",
            "port",
        ),
    ],
)
def test_load_llm_config_reports_invalid_fields(tmp_path, contents, message):
    path = tmp_path / "invalid.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_llm_config(path)
