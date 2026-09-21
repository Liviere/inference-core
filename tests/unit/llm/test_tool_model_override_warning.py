"""LLMConfig warns when tool_model_overrides references an undefined model.

An unknown model makes the after-tool switch a silent no-op at runtime, so the
mistake has to surface when the config is loaded.
"""

import logging
import textwrap

from inference_core.llm.config import LLMConfig

_YAML = """
providers:
  openai:
    name: OpenAI
    requires_api_key: true
    api_key_env: OPENAI_API_KEY
models:
  gpt-5-nano:
    provider: openai
agents:
  demo_agent:
    primary: gpt-5-nano
    tool_model_overrides:
      - tool_name: check_weather
        model: {model}
        trigger: after_tool
"""


def _load(tmp_path, monkeypatch, model: str) -> LLMConfig:
    path = tmp_path / "llm_config.yaml"
    path.write_text(textwrap.dedent(_YAML.format(model=model)), encoding="utf-8")
    monkeypatch.setenv("LLM_CONFIG_PATH", str(path))
    return LLMConfig()


def test_unknown_override_model_is_reported(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        _load(tmp_path, monkeypatch, "'gpt-oss:20b'")

    assert any(
        "tool_model_overrides" in r.getMessage() and "gpt-oss:20b" in r.getMessage()
        for r in caplog.records
    )


def test_known_override_model_is_silent(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _load(tmp_path, monkeypatch, "gpt-5-nano")

    assert cfg.agent_configs["demo_agent"].tool_model_overrides[0].model == "gpt-5-nano"
    assert not any("tool_model_overrides" in r.getMessage() for r in caplog.records)
