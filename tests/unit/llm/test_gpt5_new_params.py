from unittest.mock import MagicMock, patch

import pytest

from app.llm.config import ModelConfig, ModelProvider
from app.llm.models import LLMModelFactory


@pytest.fixture
def factory():
    mock_config = MagicMock()
    mock_config.enable_caching = False
    # minimal models map for availability checks not needed here
    return LLMModelFactory(mock_config)


@patch("app.llm.models.normalize_params")
@patch("app.llm.models.ChatOpenAI")
def test_gpt5_reasoning_params_pass_through(mock_chat_openai, mock_normalize, factory):
    config = ModelConfig(
        name="gpt-5",
        provider=ModelProvider.OPENAI,
        api_key="key",
        temperature=0.7,
        max_tokens=1000,
    )
    mock_normalize.return_value = {"reasoning_effort": "low", "medium": "high"}
    mock_chat_openai.return_value = MagicMock()

    model = factory._create_model_instance(
        config,
        reasoning_effort="high",
        verbosity="high",
    )

    # raw params should include new keys and legacy ones (legacy dropped later by policy replace)
    mock_normalize.assert_called_once()
    args, kwargs = mock_normalize.call_args
    assert kwargs["model_name"] == "gpt-5"
    passed_raw = args[1]
    assert passed_raw["reasoning_effort"] == "high"
    assert passed_raw["verbosity"] == "high"

    assert model is not None


def test_gpt5_legacy_param_rejected_in_service_explain(monkeypatch):
    # Use real service to trigger ValueError when legacy param used with gpt-5
    from app.services.llm_service import LLMService

    svc = LLMService()
    # Patch chain factory to avoid hitting real LLM
    from app.llm import chains

    class DummyChain:
        model_name = "gpt-5"

        async def generate_story(self, **_):
            return "ok"

    monkeypatch.setattr(
        chains, "create_explanation_chain", lambda model_name=None, **p: DummyChain()
    )

    with pytest.raises(ValueError):
        import asyncio

        asyncio.run(
            svc.explain(
                question="Q?",
                model_name="gpt-5",
                temperature=0.5,  # legacy should raise
            )
        )
