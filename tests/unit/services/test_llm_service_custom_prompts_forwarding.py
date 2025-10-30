from unittest.mock import AsyncMock, patch

import pytest

from inference_core.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_completion_forwards_prompt_name_to_chain():
    service = LLMService()
    with patch(
        "inference_core.services.llm_service.create_completion_chain"
    ) as mock_create:
        mock_chain = AsyncMock()
        mock_chain.model_name = "test-model"
        mock_chain.completion.return_value = "ok"
        mock_create.return_value = mock_chain

        await service.completion(prompt="hello", prompt_name="short_answer")

        assert mock_create.called
        _, kwargs = mock_create.call_args
        # prompt_name should be forwarded exactly
        assert kwargs.get("prompt_name") == "short_answer"


@pytest.mark.asyncio
async def test_chat_forwards_prompt_name_to_chain():
    service = LLMService()
    with patch("inference_core.services.llm_service.create_chat_chain") as mock_create:
        mock_chain = AsyncMock()
        mock_chain.model_name = "test-model"
        mock_chain.chat.return_value = "ok"
        mock_create.return_value = mock_chain

        await service.chat(session_id="s1", user_input="hi", prompt_name="tutor")

        assert mock_create.called
        _, kwargs = mock_create.call_args
        assert kwargs.get("prompt_name") == "tutor"
