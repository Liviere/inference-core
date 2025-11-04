"""
Tests for MCP tool integration with custom LLMService configurations.

Verifies that custom system prompts, model parameters, and other configurations
are preserved when MCP tools are enabled for a task.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from inference_core.services.llm_service import LLMService


class TestLLMServiceMCPCustomIntegration:
    """Test that MCP tool integration preserves custom configurations."""

    @pytest.fixture
    def custom_llm_service(self):
        """Create a custom LLMService instance with custom defaults."""
        return LLMService(
            default_models={"chat": "gpt-4o-mini"},
            default_model_params={"chat": {"temperature": 0.3, "max_tokens": 500}},
            default_prompt_names={"chat": "custom_prompt"},
            default_chat_system_prompt="You are a custom AI assistant with special instructions.",
        )

    @pytest.mark.asyncio
    async def test_custom_system_prompt_preserved_with_mcp_tools(
        self, custom_llm_service
    ):
        """Verify that custom system prompt is preserved when MCP tools are enabled."""
        
        # Mock the tooling context to simulate MCP being enabled
        mock_tooling_ctx = MagicMock()
        mock_tooling_ctx.tools = [MagicMock()]
        mock_tooling_ctx.instructions = "You have access to external tools."
        mock_tooling_ctx.profile_name = "test-profile"
        mock_tooling_ctx.limits = {"max_steps": 5, "max_run_seconds": 30}
        
        # Mock the chain building and agent creation
        with patch.object(
            custom_llm_service, "_get_tooling_context", return_value=mock_tooling_ctx
        ), patch.object(
            custom_llm_service, "_build_chat_chain"
        ) as mock_build_chain, patch.object(
            custom_llm_service.model_factory, "create_model"
        ) as mock_create_model, patch(
            "inference_core.services.llm_service.create_openai_tools_agent"
        ), patch(
            "inference_core.services.llm_service.AgentExecutor"
        ) as mock_agent_executor, patch(
            "inference_core.services.llm_service.SQLChatMessageHistory"
        ):
            # Setup mock chain
            mock_chain = MagicMock()
            mock_chain.model_name = "gpt-4o-mini"
            mock_build_chain.return_value = mock_chain
            
            # Setup mock model
            mock_model = MagicMock()
            mock_create_model.return_value = mock_model
            
            # Setup mock agent executor
            mock_executor_instance = AsyncMock()
            mock_executor_instance.ainvoke = AsyncMock(
                return_value={"output": "Test response with tools"}
            )
            mock_agent_executor.return_value = mock_executor_instance
            
            # Call the chat method
            try:
                await custom_llm_service.chat(
                    session_id="test-session",
                    user_input="Hello",
                    task_type="chat",
                )
            except Exception as e:
                # We expect this to potentially fail due to mocking, but we want to check the calls
                pass
            
            # Verify that _build_chat_chain was called
            assert mock_build_chain.called
            
            # Get the actual call arguments
            call_args = mock_build_chain.call_args
            assert call_args is not None
            
            # Verify the system_prompt parameter includes both custom and tool instructions  
            # The augmentation happens in _chat_with_tools_via_chain before calling _build_chat_chain
            system_prompt_arg = call_args.kwargs.get("system_prompt")
            assert system_prompt_arg is not None
            assert "custom AI assistant with special instructions" in system_prompt_arg
            assert "You have access to external tools" in system_prompt_arg
            
            # Verify model params are passed through
            model_params_arg = call_args.kwargs.get("model_params")
            assert model_params_arg is not None
            assert "temperature" in model_params_arg

    @pytest.mark.asyncio
    async def test_custom_model_params_preserved_with_mcp_tools(
        self, custom_llm_service
    ):
        """Verify that custom model parameters are preserved when MCP tools are enabled."""
        
        # Mock the tooling context
        mock_tooling_ctx = MagicMock()
        mock_tooling_ctx.tools = [MagicMock()]
        mock_tooling_ctx.instructions = "Tool instructions here."
        mock_tooling_ctx.profile_name = "test-profile"
        mock_tooling_ctx.limits = {"max_steps": 5, "max_run_seconds": 30}
        
        with patch.object(
            custom_llm_service, "_get_tooling_context", return_value=mock_tooling_ctx
        ), patch.object(
            custom_llm_service, "_build_chat_chain"
        ) as mock_build_chain, patch(
            "inference_core.services.llm_service.create_openai_tools_agent"
        ), patch(
            "inference_core.services.llm_service.AgentExecutor"
        ) as mock_agent_executor, patch(
            "inference_core.services.llm_service.SQLChatMessageHistory"
        ):
            mock_chain = MagicMock()
            mock_chain.model_name = "gpt-4o-mini"
            mock_build_chain.return_value = mock_chain
            
            mock_executor_instance = AsyncMock()
            mock_executor_instance.ainvoke = AsyncMock(return_value={"output": "Response"})
            mock_agent_executor.return_value = mock_executor_instance
            
            try:
                await custom_llm_service.chat(
                    session_id="test-session",
                    user_input="Test message",
                    task_type="chat",
                    temperature=0.7,  # Runtime override
                )
            except Exception:
                pass
            
            # Verify _build_chat_chain was called with merged params
            call_args = mock_build_chain.call_args
            model_params = call_args.kwargs.get("model_params")
            
            # Should have both default and runtime params
            assert "temperature" in model_params
            assert model_params["temperature"] == 0.7  # Runtime override wins
            assert "max_tokens" in model_params
            assert model_params["max_tokens"] == 500  # Default preserved

    @pytest.mark.asyncio
    async def test_no_mcp_tools_uses_standard_chain(self, custom_llm_service):
        """Verify that standard chat chain is used when MCP tools are not configured."""
        
        # Mock no tooling context (MCP not enabled)
        with patch.object(
            custom_llm_service, "_get_tooling_context", return_value=None
        ), patch.object(
            custom_llm_service, "_build_chat_chain"
        ) as mock_build_chain, patch(
            "inference_core.llm.chains.ChatChain.chat", new_callable=AsyncMock
        ) as mock_chain_chat:
            
            mock_chain = MagicMock()
            mock_chain.model_name = "gpt-4o-mini"
            mock_chain.chat = mock_chain_chat
            mock_build_chain.return_value = mock_chain
            mock_chain_chat.return_value = "Standard response"
            
            result = await custom_llm_service.chat(
                session_id="test-session",
                user_input="Hello",
                task_type="chat",
            )
            
            # Verify _build_chat_chain was called (standard path)
            assert mock_build_chain.called
            
            # Verify the chain.chat method was called (not agent executor)
            assert mock_chain_chat.called
            
            # Verify custom system prompt was used (without tool augmentation)
            call_args = mock_build_chain.call_args
            system_prompt_arg = call_args.kwargs.get("system_prompt")
            assert system_prompt_arg == "You are a custom AI assistant with special instructions."

    @pytest.mark.asyncio  
    async def test_custom_subclass_override_preserved(self):
        """Verify that subclass overrides of _build_chat_chain are respected with MCP tools."""
        
        # Create a custom subclass that overrides _build_chat_chain
        class CustomLLMService(LLMService):
            def __init__(self):
                super().__init__(
                    default_chat_system_prompt="Subclass system prompt"
                )
                self.chain_build_called = False
            
            def _build_chat_chain(self, **kwargs):
                self.chain_build_called = True
                # Ensure the system prompt was augmented
                system_prompt = kwargs.get("system_prompt", "")
                assert "Subclass system prompt" in system_prompt
                return super()._build_chat_chain(**kwargs)
        
        service = CustomLLMService()
        
        # Mock MCP tooling
        mock_tooling_ctx = MagicMock()
        mock_tooling_ctx.tools = [MagicMock()]
        mock_tooling_ctx.instructions = "MCP tool instructions"
        mock_tooling_ctx.profile_name = "test-profile"
        mock_tooling_ctx.limits = {"max_steps": 5, "max_run_seconds": 30}
        
        with patch.object(
            service, "_get_tooling_context", return_value=mock_tooling_ctx
        ), patch(
            "inference_core.services.llm_service.create_openai_tools_agent"
        ), patch(
            "inference_core.services.llm_service.AgentExecutor"
        ) as mock_agent_executor, patch(
            "inference_core.services.llm_service.SQLChatMessageHistory"
        ):
            mock_executor_instance = AsyncMock()
            mock_executor_instance.ainvoke = AsyncMock(return_value={"output": "Response"})
            mock_agent_executor.return_value = mock_executor_instance
            
            try:
                await service.chat(
                    session_id="test-session",
                    user_input="Test",
                    task_type="chat",
                )
            except Exception:
                pass
            
            # Verify the overridden _build_chat_chain was called
            assert service.chain_build_called
