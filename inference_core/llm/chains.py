"""
LangChain Chains for Story Processing

Contains predefined chains for common story-related tasks.
Each chain combines prompts with models for specific functionality.
"""

import asyncio
import logging
from typing import Optional, Union

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from inference_core.core.config import get_settings

from .models import get_model_factory
from .prompts import AVAILABLE_PROMPTS, get_chat_prompt_template, get_prompt_template

logger = logging.getLogger(__name__)


class BaseChain:
    """Base class for processing chains"""

    def __init__(self, model_name: Optional[str] = None, **model_params):
        self.model_factory = get_model_factory()
        self.model_name = model_name
        self.model_params = model_params
        self._chain = None

    def _get_model(self, task: str) -> BaseChatModel:
        """Get model for the chain"""
        if self.model_name:
            model = self.model_factory.create_model(
                self.model_name, **self.model_params
            )
        else:
            params = self.model_params
            model = self.model_factory.get_model_for_task(task, **params)
            self.model_name = model.model_name

        if not model:
            raise ValueError(f"Could not create model for task: {task}")
        return model

    async def arun(self, callbacks=None, **kwargs) -> str:
        """Run the chain asynchronously with optional callbacks.

        The callbacks parameter (list[BaseCallbackHandler]) is propagated via
        the Runnable config so that provider-level LLM calls emit token usage
        events to our custom handler.
        """
        if not self._chain:
            raise NotImplementedError("Chain not implemented")

        try:
            config = {}
            if callbacks:
                config = {"callbacks": callbacks}
            result = await self._chain.ainvoke(kwargs, config=config)
            return result
        except Exception as e:
            logger.error(f"Chain execution failed: {str(e)}")
            raise


# Chain implementations for specific tasks
## Completion Chain
class CompletionChain(BaseChain):
    """Chain for single-turn completions"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        prompt: Optional[PromptTemplate] = None,
        prompt_name: Optional[str] = None,
        **model_params,
    ):
        super().__init__(model_name, **model_params)
        self._custom_prompt: Optional[PromptTemplate] = prompt
        self._prompt_name: Optional[str] = prompt_name
        self._build_chain()

    def _build_chain(self):
        """Build the completion chain"""
        model = self._get_model("completion")
        # Select prompt precedence: explicit instance > by-name > default
        if self._custom_prompt is not None:
            prompt = self._custom_prompt
        elif self._prompt_name is not None:
            try:
                prompt = get_prompt_template(self._prompt_name)
            except Exception:
                logger.warning(
                    f"Unknown completion prompt_name='{self._prompt_name}', falling back to default"
                )
                prompt = AVAILABLE_PROMPTS["completion"]
        else:
            prompt = AVAILABLE_PROMPTS["completion"]

        self._chain = prompt | model | StrOutputParser()

    async def completion(
        self,
        *,
        prompt: str,
        callbacks=None,
    ) -> str:
        """Generate a completion for the given prompt."""

        if not prompt:
            raise ValueError("Prompt text is required for completion")

        return await self.arun(
            callbacks=callbacks,
            prompt=prompt,
        )


def create_completion_chain(
    model_name: Optional[str] = None,
    *,
    prompt: Optional[PromptTemplate] = None,
    prompt_name: Optional[str] = None,
    **model_params,
) -> CompletionChain:
    """Create a completion chain instance"""
    return CompletionChain(
        model_name,
        prompt=prompt,
        prompt_name=prompt_name,
        **model_params,
    )


## Chat Chain
class ChatChain(BaseChain):
    """Chain for multi-turn chat using session-based history"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        prompt: Optional[ChatPromptTemplate] = None,
        prompt_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **model_params,
    ):
        super().__init__(model_name, **model_params)
        self._custom_prompt: Optional[ChatPromptTemplate] = prompt
        self._prompt_name: Optional[str] = prompt_name
        self._system_prompt: Optional[str] = system_prompt
        self._build_chain()

    @staticmethod
    def _sync_connection_string() -> str:
        """Return a sync SQLAlchemy connection string for SQLChatMessageHistory.

        Map async drivers to their sync counterparts for sync-mode history.
        """
        settings = get_settings()
        url = settings.database_url
        if "+aiosqlite" in url:
            return url.replace("+aiosqlite", "")
        if "+asyncpg" in url:
            return url.replace("+asyncpg", "+psycopg")
        if "+aiomysql" in url:
            return url.replace("+aiomysql", "+pymysql")
        return url

    def _build_chain(self):
        """Build the chat chain with message history support"""
        model = self._get_model("chat")
        # Determine base prompt
        if self._custom_prompt is not None:
            prompt = self._custom_prompt
        elif self._prompt_name is not None:
            try:
                prompt = get_chat_prompt_template(self._prompt_name)
            except Exception:
                logger.warning(
                    f"Unknown chat prompt_name='{self._prompt_name}', falling back to default 'chat'"
                )
                prompt = get_chat_prompt_template("chat")
        else:
            prompt = get_chat_prompt_template("chat")

        # Optionally replace the first system message content
        if self._system_prompt:
            try:
                # Recompose ChatPromptTemplate with overridden system content
                base_msgs = []
                from langchain_core.prompts import (
                    ChatPromptTemplate,
                    MessagesPlaceholder,
                )

                base_msgs.append(("system", self._system_prompt))
                # Preserve placeholders and human template from existing prompt
                # We assume the canonical structure: system, history placeholder, human
                base_msgs.append(MessagesPlaceholder(variable_name="history"))
                base_msgs.append(("human", "{user_input}"))
                prompt = ChatPromptTemplate.from_messages(base_msgs)
            except Exception as e:
                logger.warning(f"Failed to apply system_prompt override: {e}")

        # Base LCEL chain (keep model output as AIMessage for history updates)
        base_chain = prompt | model

        # Wrap with message history per LangChain best practices
        self._chain = RunnableWithMessageHistory(
            base_chain,
            lambda session_id: SQLChatMessageHistory(
                session_id=session_id,
                connection_string=self._sync_connection_string(),
            ),
            input_messages_key="user_input",
            history_messages_key="history",
        )

    async def chat(self, session_id: str, user_input: str, callbacks=None) -> str:
        """Send a user message within a session and get assistant reply"""
        if not self._chain:
            raise NotImplementedError("Chat chain not initialized")

        try:
            config = {"configurable": {"session_id": session_id}}
            if callbacks:
                config["callbacks"] = callbacks
            # Use sync invoke in a worker thread due to SQLChatMessageHistory sync operations
            result = await asyncio.to_thread(
                self._chain.invoke,
                {"user_input": user_input},
                config=config,
            )
            return getattr(result, "content", str(result))
        except Exception as e:
            logger.error(f"Chat execution failed: {str(e)}")
            raise


def create_chat_chain(
    model_name: Optional[str] = None,
    *,
    prompt: Optional[ChatPromptTemplate] = None,
    prompt_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    **model_params,
) -> ChatChain:
    """Create a chat chain instance"""
    return ChatChain(
        model_name,
        prompt=prompt,
        prompt_name=prompt_name,
        system_prompt=system_prompt,
        **model_params,
    )
