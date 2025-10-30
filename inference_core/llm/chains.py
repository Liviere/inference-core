"""
LangChain Chains for Story Processing

Contains predefined chains for common story-related tasks.
Each chain combines prompts with models for specific functionality.
"""

import asyncio
import logging
from typing import Optional

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from inference_core.core.config import get_settings

from .models import get_model_factory
from .prompts import AVAILABLE_PROMPTS, get_chat_prompt_template

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

    def __init__(self, model_name: Optional[str] = None, **model_params):
        super().__init__(model_name, **model_params)
        self._build_chain()

    def _build_chain(self):
        """Build the completion chain"""
        model = self._get_model("completion")
        prompt = AVAILABLE_PROMPTS["completion"]

        self._chain = prompt | model | StrOutputParser()

    async def generate_story(
        self,
        question: str,
        callbacks=None,
    ) -> str:
        """Generate a answer to a question"""
        return await self.arun(
            callbacks=callbacks,
            question=question,
        )


def create_completion_chain(
    model_name: Optional[str] = None, **model_params
) -> CompletionChain:
    """Create a completion chain instance"""
    return CompletionChain(model_name, **model_params)


## Chat Chain
class ChatChain(BaseChain):
    """Chain for multi-turn chat using session-based history"""

    def __init__(self, model_name: Optional[str] = None, **model_params):
        super().__init__(model_name, **model_params)
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
        prompt = get_chat_prompt_template("chat")

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


def create_chat_chain(model_name: Optional[str] = None, **model_params) -> ChatChain:
    """Create a chat chain instance"""
    return ChatChain(model_name, **model_params)
