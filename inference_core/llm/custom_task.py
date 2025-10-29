"""Generic usage/cost logging abstraction for custom LLM tasks.

This module provides reusable helpers that enable usage and cost logging
for any custom LLM task (extraction, summarization, classification, etc.)
without duplicating boilerplate.

The helpers:
- Start a usage/cost logging session for an arbitrary task type
- Execute a LangChain Runnable with callbacks attached
- Finalize the session (success/error) consistently
- Support both sync and streaming modes
- Respect pricing config from llm_config.yaml

Source references:
  - LLMService explain/converse patterns (sync mode)
  - streaming.py stream_conversation (streaming mode)
  - UsageLogger and LLMUsageCallbackHandler (usage tracking)
"""

import logging
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from inference_core.llm.callbacks import LLMUsageCallbackHandler
from inference_core.llm.config import get_llm_config
from inference_core.llm.usage_logging import UsageLogger

logger = logging.getLogger(__name__)


async def run_with_usage(
    *,
    task_type: str,
    runnable: Any,
    input: Dict[str, Any],
    model_name: str,
    request_mode: str = "sync",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    extra_callbacks: Optional[list] = None,
) -> Any:
    """Execute a LangChain Runnable with automatic usage/cost logging.

    This helper wraps any LangChain Runnable (chain, model, etc.) with
    usage tracking, similar to the built-in explain/conversation tasks.

    Args:
        task_type: Task identifier (e.g., "extraction", "summarization")
        runnable: LangChain Runnable/Chain/Model to execute
        input: Input payload for runnable.ainvoke()
        model_name: Resolved model name (for pricing lookup)
        request_mode: "sync" or "streaming" (default: "sync")
        session_id: Optional session identifier
        user_id: Optional user UUID string (converted to UUID internally)
        request_id: Optional request identifier
        extra_callbacks: Additional app-specific callbacks to include

    Returns:
        Result from runnable.ainvoke()

    Raises:
        Exception: Any exception from runnable execution is re-raised
                  after finalizing the usage session

    Example:
        ```python
        from inference_core.llm.custom_task import run_with_usage
        from langchain_core.output_parsers import StrOutputParser

        # Create your custom chain
        chain = extraction_prompt | model | StrOutputParser()

        # Execute with usage tracking
        result = await run_with_usage(
            task_type="extraction",
            runnable=chain,
            input={"text": "Extract entities from this text..."},
            model_name="openai/gpt-4o-mini",
            request_mode="sync",
            session_id=f"user:{user_id}",
        )
        ```
    """
    # Read llm_config to resolve provider/pricing for the model
    cfg = get_llm_config()
    model_cfg = cfg.models.get(model_name)
    provider = model_cfg.provider if model_cfg else "unknown"

    # Initialize usage logger
    usage_logger = UsageLogger(cfg.usage_logging)

    # Convert user_id string to UUID if provided
    user_uuid = None
    if user_id:
        try:
            user_uuid = uuid.UUID(user_id)
        except (ValueError, AttributeError):
            logger.warning(f"Invalid user_id format: {user_id}")

    # Start usage session
    session = usage_logger.start_session(
        task_type=task_type,
        request_mode=request_mode,
        model_name=model_name,
        provider=provider,
        pricing_config=model_cfg.pricing if model_cfg else None,
        user_id=user_uuid,
        session_id=session_id,
        request_id=request_id,
    )

    # Build callbacks list
    callbacks = []
    if cfg.usage_logging.enabled:
        callbacks.append(
            LLMUsageCallbackHandler(
                usage_session=session,
                pricing_config=model_cfg.pricing if model_cfg else None,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    try:
        # Execute runnable with callbacks
        result = await runnable.ainvoke(input, config={"callbacks": callbacks})

        # Finalize session on success
        await session.finalize(
            success=True,
            final_usage=session.accumulated_usage,
            streamed=(request_mode == "streaming"),
            partial=False,
        )

        return result

    except Exception as e:
        # Finalize session on error
        await session.finalize(
            success=False,
            error=e,
            streamed=(request_mode == "streaming"),
            partial=False,
        )
        raise


async def stream_with_usage(
    *,
    task_type: str,
    runnable: Any,
    input: Dict[str, Any],
    model_name: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    extra_callbacks: Optional[list] = None,
) -> AsyncGenerator[Any, None]:
    """Execute a LangChain Runnable in streaming mode with usage/cost logging.

    This helper wraps any LangChain Runnable with streaming support,
    yielding chunks while accumulating usage data.

    Args:
        task_type: Task identifier (e.g., "extraction", "summarization")
        runnable: LangChain Runnable/Chain/Model to execute (must support astream)
        input: Input payload for runnable.astream()
        model_name: Resolved model name (for pricing lookup)
        session_id: Optional session identifier
        user_id: Optional user UUID string (converted to UUID internally)
        request_id: Optional request identifier
        extra_callbacks: Additional app-specific callbacks to include

    Yields:
        Chunks from runnable.astream()

    Raises:
        Exception: Any exception from runnable execution is re-raised
                  after finalizing the usage session

    Example:
        ```python
        from inference_core.llm.custom_task import stream_with_usage

        # Create your streaming chain
        chain = extraction_prompt | streaming_model | StrOutputParser()

        # Execute with usage tracking
        async for chunk in stream_with_usage(
            task_type="extraction",
            runnable=chain,
            input={"text": "Extract entities..."},
            model_name="openai/gpt-4o-mini",
            session_id=f"user:{user_id}",
        ):
            print(chunk, end="", flush=True)
        ```
    """
    # Read llm_config to resolve provider/pricing for the model
    cfg = get_llm_config()
    model_cfg = cfg.models.get(model_name)
    provider = model_cfg.provider if model_cfg else "unknown"

    # Initialize usage logger
    usage_logger = UsageLogger(cfg.usage_logging)

    # Convert user_id string to UUID if provided
    user_uuid = None
    if user_id:
        try:
            user_uuid = uuid.UUID(user_id)
        except (ValueError, AttributeError):
            logger.warning(f"Invalid user_id format: {user_id}")

    # Start usage session
    session = usage_logger.start_session(
        task_type=task_type,
        request_mode="streaming",
        model_name=model_name,
        provider=provider,
        pricing_config=model_cfg.pricing if model_cfg else None,
        user_id=user_uuid,
        session_id=session_id,
        request_id=request_id,
    )

    # Build callbacks list
    callbacks = []
    if cfg.usage_logging.enabled:
        callbacks.append(
            LLMUsageCallbackHandler(
                usage_session=session,
                pricing_config=model_cfg.pricing if model_cfg else None,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    try:
        # Stream from runnable with callbacks
        async for chunk in runnable.astream(input, config={"callbacks": callbacks}):
            yield chunk

        # Finalize session on success
        await session.finalize(
            success=True,
            final_usage=session.accumulated_usage,
            streamed=True,
            partial=False,
        )

    except Exception as e:
        # Finalize session on error
        await session.finalize(
            success=False,
            error=e,
            streamed=True,
            partial=True,  # Mark as partial since streaming was interrupted
        )
        raise


__all__ = ["run_with_usage", "stream_with_usage"]
