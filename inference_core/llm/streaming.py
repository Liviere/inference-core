"""
LLM Streaming Support Module

This module provides streaming functionality for LLM chat models using
Server-Sent Events (SSE) and LangChain's streaming callback system.

Key Components:
- StreamingForwarderHandler: Callback handler that forwards tokens to an asyncio.Queue
- SSE formatting helpers for event streams
- Async generators for chat and completion streaming
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Request
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from inference_core.llm.config import get_llm_config
from inference_core.llm.models import get_model_factory
from inference_core.llm.prompts import get_chat_prompt_template, get_prompt_template
from inference_core.llm.usage_logging import UsageLogger

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """Represents a chunk of streamed data"""

    type: str  # 'start', 'token', 'usage', 'end', 'error'
    content: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    message: Optional[str] = None
    model: Optional[str] = None
    session_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


def format_sse(event_data: Dict[str, Any]) -> bytes:
    """
    Format data as Server-Sent Events (SSE) format.

    Args:
        event_data: Dictionary to be serialized as JSON

    Returns:
        Formatted SSE data as bytes
    """
    return f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode("utf-8")


def extract_usage_details(md: Any) -> Dict[str, int]:
    """Build a usage dict from a usage_metadata mapping / object.

    Extracts core tokens and detailed breakdown (cache_read, cache_creation, reasoning, etc.).
    Each detail becomes <detail>_tokens to stay consistent with accumulate() logic & pricing extras.
    Safe against providers omitting fields or returning None.
    """
    usage: Dict[str, int] = {}
    if md is None:
        return usage

    # Support both dict-like and attribute-style
    get_val = None
    if hasattr(md, "get") and callable(getattr(md, "get")):
        get_val = md.get  # type: ignore

    def read(key: str):
        if get_val:
            return get_val(key, 0) or 0
        return getattr(md, key, 0) or 0

    try:
        input_tokens = read("input_tokens")
        output_tokens = read("output_tokens")
        total_tokens = read("total_tokens") or (input_tokens + output_tokens)
        if input_tokens:
            usage["input_tokens"] = int(input_tokens)
        if output_tokens:
            usage["output_tokens"] = int(output_tokens)
        if total_tokens and total_tokens != input_tokens + output_tokens:
            # keep if provider supplies explicit different total
            usage["total_tokens"] = int(total_tokens)

        # Details (input_token_details, output_token_details) may appear as dicts
        input_details = None
        output_details = None
        if get_val:
            input_details = get_val("input_token_details", {}) or {}
            output_details = get_val("output_token_details", {}) or {}
        else:
            input_details = getattr(md, "input_token_details", None) or {}
            output_details = getattr(md, "output_token_details", None) or {}

        def add_details(details: Any):
            if isinstance(details, dict):
                for k, v in details.items():
                    if isinstance(v, (int, float)) and v > 0:
                        usage[f"{k}_tokens"] = usage.get(f"{k}_tokens", 0) + int(v)

        add_details(input_details)
        add_details(output_details)
    except Exception:
        pass  # fail open
    return usage


async def stream_chat(
    session_id: Optional[str],
    user_input: str,
    model_name: Optional[str] = None,
    request: Optional[Request] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **model_params,
) -> AsyncGenerator[bytes, None]:
    """
    Stream a chat response using Server-Sent Events.

    Args:
        session_id: Chat session ID (auto-generated if None)
        user_input: User's message
        model_name: Optional model override
        request: FastAPI request object for disconnect detection
        **model_params: Additional model parameters

    Yields:
        SSE-formatted bytes for each streaming event
    """
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    logger.info(f"Starting chat stream for session {session_id}")

    # Create bounded queue for token forwarding
    token_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    # --- Usage logging session (streaming) ---
    usage_logger: Optional[UsageLogger] = None
    usage_session = None
    finalized = False
    llm_config = get_llm_config()
    try:
        if llm_config.usage_logging.enabled:
            usage_logger = UsageLogger(llm_config.usage_logging)
    except Exception:
        usage_logger = None  # fail open

    try:
        # Get model factory and create streaming model
        factory = get_model_factory()
        default_model_name = factory.config.get_task_model("chat")

        # Resolve model / provider for usage logging
        resolved_model_name = model_name or default_model_name
        model_cfg = (
            factory.config.models.get(resolved_model_name)
            if factory and factory.config
            else None
        )
        provider = getattr(model_cfg, "provider", "unknown")

        # Start usage session BEFORE model call so latency includes queueing/history load
        if usage_logger and model_cfg:
            try:
                usage_session = usage_logger.start_session(
                    task_type="chat",
                    request_mode="streaming",
                    model_name=resolved_model_name,
                    provider=provider,
                    pricing_config=getattr(model_cfg, "pricing", None),
                    session_id=session_id,
                    user_id=(uuid.UUID(user_id) if user_id else None),
                    request_id=request_id,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Failed to start usage session (chat stream): {e}")
                usage_session = None

        # Create model with streaming enabled and callback handler
        model = factory.create_model(
            resolved_model_name,
            streaming=True,
            **model_params,
        )

        if not model:
            error_data = {"event": "error", "message": "Failed to create model"}
            yield format_sse(error_data)
            return

        # Emit start event EARLY (before loading history) for low-latency first byte
        start_data = {
            "event": "start",
            "model": getattr(
                getattr(model, "model_name", None), "__str__", lambda: None
            )()
            or getattr(model, "model_name", None)
            or model_name
            or default_model_name,
            "session_id": session_id,
        }
        # Correct indentation: yield inside try block
        yield format_sse(start_data)

        # Load chat history (can be I/O heavy) AFTER start was sent
        from inference_core.core.config import get_settings

        settings = get_settings()
        url = settings.database_url
        if "+aiosqlite" in url:
            connection_string = url.replace("+aiosqlite", "")
        elif "+asyncpg" in url:
            connection_string = url.replace("+asyncpg", "+psycopg")
        elif "+aiomysql" in url:
            connection_string = url.replace("+aiomysql", "+pymysql")
        else:
            connection_string = url

        history = SQLChatMessageHistory(
            session_id=session_id, connection_string=connection_string
        )

        # Build message list manually for streaming
        messages = []

        # Add system message from prompt template
        prompt_template = get_chat_prompt_template("chat")
        if prompt_template and hasattr(prompt_template, "messages"):
            for msg_template in prompt_template.messages:
                if hasattr(msg_template, "format"):
                    formatted = msg_template.format(user_input=user_input, history=[])
                    if formatted.type == "system":
                        from langchain_core.messages import SystemMessage

                        messages.append(SystemMessage(content=formatted.content))

        # Add chat history
        history_messages = history.messages
        messages.extend(history_messages)

        # Add current user message
        messages.append(HumanMessage(content=user_input))

        # Start streaming using LangChain astream_events (preferred) with fallback to astream
        async def event_stream_model():
            """Consume LangChain structured events and forward token pieces to queue."""
            try:
                # Try events API first
                used_events_api = False
                if hasattr(model, "astream_events"):
                    try:
                        async for ev in model.astream_events(messages, version="v1"):
                            used_events_api = True
                            name = ev.get("event") if isinstance(ev, dict) else None
                            if not name:
                                continue
                            if name in ("on_chat_model_stream", "on_llm_new_token"):
                                data = (
                                    ev.get("data", {}) if isinstance(ev, dict) else {}
                                )
                                chunk_obj = (
                                    data.get("chunk")
                                    if isinstance(data, dict)
                                    else None
                                )
                                # Extract content from chunk_obj
                                content = getattr(chunk_obj, "content", None)
                                pieces: list[str] = []
                                if isinstance(content, str):
                                    pieces.append(content)
                                elif isinstance(content, list):
                                    for part in content:
                                        if isinstance(part, str):
                                            pieces.append(part)
                                        elif isinstance(part, dict):
                                            t = part.get("text") or part.get("content")
                                            if isinstance(t, str):
                                                pieces.append(t)
                                for piece in pieces:
                                    if not piece:
                                        continue
                                    try:
                                        token_queue.put_nowait(
                                            StreamChunk(type="token", content=piece)
                                        )
                                    except asyncio.QueueFull:
                                        logger.warning(
                                            "Queue full, dropping event token piece"
                                        )
                            elif name in ("on_chat_model_end", "on_llm_end"):
                                data = (
                                    ev.get("data", {}) if isinstance(ev, dict) else {}
                                )
                                # Attempt usage extraction
                                usage = None
                                output = (
                                    data.get("output")
                                    if isinstance(data, dict)
                                    else None
                                )
                                if (
                                    output
                                    and hasattr(output, "usage_metadata")
                                    and output.usage_metadata
                                ):
                                    md = output.usage_metadata
                                    usage = extract_usage_details(md)
                                if usage:
                                    try:
                                        token_queue.put_nowait(
                                            StreamChunk(type="usage", usage=usage)
                                        )
                                    except asyncio.QueueFull:
                                        pass
                                try:
                                    token_queue.put_nowait(StreamChunk(type="end"))
                                except asyncio.QueueFull:
                                    pass
                    except Exception as ev_err:  # fall back to astream below
                        logger.warning(
                            f"astream_events failed, falling back to astream: {ev_err}"
                        )
                if not used_events_api:
                    # Fallback: simple astream iteration (may buffer larger chunks)
                    async for chunk in model.astream(messages):
                        raw_content = getattr(chunk, "content", None)
                        parts: list[str] = []
                        if isinstance(raw_content, str):
                            parts.append(raw_content)
                        elif isinstance(raw_content, list):
                            for part in raw_content:
                                if isinstance(part, str):
                                    parts.append(part)
                                elif isinstance(part, dict):
                                    t = part.get("text") or part.get("content")
                                    if isinstance(t, str):
                                        parts.append(t)
                        for piece in parts:
                            if not piece:
                                continue
                            try:
                                token_queue.put_nowait(
                                    StreamChunk(type="token", content=piece)
                                )
                            except asyncio.QueueFull:
                                logger.warning(
                                    "Queue full, dropping fallback token piece"
                                )
                    # Ensure end if fallback path used
                    try:
                        token_queue.put_nowait(StreamChunk(type="end"))
                    except asyncio.QueueFull:
                        pass
            except Exception as e:
                logger.error(f"Error in model event streaming: {e}")
                try:
                    token_queue.put_nowait(StreamChunk(type="error", message=str(e)))
                    token_queue.put_nowait(StreamChunk(type="end"))
                except asyncio.QueueFull:
                    pass

        stream_task = asyncio.create_task(event_stream_model())

        # Process tokens from queue
        accumulated_content = ""
        while True:
            try:
                # Check if client disconnected
                if request and await request.is_disconnected():
                    logger.info("Client disconnected, stopping stream")
                    stream_task.cancel()
                    break

                # Wait for next chunk with timeout
                try:
                    chunk = await asyncio.wait_for(token_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if chunk.type == "token" and chunk.content:
                    accumulated_content += chunk.content
                    token_data = {"event": "token", "content": chunk.content}
                    yield format_sse(token_data)

                elif chunk.type == "usage" and chunk.usage:
                    # Accumulate usage into session (single snapshot or potential deltas)
                    if usage_session:
                        try:
                            usage_session.accumulate(chunk.usage)
                        except Exception as e:  # pragma: no cover
                            logger.warning(f"Failed to accumulate streaming usage: {e}")
                    usage_data = {"event": "usage", "usage": chunk.usage}
                    yield format_sse(usage_data)

                elif chunk.type == "error":
                    error_data = {"event": "error", "message": chunk.message}
                    yield format_sse(error_data)
                    break

                elif chunk.type == "end":
                    # Stream completed successfully, persist history
                    if accumulated_content:
                        try:
                            # Add assistant message to history
                            ai_message = AIMessage(content=accumulated_content)
                            await asyncio.to_thread(history.add_message, ai_message)
                            logger.info(
                                f"Persisted assistant message to session {session_id}"
                            )
                        except Exception as e:
                            logger.error(f"Failed to persist history: {str(e)}")

                    end_data = {"event": "end"}
                    yield format_sse(end_data)
                    # Finalize usage logging (success)
                    if usage_session and not finalized:
                        try:
                            await usage_session.finalize(
                                success=True,
                                final_usage=usage_session.accumulated_usage,
                                streamed=True,
                                partial=False,
                            )
                        except Exception as e:  # pragma: no cover
                            logger.error(f"Usage finalize (chat stream) failed: {e}")
                        finally:
                            finalized = True
                    break

            except Exception as e:
                logger.error(f"Error processing stream chunk: {str(e)}")
                error_data = {"event": "error", "message": "Stream processing error"}
                yield format_sse(error_data)
                # Finalize as failure if not yet
                if usage_session and not finalized:
                    try:
                        await usage_session.finalize(
                            success=False,
                            error=e,
                            final_usage=usage_session.accumulated_usage,
                            streamed=True,
                            partial=True,
                        )
                    except Exception as fe:  # pragma: no cover
                        logger.error(f"Usage finalize (error) failed: {fe}")
                    finally:
                        finalized = True
                break

        # Ensure streaming task is cancelled
        if not stream_task.done():
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.error(f"Error in chat streaming: {str(e)}")
        error_data = {"event": "error", "message": str(e)}
        yield format_sse(error_data)

    finally:
        # Client disconnect path (loop break without end usage finalize)
        if usage_session and not finalized:
            try:
                await usage_session.finalize(
                    success=False,
                    final_usage=usage_session.accumulated_usage,
                    streamed=True,
                    partial=True,
                )
            except Exception as e:  # pragma: no cover
                logger.error(f"Usage finalize (disconnect) failed: {e}")
        logger.info(f"Chat stream ended for session {session_id}")


async def stream_completion(
    question: str,
    model_name: Optional[str] = None,
    request: Optional[Request] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **model_params,
) -> AsyncGenerator[bytes, None]:
    """
    Stream a completion response using Server-Sent Events.

    Args:
    question: Question to answer
        model_name: Optional model override
        request: FastAPI request object for disconnect detection
        **model_params: Additional model parameters

    Yields:
        SSE-formatted bytes for each streaming event
    """
    logger.info(f"Starting completion stream for question: {question[:100]}...")

    # Create bounded queue for token forwarding
    token_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    # Usage logging
    usage_logger: Optional[UsageLogger] = None
    usage_session = None
    finalized = False
    llm_config = get_llm_config()
    try:
        if llm_config.usage_logging.enabled:
            usage_logger = UsageLogger(llm_config.usage_logging)
    except Exception:
        usage_logger = None

    try:
        # Get model factory and create streaming model
        factory = get_model_factory()
        default_model_name = factory.config.get_task_model("completion")

        # Resolve model/provider
        resolved_model_name = model_name or default_model_name
        model_cfg = (
            factory.config.models.get(resolved_model_name)
            if factory and factory.config
            else None
        )
        provider = getattr(model_cfg, "provider", "unknown")

        if usage_logger and model_cfg:
            try:
                usage_session = usage_logger.start_session(
                    task_type="completion",
                    request_mode="streaming",
                    model_name=resolved_model_name,
                    provider=provider,
                    pricing_config=getattr(model_cfg, "pricing", None),
                    user_id=(uuid.UUID(user_id) if user_id else None),
                    request_id=request_id,
                )
            except Exception as e:  # pragma: no cover
                logger.error(f"Failed to start usage session (completion stream): {e}")
                usage_session = None

        # Create model with streaming enabled and callback handler
        model = factory.create_model(
            resolved_model_name,
            streaming=True,
            **model_params,
        )

        if not model:
            error_data = {"event": "error", "message": "Failed to create model"}
            yield format_sse(error_data)
            return
        # Build prompt for completion (system + user)
        prompt_template = get_prompt_template("completion")
        input_data = {"question": question}

        # Emit start event
        start_data = {"event": "start", "model": model_name or "completion"}
        yield format_sse(start_data)

        # Background task using events API with fallback
        async def event_stream_model():
            try:
                messages = []
                if prompt_template:
                    try:
                        formatted = prompt_template.format(**input_data)
                        if isinstance(formatted, str):
                            messages.append(HumanMessage(content=formatted))
                        else:
                            messages.append(HumanMessage(content=question))
                    except Exception:
                        messages.append(HumanMessage(content=question))
                else:
                    messages.append(HumanMessage(content=question))

                used_events_api = False
                if hasattr(model, "astream_events"):
                    try:
                        async for ev in model.astream_events(messages, version="v1"):
                            used_events_api = True
                            name = ev.get("event") if isinstance(ev, dict) else None
                            if not name:
                                continue
                            if name in ("on_chat_model_stream", "on_llm_new_token"):
                                data = (
                                    ev.get("data", {}) if isinstance(ev, dict) else {}
                                )
                                chunk_obj = (
                                    data.get("chunk")
                                    if isinstance(data, dict)
                                    else None
                                )
                                content = getattr(chunk_obj, "content", None)
                                parts: list[str] = []
                                if isinstance(content, str):
                                    parts.append(content)
                                elif isinstance(content, list):
                                    for part in content:
                                        if isinstance(part, str):
                                            parts.append(part)
                                        elif isinstance(part, dict):
                                            t = part.get("text") or part.get("content")
                                            if isinstance(t, str):
                                                parts.append(t)
                                for piece in parts:
                                    if not piece:
                                        continue
                                    try:
                                        token_queue.put_nowait(
                                            StreamChunk(type="token", content=piece)
                                        )
                                    except asyncio.QueueFull:
                                        logger.warning(
                                            "Queue full, dropping completion event token piece"
                                        )
                            elif name in ("on_chat_model_end", "on_llm_end"):
                                data = (
                                    ev.get("data", {}) if isinstance(ev, dict) else {}
                                )
                                usage = None
                                output = (
                                    data.get("output")
                                    if isinstance(data, dict)
                                    else None
                                )
                                if (
                                    output
                                    and hasattr(output, "usage_metadata")
                                    and output.usage_metadata
                                ):
                                    md = output.usage_metadata
                                    usage = extract_usage_details(md)
                                if usage:
                                    try:
                                        token_queue.put_nowait(
                                            StreamChunk(type="usage", usage=usage)
                                        )
                                    except asyncio.QueueFull:
                                        pass
                                try:
                                    token_queue.put_nowait(StreamChunk(type="end"))
                                except asyncio.QueueFull:
                                    pass
                    except Exception as ev_err:
                        logger.warning(
                            f"astream_events(completion) failed, fallback to astream: {ev_err}"
                        )
                if not used_events_api:
                    async for chunk in model.astream(messages):
                        raw_content = getattr(chunk, "content", None)
                        parts: list[str] = []
                        if isinstance(raw_content, str):
                            parts.append(raw_content)
                        elif isinstance(raw_content, list):
                            for part in raw_content:
                                if isinstance(part, str):
                                    parts.append(part)
                                elif isinstance(part, dict):
                                    t = part.get("text") or part.get("content")
                                    if isinstance(t, str):
                                        parts.append(t)
                        for piece in parts:
                            if not piece:
                                continue
                            try:
                                token_queue.put_nowait(
                                    StreamChunk(type="token", content=piece)
                                )
                            except asyncio.QueueFull:
                                logger.warning(
                                    "Queue full, dropping completion fallback token piece"
                                )
                    try:
                        token_queue.put_nowait(StreamChunk(type="end"))
                    except asyncio.QueueFull:
                        pass
            except Exception as e:
                logger.error(f"Error in completion model event streaming: {e}")
                try:
                    token_queue.put_nowait(StreamChunk(type="error", message=str(e)))
                    token_queue.put_nowait(StreamChunk(type="end"))
                except asyncio.QueueFull:
                    pass

        stream_task = asyncio.create_task(event_stream_model())

        # Consume queue and yield SSE
        while True:
            try:
                if request and await request.is_disconnected():
                    logger.info("Client disconnected, stopping stream")
                    stream_task.cancel()
                    break
                try:
                    chunk = await asyncio.wait_for(token_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if chunk.type == "token" and chunk.content:
                    yield format_sse({"event": "token", "content": chunk.content})
                elif chunk.type == "usage" and chunk.usage:
                    if usage_session:
                        try:
                            usage_session.accumulate(chunk.usage)
                        except Exception as e:  # pragma: no cover
                            logger.warning(
                                f"Failed to accumulate completion streaming usage: {e}"
                            )
                    yield format_sse({"event": "usage", "usage": chunk.usage})
                elif chunk.type == "error":
                    yield format_sse({"event": "error", "message": chunk.message})
                    if usage_session and not finalized:
                        try:
                            await usage_session.finalize(
                                success=False,
                                error=(
                                    Exception(chunk.message) if chunk.message else None
                                ),
                                final_usage=usage_session.accumulated_usage,
                                streamed=True,
                                partial=True,
                            )
                        except Exception as fe:  # pragma: no cover
                            logger.error(
                                f"Usage finalize (completion error) failed: {fe}"
                            )
                        finally:
                            finalized = True
                    break
                elif chunk.type == "end":
                    yield format_sse({"event": "end"})
                    if usage_session and not finalized:
                        try:
                            await usage_session.finalize(
                                success=True,
                                final_usage=usage_session.accumulated_usage,
                                streamed=True,
                                partial=False,
                            )
                        except Exception as fe:  # pragma: no cover
                            logger.error(
                                f"Usage finalize (completion success) failed: {fe}"
                            )
                        finally:
                            finalized = True
                    break
            except Exception as e:
                logger.error(f"Error processing stream chunk: {str(e)}")
                yield format_sse(
                    {"event": "error", "message": "Stream processing error"}
                )
                break

        if not stream_task.done():
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.error(f"Error in completion streaming: {str(e)}")
        error_data = {"event": "error", "message": str(e)}
        yield format_sse(error_data)

    finally:
        if usage_session and not finalized:
            try:
                await usage_session.finalize(
                    success=False,
                    final_usage=usage_session.accumulated_usage,
                    streamed=True,
                    partial=True,
                )
            except Exception as e:  # pragma: no cover
                logger.error(f"Usage finalize (completion disconnect) failed: {e}")
    logger.info("Completion stream ended")
