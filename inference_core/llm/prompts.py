"""
LLM Prompts Templates

Contains predefined prompt templates for various tasks.
Uses LangChain's PromptTemplate for consistent and reusable prompts.

Also supports loading custom Jinja-style templates from
`inference_core/custom_prompts/` so that built-in prompts stay immutable.

Conventions for custom prompts:
- Completion:  custom_prompts/completion/<name>.j2  (expects `{prompt}` variable)
- Chat (system): custom_prompts/chat/<name>.j2 or <name>.system.j2 (system message text)
    The chat template is composed as: system, MessagesPlaceholder('history'), human '{user_input}'.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

logger = logging.getLogger(__name__)

# Base directory containing custom prompts
# custom_prompts lives at inference_core/custom_prompts (one level up from llm/)
_CUSTOM_BASE = Path(__file__).resolve().parent.parent / "custom_prompts"
_MCP_DIR = _CUSTOM_BASE / "mcp"


class Prompts:
    """Collection of prompt templates for tasks"""

    # Canonical completion prompt expects 'prompt' input variable
    COMPLETION = PromptTemplate(
        input_variables=["prompt"],
        template="""
You are an expert in explaining complex concepts in simple terms.

Prompt: {prompt}

Answer:""",
    )


class ChatPrompts:
    """Chat-based prompt templates for conversational interactions"""

    # New canonical name aligned with API: chat
    CHAT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant that provides concise and accurate answers to user questions.
                Always respond in a friendly and professional manner.""",
            ),
            # History placeholder is required for RunnableWithMessageHistory
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ]
    )


def _load_text_file(path: Path) -> Optional[str]:
    try:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read custom prompt file {path}: {e}")
    return None


def _find_custom_file(dir_name: str, prompt_name: str) -> Optional[Path]:
    """Look for .j2/.jinja2 files for a given prompt in a subdir.

    Searches in inference_core/custom_prompts/<dir_name>/<prompt_name>.(j2|jinja2)
    and for chat also supports '<prompt_name>.system.(j2|jinja2)'.
    """
    base = _CUSTOM_BASE / dir_name
    if not base.exists():
        return None
    # Prefer preset-named files first, then generic names
    candidates = [
        # preset variants
        base / f"{prompt_name}.preset.j2",
        base / f"{prompt_name}.preset.jinja2",
        base / f"{prompt_name}.system.preset.j2",
        base / f"{prompt_name}.system.preset.jinja2",
        # generic variants
        base / f"{prompt_name}.j2",
        base / f"{prompt_name}.jinja2",
    ]
    if dir_name == "chat":
        candidates.extend(
            [
                base / f"{prompt_name}.system.j2",
                base / f"{prompt_name}.system.jinja2",
            ]
        )
    for p in candidates:
        if p.exists():
            return p
    return None


@lru_cache(maxsize=1)
def _get_mcp_environment() -> Optional[Environment]:
    """Return a cached Jinja2 environment for MCP custom templates."""

    if not _MCP_DIR.exists() or not _MCP_DIR.is_dir():
        return None

    try:
        loader = FileSystemLoader(str(_MCP_DIR))
        return Environment(
            loader=loader,
            autoescape=False,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
    except Exception as exc:
        logger.warning("Failed to initialize MCP template environment: %s", exc)
        return None


def render_custom_mcp_instructions(
    profile_name: str, context: Dict[str, Any]
) -> Optional[str]:
    """Render additional MCP instructions for a profile, if a template exists."""

    env = _get_mcp_environment()
    if env is None:
        return None

    candidates = [
        f"{profile_name}.j2",
        f"{profile_name}.jinja2",
        f"{profile_name}.tmpl",
    ]

    for template_name in candidates:
        try:
            template = env.get_template(template_name)
            rendered = template.render(**context)
            return rendered.strip()
        except TemplateNotFound:
            continue
        except Exception as exc:
            logger.warning("Failed to render MCP template '%s': %s", template_name, exc)
            return None

    return None


def get_custom_completion_prompt(prompt_name: str) -> Optional[PromptTemplate]:
    """Attempt to load a custom completion PromptTemplate from filesystem.

    The template should reference `{prompt}`.
    """
    path = _find_custom_file("completion", prompt_name)
    if not path:
        return None
    content = _load_text_file(path)
    if not content:
        return None
    try:
        # from_template will infer variables; we expect `{prompt}` to be present
        return PromptTemplate.from_template(content)
    except Exception as e:
        logger.warning(f"Invalid custom completion template '{prompt_name}': {e}")
        return None


def get_custom_chat_prompt(prompt_name: str) -> Optional[ChatPromptTemplate]:
    """Attempt to build a ChatPromptTemplate from a custom system prompt file.

    The file content is used as the system message; we preserve the default
    history placeholder and human input structure.
    """
    path = _find_custom_file("chat", prompt_name)
    if not path:
        return None
    system_text = _load_text_file(path)
    if not system_text:
        return None
    try:
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{user_input}"),
            ]
        )
    except Exception as e:
        logger.warning(f"Invalid custom chat template '{prompt_name}': {e}")
        return None


def get_prompt_template(prompt_name: str) -> PromptTemplate:
    """
    Get a prompt template by name

    Args:
        prompt_name: Name of the prompt template

    Returns:
        PromptTemplate instance

    Raises:
        AttributeError: If prompt template doesn't exist
    """
    try:
        return cast(PromptTemplate, getattr(Prompts, prompt_name.upper()))
    except AttributeError:
        # Fallback to custom prompt loader
        custom = get_custom_completion_prompt(prompt_name)
        if custom is not None:
            return custom
        raise


def get_chat_prompt_template(prompt_name: str) -> ChatPromptTemplate:
    """
    Get a chat prompt template by name

    Args:
        prompt_name: Name of the chat prompt template

    Returns:
        ChatPromptTemplate instance

    Raises:
        AttributeError: If chat prompt template doesn't exist
    """
    try:
        return cast(ChatPromptTemplate, getattr(ChatPrompts, prompt_name.upper()))
    except AttributeError:
        # Fallback to custom chat prompt loader
        custom = get_custom_chat_prompt(prompt_name)
        if custom is not None:
            return custom
        raise


# Available prompt templates
AVAILABLE_PROMPTS = {
    "completion": Prompts.COMPLETION,
}

AVAILABLE_CHAT_PROMPTS = {
    "chat": ChatPrompts.CHAT,
}
