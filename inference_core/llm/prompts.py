"""
LLM Prompts Templates

Contains predefined prompt templates for various tasks.
Uses LangChain's PromptTemplate for consistent and reusable prompts.
"""

from typing import cast

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)


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
    return cast(PromptTemplate, getattr(Prompts, prompt_name.upper()))


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
    return cast(ChatPromptTemplate, getattr(ChatPrompts, prompt_name.upper()))


# Available prompt templates
AVAILABLE_PROMPTS = {
    "completion": Prompts.COMPLETION,
}

AVAILABLE_CHAT_PROMPTS = {
    "chat": ChatPrompts.CHAT,
}
