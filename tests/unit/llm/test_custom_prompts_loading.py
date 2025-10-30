import pytest

from inference_core.llm.prompts import get_chat_prompt_template, get_prompt_template


def test_custom_completion_template_loads_short_answer():
    tmpl = get_prompt_template("short_answer")
    formatted = tmpl.format(prompt="What is Python?")
    assert "Respond in 1–2 sentences" in formatted
    assert "Question:" in formatted


def test_custom_completion_template_loads_simple_explainer():
    tmpl = get_prompt_template("simple_explainer")
    text = tmpl.format(prompt="Neural networks")
    assert "explains complex ideas in simple language" in text
    assert text.strip().endswith("Explanation:")


def test_custom_chat_template_loads_tutor():
    prompt = get_chat_prompt_template("tutor")
    # Render messages similarly to streaming code
    system_content = None
    for msg_template in prompt.messages:
        if hasattr(msg_template, "format"):
            formatted = msg_template.format(user_input="hi", history=[])
            if getattr(formatted, "type", None) == "system":
                system_content = formatted.content
                break
    assert system_content is not None
    assert "supportive tutor" in system_content


def test_custom_chat_template_loads_sales_assistant():
    prompt = get_chat_prompt_template("sales_assistant")
    system_content = None
    for msg_template in prompt.messages:
        if hasattr(msg_template, "format"):
            formatted = msg_template.format(user_input="hello", history=[])
            if getattr(formatted, "type", None) == "system":
                system_content = formatted.content
                break
    assert system_content is not None
    assert "sales assistant" in system_content
