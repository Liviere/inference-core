import types

from inference_core.services.llm_service import LLMService


class _DummyTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class _DummyLimits:
    max_steps = 3
    max_run_seconds = 15
    tool_retry_attempts = 1
    allowlist_hosts = ["*"]
    rate_limits = None


def test_build_tool_instructions_includes_custom_template():
    tools = [_DummyTool("browser_navigate", "Navigate to a URL")]
    limits = _DummyLimits()

    text = LLMService._build_tool_instructions(
        profile_name="web-browsing",
        tools=tools,
        limits=limits,
    )

    # Should include default generated header
    assert "You have access to external tools via the Model Context Protocol" in text
    assert "Navigate to a URL" in text
