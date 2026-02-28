"""Tests for inference_core.llm.prompts.

Covers built-in prompt templates, custom file loading, MCP template rendering,
and fallback chains in get_prompt_template / get_chat_prompt_template.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from jinja2 import TemplateNotFound

from inference_core.llm.prompts import (
    AVAILABLE_CHAT_PROMPTS,
    AVAILABLE_PROMPTS,
    ChatPrompts,
    Prompts,
    _find_custom_file,
    _get_mcp_environment,
    _load_text_file,
    get_chat_prompt_template,
    get_custom_chat_prompt,
    get_custom_completion_prompt,
    get_prompt_template,
    render_custom_mcp_instructions,
)

# ============================================================================
# Built-in templates
# ============================================================================


class TestBuiltinPrompts:
    """Verify built-in PromptTemplate and ChatPromptTemplate structure."""

    def test_completion_prompt_has_prompt_variable(self):
        assert "prompt" in Prompts.COMPLETION.input_variables

    def test_chat_prompt_has_expected_messages(self):
        # ChatPromptTemplate should have system, history, human
        msgs = ChatPrompts.CHAT.messages
        assert len(msgs) >= 3

    def test_available_prompts_dict(self):
        assert "completion" in AVAILABLE_PROMPTS
        assert AVAILABLE_PROMPTS["completion"] is Prompts.COMPLETION

    def test_available_chat_prompts_dict(self):
        assert "chat" in AVAILABLE_CHAT_PROMPTS
        assert AVAILABLE_CHAT_PROMPTS["chat"] is ChatPrompts.CHAT


# ============================================================================
# _load_text_file
# ============================================================================


class TestLoadTextFile:
    """_load_text_file reads file content or returns None."""

    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        assert _load_text_file(f) == "hello"

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _load_text_file(tmp_path / "nonexistent.txt") is None

    def test_returns_none_for_directory(self, tmp_path):
        assert _load_text_file(tmp_path) is None

    def test_returns_none_on_read_error(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("data", encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result = _load_text_file(f)
        assert result is None


# ============================================================================
# _find_custom_file
# ============================================================================


class TestFindCustomFile:
    """_find_custom_file searches for templates in priority order."""

    def test_returns_none_when_dir_missing(self):
        with patch(
            "inference_core.llm.prompts._CUSTOM_BASE",
            Path("/nonexistent/path"),
        ):
            assert _find_custom_file("completion", "test") is None

    def test_finds_preset_j2_first(self, tmp_path):
        """Preset variant takes priority over generic."""
        base = tmp_path / "completion"
        base.mkdir()
        (base / "test.preset.j2").write_text("preset")
        (base / "test.j2").write_text("generic")

        with patch("inference_core.llm.prompts._CUSTOM_BASE", tmp_path):
            result = _find_custom_file("completion", "test")
        assert result is not None
        assert "preset" in result.name

    def test_finds_generic_j2(self, tmp_path):
        base = tmp_path / "completion"
        base.mkdir()
        (base / "mytest.j2").write_text("generic template")

        with patch("inference_core.llm.prompts._CUSTOM_BASE", tmp_path):
            result = _find_custom_file("completion", "mytest")
        assert result is not None
        assert result.name == "mytest.j2"

    def test_chat_dir_adds_system_variants(self, tmp_path):
        """Chat dir also searches for .system.j2 variants."""
        base = tmp_path / "chat"
        base.mkdir()
        (base / "test.system.j2").write_text("system prompt")

        with patch("inference_core.llm.prompts._CUSTOM_BASE", tmp_path):
            result = _find_custom_file("chat", "test")
        assert result is not None
        assert "system" in result.name

    def test_returns_none_when_no_match(self, tmp_path):
        base = tmp_path / "completion"
        base.mkdir()
        with patch("inference_core.llm.prompts._CUSTOM_BASE", tmp_path):
            assert _find_custom_file("completion", "nonexistent") is None


# ============================================================================
# _get_mcp_environment
# ============================================================================


class TestGetMcpEnvironment:
    """_get_mcp_environment returns cached Jinja2 Environment or None."""

    def setup_method(self):
        # Clear LRU cache between tests
        _get_mcp_environment.cache_clear()

    def test_returns_none_when_dir_missing(self):
        with patch(
            "inference_core.llm.prompts._MCP_DIR",
            Path("/nonexistent/mcp"),
        ):
            assert _get_mcp_environment() is None

    def test_returns_environment_when_dir_exists(self, tmp_path):
        mcp_dir = tmp_path / "mcp"
        mcp_dir.mkdir()
        with patch("inference_core.llm.prompts._MCP_DIR", mcp_dir):
            env = _get_mcp_environment()
        assert env is not None

    def test_returns_none_on_init_error(self, tmp_path):
        mcp_dir = tmp_path / "mcp"
        mcp_dir.mkdir()
        with patch("inference_core.llm.prompts._MCP_DIR", mcp_dir):
            with patch(
                "inference_core.llm.prompts.FileSystemLoader",
                side_effect=RuntimeError("init error"),
            ):
                assert _get_mcp_environment() is None


# ============================================================================
# render_custom_mcp_instructions
# ============================================================================


class TestRenderCustomMcpInstructions:
    """render_custom_mcp_instructions renders MCP Jinja templates."""

    def setup_method(self):
        _get_mcp_environment.cache_clear()

    def test_returns_none_when_no_env(self):
        with patch(
            "inference_core.llm.prompts._get_mcp_environment", return_value=None
        ):
            assert render_custom_mcp_instructions("test", {}) is None

    def test_renders_j2_template(self, tmp_path):
        """Finds and renders a .j2 template successfully."""
        mcp_dir = tmp_path / "mcp"
        mcp_dir.mkdir()
        (mcp_dir / "profile.j2").write_text("Hello {{ name }}")

        with patch("inference_core.llm.prompts._MCP_DIR", mcp_dir):
            result = render_custom_mcp_instructions("profile", {"name": "World"})
        assert result == "Hello World"

    def test_returns_none_when_no_template_found(self, tmp_path):
        """All candidates miss → returns None."""
        mcp_dir = tmp_path / "mcp"
        mcp_dir.mkdir()

        with patch("inference_core.llm.prompts._MCP_DIR", mcp_dir):
            result = render_custom_mcp_instructions("missing_profile", {})
        assert result is None

    def test_returns_none_on_render_error(self):
        """Non-TemplateNotFound exception → returns None."""
        mock_env = MagicMock()
        mock_env.get_template.side_effect = RuntimeError("render boom")

        with patch(
            "inference_core.llm.prompts._get_mcp_environment",
            return_value=mock_env,
        ):
            result = render_custom_mcp_instructions("test", {})
        assert result is None


# ============================================================================
# get_custom_completion_prompt / get_custom_chat_prompt
# ============================================================================


class TestGetCustomCompletionPrompt:
    """get_custom_completion_prompt loads PromptTemplate from filesystem."""

    def test_returns_none_when_no_file(self):
        with patch(
            "inference_core.llm.prompts._find_custom_file", return_value=None
        ):
            assert get_custom_completion_prompt("test") is None

    def test_returns_template_from_file(self, tmp_path):
        f = tmp_path / "test.j2"
        f.write_text("Answer this: {prompt}")
        with patch(
            "inference_core.llm.prompts._find_custom_file", return_value=f
        ):
            result = get_custom_completion_prompt("test")
        assert result is not None
        assert "prompt" in result.input_variables

    def test_returns_none_on_invalid_template(self, tmp_path):
        """Malformed template content → warning + None."""
        f = tmp_path / "test.j2"
        f.write_text("valid content")
        with patch(
            "inference_core.llm.prompts._find_custom_file", return_value=f
        ):
            with patch(
                "inference_core.llm.prompts.PromptTemplate.from_template",
                side_effect=ValueError("bad template"),
            ):
                assert get_custom_completion_prompt("broken") is None


class TestGetCustomChatPrompt:
    """get_custom_chat_prompt builds ChatPromptTemplate from system prompt file."""

    def test_returns_none_when_no_file(self):
        with patch(
            "inference_core.llm.prompts._find_custom_file", return_value=None
        ):
            assert get_custom_chat_prompt("test") is None

    def test_returns_chat_template_from_file(self, tmp_path):
        f = tmp_path / "test.j2"
        f.write_text("You are a helpful bot.")
        with patch(
            "inference_core.llm.prompts._find_custom_file", return_value=f
        ):
            result = get_custom_chat_prompt("test")
        assert result is not None
        assert len(result.messages) >= 3  # system, history, human


# ============================================================================
# get_prompt_template / get_chat_prompt_template
# ============================================================================


class TestGetPromptTemplate:
    """get_prompt_template resolves builtin → custom → raises AttributeError."""

    def test_returns_builtin_completion(self):
        result = get_prompt_template("completion")
        assert result is Prompts.COMPLETION

    def test_falls_back_to_custom(self):
        mock_custom = MagicMock()
        with patch(
            "inference_core.llm.prompts.get_custom_completion_prompt",
            return_value=mock_custom,
        ):
            result = get_prompt_template("nonexistent_name")
        assert result is mock_custom

    def test_raises_when_not_found(self):
        with patch(
            "inference_core.llm.prompts.get_custom_completion_prompt",
            return_value=None,
        ):
            with pytest.raises(AttributeError):
                get_prompt_template("totally_missing")


class TestGetChatPromptTemplate:
    """get_chat_prompt_template resolves builtin → custom → raises AttributeError."""

    def test_returns_builtin_chat(self):
        result = get_chat_prompt_template("chat")
        assert result is ChatPrompts.CHAT

    def test_falls_back_to_custom(self):
        mock_custom = MagicMock()
        with patch(
            "inference_core.llm.prompts.get_custom_chat_prompt",
            return_value=mock_custom,
        ):
            result = get_chat_prompt_template("nonexistent_name")
        assert result is mock_custom

    def test_raises_when_not_found(self):
        with patch(
            "inference_core.llm.prompts.get_custom_chat_prompt",
            return_value=None,
        ):
            with pytest.raises(AttributeError):
                get_chat_prompt_template("totally_missing")
