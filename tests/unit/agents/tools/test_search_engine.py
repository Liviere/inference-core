"""Tests for InternetSearchTool (Tavily wrapper).

Covers: _run happy path, missing API key, _arun async path,
_format_log with optional params, get_search_tools factory.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestInternetSearchToolRun:
    """Test InternetSearchTool._run with mocked TavilyClient."""

    @patch("inference_core.agents.tools.search_engine._tavily_client_instance", None)
    @patch("inference_core.agents.tools.search_engine.API_KEY", "tvly-test-key")
    def test_run_happy_path(self):
        """_run calls TavilyClient.search with correct params and returns result."""
        with patch("inference_core.agents.tools.search_engine.TavilyClient") as MockClient:
            mock_client = MagicMock()
            mock_client.search.return_value = {
                "results": [{"title": "Example", "url": "https://example.com"}]
            }
            MockClient.return_value = mock_client

            # Re-import to reset singleton
            from inference_core.agents.tools.search_engine import InternetSearchTool

            tool = InternetSearchTool()
            result = tool._run(query="test query", max_results=3, topic="general")

            mock_client.search.assert_called_once_with(
                query="test query",
                max_results=3,
                topic="general",
                include_raw_content=None,
                time_range=None,
                start_date=None,
                end_date=None,
                country=None,
                include_domains=None,
                exclude_domains=None,
            )
            assert result["results"][0]["title"] == "Example"

    @patch("inference_core.agents.tools.search_engine._tavily_client_instance", None)
    @patch("inference_core.agents.tools.search_engine.API_KEY", None)
    def test_run_missing_api_key_raises(self):
        """_run raises ValueError when TAVILY_API_KEY is not set."""
        from inference_core.agents.tools.search_engine import InternetSearchTool

        tool = InternetSearchTool()
        with pytest.raises(ValueError, match="TAVILY_API_KEY"):
            tool._run(query="test")


class TestInternetSearchToolArun:
    """Test async path of InternetSearchTool."""

    @patch("inference_core.agents.tools.search_engine._tavily_client_instance", None)
    @patch("inference_core.agents.tools.search_engine.API_KEY", "tvly-key")
    async def test_arun_delegates_to_run(self):
        """_arun executes _run via thread executor and returns result."""
        with patch("inference_core.agents.tools.search_engine.TavilyClient") as MockClient:
            mock_client = MagicMock()
            mock_client.search.return_value = {"results": [{"title": "Async"}]}
            MockClient.return_value = mock_client

            from inference_core.agents.tools.search_engine import InternetSearchTool

            tool = InternetSearchTool()
            result = await tool._arun(query="async test")

            assert result["results"][0]["title"] == "Async"
            mock_client.search.assert_called_once()


class TestFormatLog:
    """Test _format_log static method."""

    def test_basic_log(self):
        """Basic log with just required params."""
        from inference_core.agents.tools.search_engine import InternetSearchTool

        log = InternetSearchTool._format_log(
            "hello", 5, "general", None, None, None, None, None, None, None
        )
        assert "hello" in log
        assert "max_results: 5" in log

    def test_optional_params_included(self):
        """Optional params appear in log when set."""
        from inference_core.agents.tools.search_engine import InternetSearchTool

        log = InternetSearchTool._format_log(
            "query", 5, "news", "markdown", "week",
            "2026-01-01", "2026-02-01", "US",
            ["example.com"], ["spam.com"],
        )
        assert "include_raw_content: markdown" in log
        assert "time_range: week" in log
        assert "country: US" in log
        assert "include_domains: example.com" in log
        assert "exclude_domains: spam.com" in log


class TestGetSearchTools:
    """Test get_search_tools factory."""

    def test_returns_list_with_one_tool(self):
        """Factory returns a list containing one InternetSearchTool."""
        from inference_core.agents.tools.search_engine import (
            InternetSearchTool,
            get_search_tools,
        )

        tools = get_search_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], InternetSearchTool)
