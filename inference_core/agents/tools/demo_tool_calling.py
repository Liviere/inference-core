"""
Demo tool-calling tools for the LangChain 'tool-calling' frontend pattern.

WHY: Mirrors the three tools from the LangChain playground example
(`get_weather`, `calculate`, `search_web`) with stable, JSON-serialised
payloads whose shape matches the React cards on the frontend.

These are intentionally *mock* implementations — they fabricate plausible
data without external API calls so the pattern can be demonstrated without
extra credentials. Swap the bodies with real providers (Open-Meteo, sympy,
Tavily…) when graduating beyond the demo.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool: get_weather
# ---------------------------------------------------------------------------
_WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Clear"]


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Returns a JSON string with ``city``, ``temperature`` (Fahrenheit),
    ``condition``, ``unit``, ``humidity`` and ``wind_speed``. The payload
    shape is what the frontend ``WeatherToolCard`` expects.
    """
    # Seed by city so the same query returns deterministic-ish values.
    rng = random.Random(hash(city.lower()) & 0xFFFFFFFF)
    payload = {
        "city": city,
        "temperature": rng.randint(50, 80),
        "condition": rng.choice(_WEATHER_CONDITIONS),
        "unit": "fahrenheit",
        "humidity": rng.randint(30, 90),
        "wind_speed": rng.randint(2, 18),
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Tool: calculate
# ---------------------------------------------------------------------------
_SAFE_EXPR_RE = re.compile(r"^[\d+\-*/().\s]+$")


def _safe_eval(expression: str) -> float:
    """Evaluate *expression* after a strict character whitelist check.

    WHY: ``eval`` is dangerous; the whitelist guarantees we only ever see
    numeric literals and the four basic operators plus parentheses.
    """
    if not _SAFE_EXPR_RE.match(expression):
        raise ValueError(
            f"Invalid expression: {expression!r}. "
            "Only numbers and +, -, *, /, (, ) are allowed."
        )
    # Evaluate with empty globals/locals to prevent name resolution.
    return eval(expression, {"__builtins__": {}}, {})  # noqa: S307


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple mathematical expression.

    Supports +, -, *, / and parentheses. Returns a JSON string with the
    original ``expression`` and either ``result`` on success or ``error``
    on failure. Matches the playload shape expected by ``CalculatorToolCard``.
    """
    try:
        result = _safe_eval(expression)
        return json.dumps({"expression": expression, "result": result})
    except Exception as err:
        return json.dumps({"expression": expression, "error": str(err)})


# ---------------------------------------------------------------------------
# Tool: search_web
# ---------------------------------------------------------------------------


@tool
def search_web(query: str) -> str:
    """Mock web search that returns three fabricated results for *query*.

    Returns a JSON string shaped as ``{"query", "results": [...]}`` where
    each result has ``title``, ``url`` and ``snippet``. This mirrors the
    TypeScript example exactly so the same ``SearchToolCard`` renders
    without modification.
    """
    slug = re.sub(r"\s+", "-", query.strip().lower())
    payload = {
        "query": query,
        "results": [
            {
                "title": f"Getting Started with {query}",
                "url": f"https://example.com/{slug}",
                "snippet": (
                    "A comprehensive guide covering the fundamentals and best "
                    "practices for getting started."
                ),
            },
            {
                "title": f"{query} — Official Documentation",
                "url": f"https://docs.example.com/{slug}",
                "snippet": (
                    "Official reference documentation with detailed API "
                    "specifications and usage examples."
                ),
            },
            {
                "title": f"{query}: Best Practices & Tips",
                "url": f"https://blog.example.com/{slug}",
                "snippet": (
                    "Expert tips and industry best practices compiled from "
                    "real-world production experience."
                ),
            },
        ],
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Pluggable tool provider
# ---------------------------------------------------------------------------


class DemoToolCallingProvider:
    """Provides the three demo tools under the name ``demo_tool_calling``.

    WHY: The LLM service resolves providers by name from
    ``local_tool_providers`` in the agent YAML. Bundling the tools here
    keeps the registration logic next to their definitions and allows
    the provider to be swapped or extended without touching the agent
    config.
    """

    name: str = "demo_tool_calling"

    async def get_tools(self, task_type: str, **kwargs: Any) -> List[Any]:
        """Return the three demo tools regardless of task type."""
        return [get_weather, calculate, search_web]


def register_demo_tool_calling_provider() -> None:
    """Register the demo provider with the global tool registry.

    Must be called **before** ``build_agent_graph("tool_calling_demo")``
    so the graph picks up the tools at compile time.
    """
    from inference_core.llm.tools import register_tool_provider

    register_tool_provider(DemoToolCallingProvider())
    logger.info("Registered DemoToolCallingProvider (demo_tool_calling)")


__all__ = [
    "DemoToolCallingProvider",
    "calculate",
    "get_weather",
    "register_demo_tool_calling_provider",
    "search_web",
]
