"""Weather tool and providers for weather_agent and default_agent.

WHY: weather_agent needs a real check_weather tool backed by OpenWeatherMap.
default_agent additionally needs internet_search alongside check_weather.
Both providers live here to avoid duplicating the tool definition.

REQUIREMENTS:
  OPEN_WEATHER_API_KEY — OpenWeatherMap API key (free tier works).
  TAVILY_API_KEY       — required only by DefaultAgentToolsProvider via
                         InternetSearchTool.
"""

import json
import logging
import os
from urllib.parse import quote

import requests
from langchain_core.tools import tool

from inference_core.agents.tools.search_engine import InternetSearchTool
from inference_core.llm.tools import ToolProvider, register_tool_provider

logger = logging.getLogger(__name__)

_OPEN_WEATHER_API_KEY = os.getenv("OPEN_WEATHER_API_KEY")

# Nominatim requires a descriptive User-Agent per usage policy.
_NOMINATIM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:146.0) Gecko/20100101 Firefox/145.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pl,en-US;q=0.7,en;q=0.3",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Connection": "keep-alive",
    "Host": "nominatim.openstreetmap.org",
    "Upgrade-Insecure-Requests": "1",
    "TE": "trailers",
}

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@tool
def check_weather(country: str, city: str) -> str:
    """Check the current weather forecast for a given city.

    Geocodes the location via Nominatim (OpenStreetMap), then fetches a
    5-day / 3-hour forecast from OpenWeatherMap.

    Parameters
    ----------
    country : str
        Country name (e.g., "Poland").
    city : str
        City name (e.g., "Warsaw").

    Returns
    -------
    str
        JSON-encoded forecast data on success, or a JSON-encoded error dict
        like ``{"error": "..."}`` when geocoding or the API call fails.
    """
    if not _OPEN_WEATHER_API_KEY:
        logger.error("OPEN_WEATHER_API_KEY is not set — cannot fetch weather.")
        return json.dumps({"error": "Weather service is not configured."})

    # Step 1: Resolve city name to lat/lon via Nominatim.
    geo_url = (
        f"https://nominatim.openstreetmap.org/search"
        f"?q={quote(city + ', ' + country)}&format=json&limit=1"
    )
    try:
        geo_response = requests.get(geo_url, headers=_NOMINATIM_HEADERS, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
    except Exception as exc:
        logger.warning("Nominatim geocoding failed for %s, %s: %s", city, country, exc)
        return json.dumps({"error": f"Geocoding failed: {exc}"})

    if not geo_data:
        return json.dumps({"error": f"Location not found: {city}, {country}"})

    lat = geo_data[0]["lat"]
    lon = geo_data[0]["lon"]

    # Step 2: Fetch 5-day forecast from OpenWeatherMap.
    weather_url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}&lon={lon}&appid={_OPEN_WEATHER_API_KEY}&units=metric"
    )
    try:
        weather_response = requests.get(weather_url, timeout=10)
        weather_response.raise_for_status()
        return json.dumps(weather_response.json())
    except Exception as exc:
        logger.warning(
            "OpenWeatherMap request failed for %s, %s: %s", city, country, exc
        )
        return json.dumps({"error": f"Weather API failed: {exc}"})


# ---------------------------------------------------------------------------
# Tool providers
# ---------------------------------------------------------------------------


class WeatherToolsProvider(ToolProvider):
    """Provides check_weather for weather_agent.

    WHY: Registered under the name 'weather_agent_tools' so that
    llm_config.yaml's local_tool_providers entry resolves to it.
    """

    name = "weather_agent_tools"

    async def get_tools(self, task_type: str, **kwargs) -> list:
        """Return the weather tool list."""
        return [check_weather]


class DefaultAgentToolsProvider(ToolProvider):
    """Provides check_weather + internet_search for default_agent.

    WHY: default_agent's allowed_tools are ['check_weather', 'internet_search'].
    Both tools are bundled here under 'default_agent_tools'.
    """

    name = "default_agent_tools"

    async def get_tools(self, task_type: str, **kwargs) -> list:
        """Return weather + search tools."""
        return [check_weather, InternetSearchTool()]


# ---------------------------------------------------------------------------
# Registration helpers — called from agent_graphs.py before graph builds.
# ---------------------------------------------------------------------------


def register_weather_tools_provider() -> None:
    """Register WeatherToolsProvider so weather_agent can resolve its tools."""
    register_tool_provider(WeatherToolsProvider())


def register_default_agent_tools_provider() -> None:
    """Register DefaultAgentToolsProvider so default_agent can resolve its tools."""
    register_tool_provider(DefaultAgentToolsProvider())


__all__ = [
    "check_weather",
    "WeatherToolsProvider",
    "DefaultAgentToolsProvider",
    "register_weather_tools_provider",
    "register_default_agent_tools_provider",
]
