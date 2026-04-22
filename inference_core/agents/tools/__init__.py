"""Tools module exports."""

from .demo_tool_calling import (
    DemoToolCallingProvider,
    calculate,
    get_weather,
    register_demo_tool_calling_provider,
    search_web,
)
from .email_provider import EmailToolsProvider, register_email_tools_provider
from .email_tools import (
    EmailSummary,
    EmailType,
    ListEmailAccountsTool,
    ReadUnseenEmailsTool,
    SearchEmailsTool,
    SendEmailTool,
    SummarizeEmailTool,
    generate_email_tools_system_instructions,
    get_email_tools,
)
from .memory_tools import (
    RecallMemoryStoreTool,
    SaveMemoryStoreTool,
    generate_memory_tools_system_instructions,
    get_memory_tools,
)
from .search_engine import InternetSearchTool, get_search_tools
from .weather_provider import (
    DefaultAgentToolsProvider,
    WeatherToolsProvider,
    check_weather,
    register_default_agent_tools_provider,
    register_weather_tools_provider,
)

__all__ = [
    # Search tools
    "InternetSearchTool",
    "get_search_tools",
    # Memory tools
    "SaveMemoryStoreTool",
    "RecallMemoryStoreTool",
    "get_memory_tools",
    "generate_memory_tools_system_instructions",
    # Email tools
    "ReadUnseenEmailsTool",
    "SearchEmailsTool",
    "SendEmailTool",
    "SummarizeEmailTool",
    "ListEmailAccountsTool",
    "EmailSummary",
    "EmailType",
    "get_email_tools",
    "generate_email_tools_system_instructions",
    # Email provider
    "EmailToolsProvider",
    "register_email_tools_provider",
    # Demo tool-calling (frontend pattern showcase)
    "DemoToolCallingProvider",
    "register_demo_tool_calling_provider",
    "get_weather",
    "calculate",
    "search_web",
    # Weather & default_agent providers
    "check_weather",
    "WeatherToolsProvider",
    "DefaultAgentToolsProvider",
    "register_weather_tools_provider",
    "register_default_agent_tools_provider",
]
