from ..services.agents_service import AgentService
from .tools import internet_search


def create_browser_agent(agent_name: str = "internet_browser") -> AgentService:
    """Create an AgentService with internet search tool included."""
    return AgentService(agent_name=agent_name, tools=[internet_search])
