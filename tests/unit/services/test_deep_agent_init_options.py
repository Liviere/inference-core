"""Options of ``AgentService`` that ``DeepAgentService`` takes as well.

A caller builds either service with the same history options, so a deep
agent must take them and hand them to ``AgentService``; an option it does not
know makes its construction fail with a ``TypeError``.
"""

from unittest.mock import patch

from inference_core.services.agents_service import AgentService, DeepAgentService


def _forwarded(**kwargs):
    captured = {}

    def fake_init(self, *args, **init_kwargs):
        captured.update(init_kwargs)
        # What DeepAgentService reads after super().__init__().
        self.agent_config = None

    with patch.object(AgentService, "__init__", fake_init):
        DeepAgentService("deep_planner", **kwargs)
    return captured


def test_the_history_options_reach_agent_service():
    captured = _forwarded(
        strip_foreign_thinking=True, memory_context_placement="message"
    )

    assert captured["strip_foreign_thinking"] is True
    assert captured["memory_context_placement"] == "message"


def test_the_history_options_keep_agent_service_defaults():
    captured = _forwarded()

    assert captured["strip_foreign_thinking"] is False
    assert captured["memory_context_placement"] == "system_prompt"
