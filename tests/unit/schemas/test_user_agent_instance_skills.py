import pytest
from pydantic import ValidationError

from inference_core.schemas.user_agent_instance import (
    AgentInstanceCreate,
    AgentInstanceUpdate,
)


def _create_payload(skills):
    return {
        "instance_name": "currency-agent",
        "display_name": "Currency Agent",
        "base_agent_name": "deep_agent",
        "skills": skills,
    }


def test_agent_instance_create_accepts_valid_skill() -> None:
    model = AgentInstanceCreate(
        **_create_payload(
            [
                {
                    "name": "monitor-currency-rates",
                    "description": "Monitor public currency rates",
                    "content": "---\nname: monitor-currency-rates\n---\n",
                }
            ]
        )
    )

    assert model.skills[0]["name"] == "monitor-currency-rates"


@pytest.mark.parametrize(
    "skills",
    [
        [
            {
                "name": "Invalid_Name",
                "description": "Bad name",
                "content": "content",
            }
        ],
        [
            {
                "name": "duplicate-skill",
                "description": "First",
                "content": "content",
            },
            {
                "name": "duplicate-skill",
                "description": "Second",
                "content": "content",
            },
        ],
        [
            {
                "name": "too-long",
                "description": "x" * 1025,
                "content": "content",
            }
        ],
    ],
)
def test_agent_instance_create_rejects_unsafe_skills(skills) -> None:
    with pytest.raises(ValidationError):
        AgentInstanceCreate(**_create_payload(skills))


def test_agent_instance_update_rejects_unsafe_skill_name() -> None:
    with pytest.raises(ValidationError):
        AgentInstanceUpdate(
            skills=[
                {
                    "name": "-bad",
                    "description": "Bad name",
                    "content": "content",
                }
            ]
        )
