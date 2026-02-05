from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from inference_core.database.sql.models import (
    AllowedUserOverride,
    ConfigScope,
    LLMConfigOverride,
    UserLLMPreference,
)
from inference_core.services.llm_config_service import LLMConfigService


@pytest.fixture
def mock_db_session():
    session = AsyncMock()
    # Mock result scalar/scalars/all methods
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_result.scalars.return_value.all.return_value = []
    session.execute.return_value = mock_result

    # session.add is synchronous
    session.add = MagicMock()

    return session


@pytest.fixture
def config_service(mock_db_session):
    with patch(
        "inference_core.services.llm_config_service.get_redis"
    ) as mock_get_redis:
        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis
        service = LLMConfigService(mock_db_session)
        # Manually set redis mock to ensure it's the same object we can assert on
        service.redis = mock_redis
        yield service


class TestLLMConfigService:

    @pytest.mark.asyncio
    async def test_get_admin_overrides_empty(self, config_service):
        """Test fetching overrides when none exist."""
        result = await config_service.get_admin_overrides()
        assert result == []
        config_service.db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_admin_override(self, config_service):
        """Test creating an admin override."""
        override = await config_service.create_admin_override(
            scope=ConfigScope.GLOBAL,
            config_key="temperature",
            config_value={"value": 0.5},
        )

        assert override.config_key == "temperature"
        assert override.scope == "global"
        config_service.db.add.assert_called_once()
        config_service.db.commit.assert_called_once()
        # Should invalidate cache
        config_service.redis.delete.assert_called()

    @pytest.mark.asyncio
    async def test_load_admin_overrides_dict_structure(self, config_service):
        """Test that admin overrides are correctly structured into dictionary."""
        # Mock database response
        mock_override_global = MagicMock(spec=LLMConfigOverride)
        mock_override_global.scope = "global"
        mock_override_global.scope_key = None
        mock_override_global.config_key = "temperature"
        mock_override_global.config_value = {"value": 0.8}

        mock_override_model = MagicMock(spec=LLMConfigOverride)
        mock_override_model.scope = "model"
        mock_override_model.scope_key = "gpt-4"
        mock_override_model.config_key = "max_tokens"
        mock_override_model.config_value = {"value": 4096}

        # Setup mock return
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            mock_override_global,
            mock_override_model,
        ]
        config_service.db.execute.return_value = mock_result

        # Execute
        result = await config_service._load_admin_overrides_dict()

        # different structure than I thought? Let's check implementation
        # implementation: result["global"][key] = value, result["model"][scope_key][key] = value

        assert result["global"]["temperature"] == {"value": 0.8}
        assert result["model"]["gpt-4"]["max_tokens"] == {"value": 4096}

    @pytest.mark.asyncio
    async def test_get_resolved_config_merge_logic(self, config_service):
        """Test that get_resolved_config merges sources correctly."""
        user_id = uuid4()

        # Mock base config
        base_defaults = {"temperature": 0.7}

        # Mock admin overrides
        admin_overrides = {
            "global": {"temperature": {"value": 0.5}},  # Admin lowers temp
            "model": {},
            "task": {},
            "agent": {},
        }

        # Mock user preferences
        user_prefs = {
            "model_params": {"temperature": {"value": 0.9}},  # User wants high temp
            "default_model": {},
        }

        # Mock methods
        with patch.object(
            config_service, "_load_admin_overrides_dict", return_value=admin_overrides
        ), patch.object(
            config_service, "_load_user_preferences_dict", return_value=user_prefs
        ), patch(
            "inference_core.services.llm_config_service.get_llm_config"
        ) as mock_get_config:

            # Setup mock config
            mock_config_instance = MagicMock()
            mock_config_instance.models = {"gpt-4": MagicMock()}  # Available model
            mock_config_instance.task_configs = {}
            mock_config_instance.agent_configs = {}
            mock_config_instance.is_model_available.return_value = True
            mock_get_config.return_value = mock_config_instance

            # Execute
            resolved = await config_service.get_resolved_config(user_id)

            # Assertions
            # User pref (0.9) should override Admin (0.5) which overrides Base (0.7) for allowed keys
            # Assuming 'temperature' is allowed.
            # Wait, resolved config logic applies directly: defaults[key] = value

            assert resolved.defaults["temperature"] == 0.9
            assert resolved.sources["defaults.temperature"] == "user"
