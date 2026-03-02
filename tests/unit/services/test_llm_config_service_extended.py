"""Extended tests for LLMConfigService.

Covers cache helpers, user preference validation, CRUD for allowed overrides,
effective model params resolution, get_available_options, and
get_config_with_overrides.

Complements the existing test_llm_config_service.py which covers admin CRUD
and basic resolved-config merge logic.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from inference_core.database.sql.models import (
    AllowedUserOverride,
    LLMConfigOverride,
    UserLLMPreference,
)
from inference_core.schemas.llm_config import PreferenceTypeEnum
from inference_core.services.llm_config_service import (
    ConfigValidationError,
    LLMConfigService,
)

# ---------------------------------------------------------------------------
# Fixtures (same pattern as existing test file)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_session():
    """Async session mock with default empty result set."""
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_result.scalars.return_value.all.return_value = []
    session.execute.return_value = mock_result
    session.add = MagicMock()
    return session


@pytest.fixture
def config_service(mock_db_session):
    """LLMConfigService with mocked Redis and DB."""
    with patch(
        "inference_core.services.llm_config_service.get_redis"
    ) as mock_get_redis:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # No cache hits by default
        mock_get_redis.return_value = mock_redis
        service = LLMConfigService(mock_db_session)
        service.redis = mock_redis
        yield service


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


class TestCacheHelpers:
    """Verify Redis cache key building and get/set/invalidate."""

    def test_cache_key_builds_from_parts(self, config_service):
        """_cache_key joins prefix with parts using colons."""
        key = config_service._cache_key("a", "b", "c")
        assert key == "llm_config:a:b:c"

    def test_user_cache_key(self, config_service):
        """_user_cache_key includes 'resolved' and stringified UUID."""
        uid = uuid4()
        key = config_service._user_cache_key(uid)
        assert "resolved" in key
        assert str(uid) in key

    @pytest.mark.asyncio
    async def test_get_cached_returns_parsed_json(self, config_service):
        """_get_cached parses JSON string from Redis."""
        config_service.redis.get.return_value = '{"foo": "bar"}'
        result = await config_service._get_cached("some_key")
        assert result == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_get_cached_returns_none_on_miss(self, config_service):
        """_get_cached returns None when Redis returns None."""
        config_service.redis.get.return_value = None
        result = await config_service._get_cached("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_swallows_redis_error(self, config_service):
        """_get_cached returns None on Redis exception (not crash)."""
        config_service.redis.get.side_effect = ConnectionError("down")
        result = await config_service._get_cached("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_cached_calls_setex(self, config_service):
        """_set_cached serializes to JSON and sets with TTL."""
        await config_service._set_cached("mykey", {"x": 1}, ttl=120)
        config_service.redis.setex.assert_called_once_with("mykey", 120, '{"x": 1}')

    @pytest.mark.asyncio
    async def test_set_cached_uses_default_ttl(self, config_service):
        """_set_cached uses CACHE_TTL_SECONDS when ttl is None."""
        await config_service._set_cached("k", {"v": 2})
        _, args, _ = config_service.redis.setex.mock_calls[0]
        assert args[1] == config_service.CACHE_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_set_cached_swallows_redis_error(self, config_service):
        """_set_cached does not raise on Redis failure."""
        config_service.redis.setex.side_effect = ConnectionError("down")
        # Should not raise
        await config_service._set_cached("k", {"v": 1})

    @pytest.mark.asyncio
    async def test_invalidate_user_cache(self, config_service):
        """_invalidate_user_cache deletes the user's resolved key."""
        uid = uuid4()
        await config_service._invalidate_user_cache(uid)
        config_service.redis.delete.assert_called_once_with(
            config_service._user_cache_key(uid)
        )

    @pytest.mark.asyncio
    async def test_invalidate_admin_cache(self, config_service):
        """_invalidate_admin_cache deletes admin key + all resolved:* keys."""

        # Mock scan_iter to return some user keys
        async def mock_scan(*args, **kwargs):
            for key in [b"llm_config:resolved:user1", b"llm_config:resolved:user2"]:
                yield key

        config_service.redis.scan_iter = mock_scan

        await config_service._invalidate_admin_cache()

        # Should delete admin cache key AND the scanned resolved keys
        assert config_service.redis.delete.call_count >= 2


# ---------------------------------------------------------------------------
# User preference validation
# ---------------------------------------------------------------------------


class TestValidateUserPreference:
    """Verify _validate_user_preference against allowlist and constraints."""

    @pytest.mark.asyncio
    async def test_rejects_key_not_in_allowlist(self, config_service):
        """Non-allowlisted keys raise ConfigValidationError."""
        with patch.object(config_service, "get_allowed_overrides", return_value=[]):
            with pytest.raises(ConfigValidationError, match="not user-overridable"):
                await config_service._validate_user_preference(
                    "secret_param", {"value": 42}
                )

    @pytest.mark.asyncio
    async def test_passes_when_no_constraints(self, config_service):
        """Allowed key with no constraints passes validation."""
        mock_allowed = MagicMock(spec=AllowedUserOverride)
        mock_allowed.config_key = "temperature"
        mock_allowed.constraints = None

        with patch.object(
            config_service, "get_allowed_overrides", return_value=[mock_allowed]
        ):
            # Should not raise
            await config_service._validate_user_preference(
                "temperature", {"value": 0.5}
            )

    @pytest.mark.asyncio
    async def test_rejects_number_below_min(self, config_service):
        """Number value below constraint min raises error."""
        mock_allowed = MagicMock(spec=AllowedUserOverride)
        mock_allowed.config_key = "temperature"
        mock_allowed.constraints = {"type": "number", "min": 0.0, "max": 2.0}

        with patch.object(
            config_service, "get_allowed_overrides", return_value=[mock_allowed]
        ):
            with pytest.raises(ConfigValidationError, match="below minimum"):
                await config_service._validate_user_preference(
                    "temperature", {"value": -0.1}
                )

    @pytest.mark.asyncio
    async def test_rejects_number_above_max(self, config_service):
        """Number value above constraint max raises error."""
        mock_allowed = MagicMock(spec=AllowedUserOverride)
        mock_allowed.config_key = "max_tokens"
        mock_allowed.constraints = {"type": "number", "min": 1, "max": 8000}

        with patch.object(
            config_service, "get_allowed_overrides", return_value=[mock_allowed]
        ):
            with pytest.raises(ConfigValidationError, match="above maximum"):
                await config_service._validate_user_preference(
                    "max_tokens", {"value": 9999}
                )

    @pytest.mark.asyncio
    async def test_passes_number_within_range(self, config_service):
        """Number value within range passes."""
        mock_allowed = MagicMock(spec=AllowedUserOverride)
        mock_allowed.config_key = "temperature"
        mock_allowed.constraints = {"type": "number", "min": 0.0, "max": 2.0}

        with patch.object(
            config_service, "get_allowed_overrides", return_value=[mock_allowed]
        ):
            await config_service._validate_user_preference(
                "temperature", {"value": 1.0}
            )

    @pytest.mark.asyncio
    async def test_rejects_string_not_in_allowed_values(self, config_service):
        """String value not in allowed_values raises error."""
        mock_allowed = MagicMock(spec=AllowedUserOverride)
        mock_allowed.config_key = "style"
        mock_allowed.constraints = {
            "type": "string",
            "allowed_values": ["formal", "casual"],
        }

        with patch.object(
            config_service, "get_allowed_overrides", return_value=[mock_allowed]
        ):
            with pytest.raises(ConfigValidationError, match="not in allowed values"):
                await config_service._validate_user_preference(
                    "style", {"value": "pirate"}
                )

    @pytest.mark.asyncio
    async def test_rejects_string_exceeding_max_length(self, config_service):
        """String value exceeding max_length raises error."""
        mock_allowed = MagicMock(spec=AllowedUserOverride)
        mock_allowed.config_key = "prompt"
        mock_allowed.constraints = {"type": "string", "max_length": 5}

        with patch.object(
            config_service, "get_allowed_overrides", return_value=[mock_allowed]
        ):
            with pytest.raises(ConfigValidationError, match="exceeds max length"):
                await config_service._validate_user_preference(
                    "prompt", {"value": "way too long"}
                )

    @pytest.mark.asyncio
    async def test_rejects_select_not_in_options(self, config_service):
        """Select value not in allowed options raises error."""
        mock_allowed = MagicMock(spec=AllowedUserOverride)
        mock_allowed.config_key = "default_model"
        mock_allowed.constraints = {
            "type": "select",
            "allowed_values": ["gpt-4", "gpt-3.5-turbo"],
        }

        with patch.object(
            config_service, "get_allowed_overrides", return_value=[mock_allowed]
        ):
            with pytest.raises(ConfigValidationError, match="not in allowed options"):
                await config_service._validate_user_preference(
                    "default_model", {"value": "claude-3"}
                )


# ---------------------------------------------------------------------------
# create_user_preference (upsert)
# ---------------------------------------------------------------------------


class TestCreateUserPreference:
    """Verify user preference creation and upsert logic."""

    @pytest.mark.asyncio
    async def test_creates_new_preference(self, config_service):
        """New preference is added to DB when none exists."""
        with patch.object(config_service, "_validate_user_preference") as mock_validate:
            uid = uuid4()
            pref = await config_service.create_user_preference(
                user_id=uid,
                preference_type=PreferenceTypeEnum.MODEL_PARAMS,
                preference_key="temperature",
                preference_value={"value": 0.8},
            )

            mock_validate.assert_called_once()
            config_service.db.add.assert_called_once()
            config_service.db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_updates_existing_preference(self, config_service):
        """Existing preference is updated in place (upsert)."""
        existing = MagicMock(spec=UserLLMPreference)
        existing.preference_value = {"value": 0.5}

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        config_service.db.execute.return_value = mock_result

        with patch.object(config_service, "_validate_user_preference"):
            uid = uuid4()
            result = await config_service.create_user_preference(
                user_id=uid,
                preference_type=PreferenceTypeEnum.MODEL_PARAMS,
                preference_key="temperature",
                preference_value={"value": 0.9},
            )

        assert existing.preference_value == {"value": 0.9}
        assert existing.is_active is True


# ---------------------------------------------------------------------------
# delete_user_preference
# ---------------------------------------------------------------------------


class TestDeleteUserPreference:
    """Verify user preference deletion."""

    @pytest.mark.asyncio
    async def test_deletes_existing(self, config_service):
        """Returns True and deletes when preference exists."""
        existing = MagicMock(spec=UserLLMPreference)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        config_service.db.execute.return_value = mock_result

        result = await config_service.delete_user_preference(uuid4(), "temperature")

        assert result is True
        config_service.db.delete.assert_called_once_with(existing)

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self, config_service):
        """Returns False when preference does not exist."""
        result = await config_service.delete_user_preference(uuid4(), "nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# create_allowed_override
# ---------------------------------------------------------------------------


class TestCreateAllowedOverride:
    """Verify allowed override creation for admin allowlist."""

    @pytest.mark.asyncio
    async def test_creates_new_allowed_override(self, config_service):
        """Creates AllowedUserOverride with correct fields."""
        result = await config_service.create_allowed_override(
            config_key="temperature",
            constraints={"type": "number", "min": 0, "max": 2},
            allowed_scopes=["global"],
            display_name="Temperature",
            description="Controls randomness",
        )

        config_service.db.add.assert_called_once()
        config_service.db.commit.assert_called()
        # Should invalidate allowed overrides cache
        config_service.redis.delete.assert_called()


# ---------------------------------------------------------------------------
# update_admin_override
# ---------------------------------------------------------------------------


class TestUpdateAdminOverride:
    """Verify admin override update logic."""

    @pytest.mark.asyncio
    async def test_updates_found_override(self, config_service):
        """Returns updated override when found."""
        existing = MagicMock(spec=LLMConfigOverride)
        existing.config_value = {"value": 0.5}
        existing.description = "old"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        config_service.db.execute.return_value = mock_result

        result = await config_service.update_admin_override(
            override_id=uuid4(),
            config_value={"value": 0.8},
            description="updated",
        )

        assert result is existing
        config_service.db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, config_service):
        """Returns None when override ID does not exist."""
        result = await config_service.update_admin_override(override_id=uuid4())
        assert result is None


# ---------------------------------------------------------------------------
# delete_admin_override
# ---------------------------------------------------------------------------


class TestDeleteAdminOverride:
    """Verify admin override deletion."""

    @pytest.mark.asyncio
    async def test_deletes_existing(self, config_service):
        """Returns True and hard-deletes when found."""
        existing = MagicMock(spec=LLMConfigOverride)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        config_service.db.execute.return_value = mock_result

        result = await config_service.delete_admin_override(uuid4())

        assert result is True
        config_service.db.delete.assert_called_once_with(existing)

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self, config_service):
        """Returns False when override does not exist."""
        result = await config_service.delete_admin_override(uuid4())
        assert result is False


# ---------------------------------------------------------------------------
# get_effective_model_params
# ---------------------------------------------------------------------------


class TestGetEffectiveModelParams:
    """Verify parameter resolution for actual model calls."""

    @pytest.mark.asyncio
    async def test_returns_empty_for_unknown_model(self, config_service):
        """Returns empty dict when model is not in config."""
        result = await config_service.get_effective_model_params(
            user_id=uuid4(),
            model_name="nonexistent-model",
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_yaml_defaults(self, config_service):
        """Returns YAML base defaults when no overrides exist."""
        mock_model = MagicMock()
        mock_model.temperature = 0.7
        mock_model.max_tokens = 2048
        config_service._base_config.models = {"gpt-4": mock_model}

        with patch.object(
            config_service,
            "_load_admin_overrides_dict",
            return_value={"global": {}, "model": {}, "task": {}, "agent": {}},
        ), patch.object(
            config_service,
            "_load_user_preferences_dict",
            return_value={"model_params": {}, "task_params": {}},
        ):
            result = await config_service.get_effective_model_params(
                user_id=uuid4(),
                model_name="gpt-4",
            )

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_admin_override_wins_over_yaml(self, config_service):
        """Admin model-specific override overrides YAML default."""
        mock_model = MagicMock()
        mock_model.temperature = 0.7
        mock_model.max_tokens = 2048
        config_service._base_config.models = {"gpt-4": mock_model}

        admin_overrides = {
            "global": {},
            "model": {"gpt-4": {"temperature": {"value": 0.3}}},
            "task": {},
            "agent": {},
        }

        with patch.object(
            config_service,
            "_load_admin_overrides_dict",
            return_value=admin_overrides,
        ), patch.object(
            config_service,
            "_load_user_preferences_dict",
            return_value={"model_params": {}, "task_params": {}},
        ):
            result = await config_service.get_effective_model_params(
                user_id=uuid4(),
                model_name="gpt-4",
            )

        assert result["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_user_pref_wins_over_admin(self, config_service):
        """User-specific model param overrides admin value."""
        mock_model = MagicMock()
        mock_model.temperature = 0.7
        mock_model.max_tokens = 2048
        config_service._base_config.models = {"gpt-4": mock_model}

        admin_overrides = {
            "global": {},
            "model": {"gpt-4": {"temperature": {"value": 0.3}}},
            "task": {},
            "agent": {},
        }

        user_prefs = {
            "model_params": {"gpt-4.temperature": {"value": 1.5}},
            "task_params": {},
        }

        with patch.object(
            config_service,
            "_load_admin_overrides_dict",
            return_value=admin_overrides,
        ), patch.object(
            config_service,
            "_load_user_preferences_dict",
            return_value=user_prefs,
        ):
            result = await config_service.get_effective_model_params(
                user_id=uuid4(),
                model_name="gpt-4",
            )

        assert result["temperature"] == 1.5

    @pytest.mark.asyncio
    async def test_anonymous_user_skips_user_prefs(self, config_service):
        """When user_id is None, user preferences are not loaded."""
        mock_model = MagicMock()
        mock_model.temperature = 0.7
        mock_model.max_tokens = 2048
        config_service._base_config.models = {"gpt-4": mock_model}

        with patch.object(
            config_service,
            "_load_admin_overrides_dict",
            return_value={"global": {}, "model": {}, "task": {}, "agent": {}},
        ):
            result = await config_service.get_effective_model_params(
                user_id=None,
                model_name="gpt-4",
            )

        assert result["temperature"] == 0.7


# ---------------------------------------------------------------------------
# get_config_with_overrides
# ---------------------------------------------------------------------------


class TestGetConfigWithOverrides:
    """Verify full LLMConfig reconstruction with all override layers."""

    @pytest.mark.asyncio
    async def test_calls_with_overrides_on_base(self, config_service):
        """Calls _base_config.with_overrides with admin overrides."""
        admin = {"global": {}, "model": {}, "task": {}, "agent": {}}
        mock_config = MagicMock()
        config_service._base_config = mock_config

        with patch.object(
            config_service, "_load_admin_overrides_dict", return_value=admin
        ):
            await config_service.get_config_with_overrides(user_id=None)

        mock_config.with_overrides.assert_called_once()

    @pytest.mark.asyncio
    async def test_merges_user_agent_params(self, config_service):
        """User agent_params are merged into admin overrides before with_overrides."""
        admin = {"global": {}, "model": {}, "task": {}, "agent": {}}
        user_prefs = {
            "agent_params": {
                "my_agent.allowed_tools": {"value": ["tool_a", "tool_b"]},
            },
            "model_params": {},
            "default_model": {},
            "task_params": {},
        }

        mock_config = MagicMock()
        config_service._base_config = mock_config

        uid = uuid4()
        with patch.object(
            config_service, "_load_admin_overrides_dict", return_value=admin
        ), patch.object(
            config_service, "_load_user_preferences_dict", return_value=user_prefs
        ):
            await config_service.get_config_with_overrides(user_id=uid)

        # with_overrides should receive agent overrides containing user's tools
        call_kwargs = mock_config.with_overrides.call_args
        agent_overrides = (
            call_kwargs[1].get("agent_overrides") or call_kwargs[0][2]
            if call_kwargs[0]
            else call_kwargs[1]["agent_overrides"]
        )
        assert "my_agent" in agent_overrides
        assert agent_overrides["my_agent"]["allowed_tools"] == ["tool_a", "tool_b"]

    @pytest.mark.asyncio
    async def test_normalises_admin_override_value_wrappers(self, config_service):
        """Admin DB overrides stored as {'value': X} are unwrapped before with_overrides."""
        admin = {
            "global": {},
            "model": {
                "gpt-5": {"temperature": {"value": 0.3}, "max_tokens": {"value": 2048}}
            },
            "task": {},
            "agent": {
                "my_agent": {
                    "primary": {"value": "gpt-5"},
                    "allowed_tools": {"value": ["search"]},
                }
            },
        }

        mock_config = MagicMock()
        config_service._base_config = mock_config

        with patch.object(
            config_service, "_load_admin_overrides_dict", return_value=admin
        ):
            await config_service.get_config_with_overrides(user_id=None)

        call_kwargs = mock_config.with_overrides.call_args[1]

        # Model overrides should be unwrapped
        model_ov = call_kwargs.get("model_overrides", {})
        assert model_ov["gpt-5"]["temperature"] == 0.3
        assert model_ov["gpt-5"]["max_tokens"] == 2048

        # Agent overrides should be unwrapped (primary is a string, not a dict)
        agent_ov = call_kwargs.get("agent_overrides", {})
        assert agent_ov["my_agent"]["primary"] == "gpt-5"
        assert agent_ov["my_agent"]["allowed_tools"] == ["search"]

    @pytest.mark.asyncio
    async def test_global_overrides_not_passed_to_with_overrides(self, config_service):
        """Admin global overrides are NOT forwarded to with_overrides to prevent models corruption."""
        admin = {
            "global": {
                "models": {"value": {"gpt-5": {}}},
                "disabled_models": {"value": ["gpt-4"]},
            },
            "model": {},
            "task": {},
            "agent": {},
        }

        mock_config = MagicMock()
        config_service._base_config = mock_config

        with patch.object(
            config_service, "_load_admin_overrides_dict", return_value=admin
        ):
            await config_service.get_config_with_overrides(user_id=None)

        call_kwargs = mock_config.with_overrides.call_args[1]
        assert "global_overrides" not in call_kwargs


# ---------------------------------------------------------------------------
# ConfigValidationError
# ---------------------------------------------------------------------------


class TestConfigValidationError:
    """Verify custom exception attributes."""

    def test_stores_key_and_message(self):
        err = ConfigValidationError(
            key="temperature",
            message="too high",
            constraint="max",
        )
        assert err.key == "temperature"
        assert err.constraint == "max"
        assert "temperature" in str(err)
        assert "too high" in str(err)

    def test_constraint_is_optional(self):
        err = ConfigValidationError(key="k", message="bad")
        assert err.constraint is None
