"""
LLM Configuration Service

Service for managing dynamic LLM configuration with multi-layer resolution:
YAML base → Admin DB overrides → User preferences.

Uses Redis for caching resolved configurations to minimize DB queries.
"""

import copy
import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.core.config import get_settings
from inference_core.core.redis_client import get_redis
from inference_core.database.sql.models import (
    AllowedUserOverride,
    ConfigScope,
    LLMConfigOverride,
    UserLLMPreference,
)
from inference_core.llm.config import LLMConfig, get_llm_config
from inference_core.schemas.llm_config import (
    AllowedOverrideResponse,
    AvailableOptionsResponse,
    ConfigScopeEnum,
    PreferenceTypeEnum,
    ResolvedAgentConfig,
    ResolvedConfigResponse,
    ResolvedTaskConfig,
)

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, key: str, message: str, constraint: Optional[str] = None):
        self.key = key
        self.message = message
        self.constraint = constraint
        super().__init__(f"Validation failed for '{key}': {message}")


class LLMConfigService:
    """
    Service for managing dynamic LLM configuration.

    WHY this exists:
    - Enables runtime config changes without server restart
    - Allows user personalization within admin-defined boundaries
    - Provides caching layer to minimize DB load

    Resolution order (later wins):
    1. YAML base config (llm_config.yaml)
    2. Admin DB overrides (LLMConfigOverride)
    3. User preferences (UserLLMPreference)
    """

    # Redis cache settings
    CACHE_PREFIX = "llm_config:"
    CACHE_TTL_SECONDS = 60  # 1 minute default TTL
    ADMIN_OVERRIDES_CACHE_KEY = "admin_overrides"
    ALLOWED_OVERRIDES_CACHE_KEY = "allowed_overrides"

    def __init__(self, db: AsyncSession):
        self.db = db
        self.redis = get_redis()
        self.settings = get_settings()
        self._base_config = get_llm_config()

    # =========================================================
    # Cache Helpers
    # =========================================================

    def _cache_key(self, *parts: str) -> str:
        """Build a cache key from parts."""
        return f"{self.CACHE_PREFIX}{':'.join(parts)}"

    def _user_cache_key(self, user_id: UUID) -> str:
        """Build cache key for user-specific resolved config."""
        return self._cache_key("resolved", str(user_id))

    async def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached JSON value."""
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis cache get failed: {e}")
        return None

    async def _set_cached(
        self, key: str, value: Dict[str, Any], ttl: Optional[int] = None
    ) -> None:
        """Set cached JSON value with TTL."""
        try:
            ttl = ttl or self.CACHE_TTL_SECONDS
            await self.redis.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning(f"Redis cache set failed: {e}")

    async def _invalidate_user_cache(self, user_id: UUID) -> None:
        """Invalidate cached config for a specific user."""
        try:
            await self.redis.delete(self._user_cache_key(user_id))
        except Exception as e:
            logger.warning(f"Redis cache invalidation failed: {e}")

    async def _invalidate_admin_cache(self) -> None:
        """Invalidate admin overrides cache (affects all users)."""
        try:
            # Delete admin cache
            await self.redis.delete(self._cache_key(self.ADMIN_OVERRIDES_CACHE_KEY))
            # Also delete all resolved configs (pattern delete)
            keys = []
            async for key in self.redis.scan_iter(
                match=f"{self.CACHE_PREFIX}resolved:*"
            ):
                keys.append(key)
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis admin cache invalidation failed: {e}")

    # =========================================================
    # Admin Overrides (DB CRUD)
    # =========================================================

    async def get_admin_overrides(
        self,
        scope: Optional[ConfigScopeEnum] = None,
        scope_key: Optional[str] = None,
        active_only: bool = True,
    ) -> List[LLMConfigOverride]:
        """
        Fetch admin configuration overrides from database.

        Args:
            scope: Filter by scope (global, model, task, agent)
            scope_key: Filter by scope key (e.g., model name)
            active_only: Only return active, non-expired overrides
        """
        conditions = []

        if active_only:
            conditions.append(LLMConfigOverride.is_active == True)
            # Exclude expired overrides
            conditions.append(
                or_(
                    LLMConfigOverride.expires_at.is_(None),
                    LLMConfigOverride.expires_at > datetime.now(UTC),
                )
            )

        if scope:
            conditions.append(LLMConfigOverride.scope == scope.value)
        if scope_key:
            conditions.append(LLMConfigOverride.scope_key == scope_key)

        stmt = (
            select(LLMConfigOverride)
            .where(and_(*conditions))
            .order_by(LLMConfigOverride.priority.desc())
        )
        query_debug = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def create_admin_override(
        self,
        scope: ConfigScopeEnum,
        config_key: str,
        config_value: Dict[str, Any],
        scope_key: Optional[str] = None,
        priority: int = 0,
        description: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        created_by_id: Optional[UUID] = None,
    ) -> LLMConfigOverride:
        """Create a new admin configuration override."""
        override = LLMConfigOverride(
            scope=scope.value,
            scope_key=scope_key,
            config_key=config_key,
            config_value=config_value,
            priority=priority,
            description=description,
            expires_at=expires_at,
            created_by_id=created_by_id,
            is_active=True,
        )
        self.db.add(override)
        await self.db.commit()
        await self.db.refresh(override)
        await self._invalidate_admin_cache()
        return override

    async def update_admin_override(
        self,
        override_id: UUID,
        **updates: Any,
    ) -> Optional[LLMConfigOverride]:
        """Update an existing admin override."""
        stmt = select(LLMConfigOverride).where(
            LLMConfigOverride.id == override_id,
        )
        result = await self.db.execute(stmt)
        override = result.scalar_one_or_none()

        if not override:
            return None

        for key, value in updates.items():
            if hasattr(override, key) and value is not None:
                setattr(override, key, value)

        await self.db.commit()
        await self.db.refresh(override)
        await self._invalidate_admin_cache()
        return override

    async def delete_admin_override(self, override_id: UUID) -> bool:
        """Hard-delete an admin override."""
        stmt = select(LLMConfigOverride).where(
            LLMConfigOverride.id == override_id,
        )
        result = await self.db.execute(stmt)
        override = result.scalar_one_or_none()

        if not override:
            return False

        await self.db.delete(override)
        await self.db.commit()
        await self._invalidate_admin_cache()
        return True

    # =========================================================
    # User Preferences (DB CRUD)
    # =========================================================

    async def get_user_preferences(
        self,
        user_id: UUID,
        preference_type: Optional[PreferenceTypeEnum] = None,
        active_only: bool = True,
    ) -> List[UserLLMPreference]:
        """Fetch user LLM preferences from database."""
        conditions = [
            UserLLMPreference.user_id == user_id,
        ]

        if active_only:
            conditions.append(UserLLMPreference.is_active == True)
        if preference_type:
            conditions.append(
                UserLLMPreference.preference_type == preference_type.value
            )

        stmt = select(UserLLMPreference).where(and_(*conditions))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_allowed_overrides(
        self, active_only: bool = True
    ) -> List[AllowedUserOverride]:
        """Fetch list of configuration keys users are allowed to override."""
        conditions = []
        if active_only:
            conditions.append(AllowedUserOverride.is_active == True)

        stmt = select(AllowedUserOverride).where(and_(*conditions))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def _validate_user_preference(
        self,
        config_key: str,
        value: Dict[str, Any],
    ) -> None:
        """
        Validate user preference against allowed overrides.

        WHY: Security boundary - users can only override keys in the allowlist,
        and values must satisfy constraints (min/max, allowed_values, etc.)
        """
        allowed = await self.get_allowed_overrides()
        allowed_keys = {a.config_key: a for a in allowed}

        if config_key not in allowed_keys:
            raise ConfigValidationError(
                key=config_key,
                message=f"Configuration key '{config_key}' is not user-overridable",
                constraint="allowlist",
            )

        allowed_def = allowed_keys[config_key]
        constraints = allowed_def.constraints

        if not constraints:
            return  # No constraints to validate

        # Extract the actual value from the dict wrapper
        actual_value = value.get("value", value)

        # Validate numeric constraints
        if constraints.get("type") == "number" and isinstance(
            actual_value, (int, float)
        ):
            if "min" in constraints and actual_value < constraints["min"]:
                raise ConfigValidationError(
                    key=config_key,
                    message=f"Value {actual_value} is below minimum {constraints['min']}",
                    constraint="min",
                )
            if "max" in constraints and actual_value > constraints["max"]:
                raise ConfigValidationError(
                    key=config_key,
                    message=f"Value {actual_value} is above maximum {constraints['max']}",
                    constraint="max",
                )

        # Validate string constraints
        if constraints.get("type") == "string" and isinstance(actual_value, str):
            if "allowed_values" in constraints:
                if actual_value not in constraints["allowed_values"]:
                    raise ConfigValidationError(
                        key=config_key,
                        message=f"Value '{actual_value}' not in allowed values",
                        constraint="allowed_values",
                    )
            if "max_length" in constraints:
                if len(actual_value) > constraints["max_length"]:
                    raise ConfigValidationError(
                        key=config_key,
                        message=f"Value exceeds max length {constraints['max_length']}",
                        constraint="max_length",
                    )

        # Validate select/enum constraints
        if constraints.get("type") == "select":
            if "allowed_values" in constraints:
                if actual_value not in constraints["allowed_values"]:
                    raise ConfigValidationError(
                        key=config_key,
                        message=f"Value '{actual_value}' not in allowed options",
                        constraint="allowed_values",
                    )

    async def create_user_preference(
        self,
        user_id: UUID,
        preference_type: PreferenceTypeEnum,
        preference_key: str,
        preference_value: Dict[str, Any],
    ) -> UserLLMPreference:
        """Create or update a user preference."""
        # Extract config_key for validation (e.g., "chat.temperature" -> "temperature")
        config_key = (
            preference_key.split(".")[-1] if "." in preference_key else preference_key
        )
        await self._validate_user_preference(config_key, preference_value)

        # Check if preference already exists (upsert logic)
        stmt = select(UserLLMPreference).where(
            UserLLMPreference.user_id == user_id,
            UserLLMPreference.preference_type == preference_type.value,
            UserLLMPreference.preference_key == preference_key,
        )
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.preference_value = preference_value
            existing.is_active = True
            await self.db.commit()
            await self.db.refresh(existing)
            await self._invalidate_user_cache(user_id)
            return existing

        preference = UserLLMPreference(
            user_id=user_id,
            preference_type=preference_type.value,
            preference_key=preference_key,
            preference_value=preference_value,
            is_active=True,
        )
        self.db.add(preference)
        await self.db.commit()
        await self.db.refresh(preference)
        await self._invalidate_user_cache(user_id)
        return preference

    async def delete_user_preference(
        self,
        user_id: UUID,
        preference_key: str,
    ) -> bool:
        """Delete a user preference."""
        stmt = select(UserLLMPreference).where(
            UserLLMPreference.user_id == user_id,
            UserLLMPreference.preference_key == preference_key,
        )
        result = await self.db.execute(stmt)
        preference = result.scalar_one_or_none()

        if not preference:
            return False

        await self.db.delete(preference)
        await self.db.commit()
        await self._invalidate_user_cache(user_id)
        return True

    # =========================================================
    # Allowed Overrides Management (Admin only)
    # =========================================================

    async def create_allowed_override(
        self,
        config_key: str,
        constraints: Optional[Dict[str, Any]] = None,
        allowed_scopes: Optional[List[str]] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> AllowedUserOverride:
        """Create a new allowed override definition."""
        allowed = AllowedUserOverride(
            config_key=config_key,
            constraints=constraints,
            allowed_scopes={"scopes": allowed_scopes} if allowed_scopes else None,
            display_name=display_name,
            description=description,
            is_active=True,
        )
        self.db.add(allowed)
        await self.db.commit()
        await self.db.refresh(allowed)
        # Invalidate allowed overrides cache
        await self.redis.delete(self._cache_key(self.ALLOWED_OVERRIDES_CACHE_KEY))
        return allowed

    # =========================================================
    # Configuration Resolution
    # =========================================================

    async def _load_admin_overrides_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Load and structure admin overrides for merging.

        Returns nested dict: {scope: {scope_key: {config_key: value}}}
        """
        # Check cache first
        cache_key = self._cache_key(self.ADMIN_OVERRIDES_CACHE_KEY)
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        overrides = await self.get_admin_overrides(active_only=True)

        result: Dict[str, Dict[str, Any]] = {
            "global": {},
            "model": {},
            "task": {},
            "agent": {},
        }

        for override in overrides:
            scope = override.scope
            key = override.scope_key or "__global__"

            if scope == ConfigScope.GLOBAL.value:
                result["global"][override.config_key] = override.config_value
            else:
                if key not in result[scope]:
                    result[scope][key] = {}
                result[scope][key][override.config_key] = override.config_value

        await self._set_cached(cache_key, result)
        return result

    async def _load_user_preferences_dict(
        self, user_id: UUID
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load and structure user preferences for merging.

        Returns nested dict: {preference_type: {preference_key: value}}
        """
        preferences = await self.get_user_preferences(user_id, active_only=True)

        result: Dict[str, Dict[str, Any]] = {
            "default_model": {},
            "model_params": {},
            "task_params": {},
            "agent_params": {},
        }

        for pref in preferences:
            ptype = pref.preference_type
            result[ptype][pref.preference_key] = pref.preference_value

        return result

    async def get_resolved_config(
        self,
        user_id: Optional[UUID] = None,
        use_cache: bool = True,
    ) -> ResolvedConfigResponse:
        """
        Resolve full configuration for a user.

        WHY: Merges YAML → admin overrides → user preferences
        into a single effective configuration.

        Args:
            user_id: User ID for user-specific preferences (None = base config only)
            use_cache: Whether to use Redis cache

        Returns:
            ResolvedConfigResponse with effective configuration
        """
        # Check cache for user-specific config
        if user_id and use_cache:
            cache_key = self._user_cache_key(user_id)
            cached = await self._get_cached(cache_key)
            if cached:
                cached["cached_at"] = datetime.now(UTC).isoformat()
                return ResolvedConfigResponse(**cached)

        # Start with base YAML config
        base = self._base_config
        admin_overrides = await self._load_admin_overrides_dict()

        # Build available models list (from YAML, filtered by admin)
        available_models = list(base.models.keys())
        sources: Dict[str, str] = {}

        # Apply admin global model overrides (full replacement)
        if "models" in admin_overrides["global"]:
            available_models = list(admin_overrides["global"]["models"].keys())
            sources["available_models"] = "admin"

        # Apply admin global disables
        elif "disabled_models" in admin_overrides["global"]:
            disabled = admin_overrides["global"]["disabled_models"].get("value", [])
            available_models = [m for m in available_models if m not in disabled]
            sources["available_models"] = "admin"

        # Build task configs
        tasks: Dict[str, ResolvedTaskConfig] = {}

        for task_name, task_config in base.task_configs.items():
            primary = task_config.primary
            fallback = task_config.fallback

            # Apply admin task overrides
            if task_name in admin_overrides.get("task", {}):
                task_override = admin_overrides["task"][task_name]
                if "primary" in task_override:
                    primary = task_override["primary"].get("value", primary)
                    sources[f"tasks.{task_name}.primary"] = "admin"
                if "fallback" in task_override:
                    fallback = task_override["fallback"].get("value", fallback)
                    sources[f"tasks.{task_name}.fallback"] = "admin"

            tasks[task_name] = ResolvedTaskConfig(
                primary_model=primary,
                fallback_models=fallback,
            )

        # Resolve Agents
        agents = {}
        for agent_name, agent_config in base.agent_configs.items():
            primary = agent_config.primary
            fallback = agent_config.fallback
            allowed_tools = agent_config.allowed_tools
            mcp_profile = agent_config.mcp_profile

            # Apply admin agent overrides
            if agent_name in admin_overrides.get("agent", {}):
                agent_override = admin_overrides["agent"][agent_name]

                if "primary" in agent_override:
                    primary = agent_override["primary"].get("value", primary)
                    sources[f"agents.{agent_name}.primary"] = "admin"

                if "fallback" in agent_override:
                    fallback = agent_override["fallback"].get("value", fallback)
                    sources[f"agents.{agent_name}.fallback"] = "admin"

                if "allowed_tools" in agent_override:
                    allowed_tools = agent_override["allowed_tools"].get(
                        "value", allowed_tools
                    )
                    sources[f"agents.{agent_name}.allowed_tools"] = "admin"

            agents[agent_name] = ResolvedAgentConfig(
                primary_model=primary,
                fallback_models=fallback,
                allowed_tools=allowed_tools,
                mcp_profile=mcp_profile,
                description=agent_config.description,
            )

        # Build defaults dict
        defaults: Dict[str, Any] = {
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        # Apply admin global defaults
        for key in ["temperature", "max_tokens", "top_p"]:
            if key in admin_overrides["global"]:
                defaults[key] = admin_overrides["global"][key].get(
                    "value", defaults.get(key)
                )
                sources[f"defaults.{key}"] = "admin"

        # Apply user preferences if user_id provided
        if user_id:
            user_prefs = await self._load_user_preferences_dict(user_id)

            # User's default model preference
            if "default_model" in user_prefs.get("default_model", {}):
                model_pref = user_prefs["default_model"]["default_model"]
                model_name = model_pref.get("value")
                if model_name and model_name in available_models:
                    defaults["default_model"] = model_name
                    sources["defaults.default_model"] = "user"

            # User's parameter preferences
            for key, value_dict in user_prefs.get("model_params", {}).items():
                # key format: "temperature" or "model_name.temperature"
                param_name = key.split(".")[-1] if "." in key else key
                if param_name in ["temperature", "max_tokens", "top_p"]:
                    defaults[param_name] = value_dict.get(
                        "value", defaults.get(param_name)
                    )
                    sources[f"defaults.{param_name}"] = "user"

        response = ResolvedConfigResponse(
            sources=sources,
            available_models=available_models,
            tasks=tasks,
            agents=agents,
            defaults=defaults,
            cache_ttl_seconds=self.CACHE_TTL_SECONDS,
        )

        # Cache the result
        if user_id and use_cache:
            await self._set_cached(
                self._user_cache_key(user_id),
                response.model_dump(exclude={"cached_at"}),
            )

        return response

    async def get_effective_model_params(
        self,
        user_id: Optional[UUID],
        model_name: str,
        task_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get effective parameters for a specific model call.

        WHY: Used by LLMService when making actual LLM calls.
        Merges YAML model defaults → admin overrides → user preferences.

        Args:
            user_id: User making the request (None for anonymous)
            model_name: Target model name
            task_type: Optional task type for task-specific overrides

        Returns:
            Dict of effective parameters for the model call
        """
        base = self._base_config
        model_config = base.models.get(model_name)

        if not model_config:
            logger.warning(f"Model '{model_name}' not found in config")
            return {}

        # Start with model defaults from YAML
        params = {
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
        }

        # Apply admin model-specific overrides
        admin_overrides = await self._load_admin_overrides_dict()
        if model_name in admin_overrides.get("model", {}):
            model_overrides = admin_overrides["model"][model_name]
            for key in ["temperature", "max_tokens", "top_p"]:
                if key in model_overrides:
                    params[key] = model_overrides[key].get("value", params.get(key))

        # Apply user preferences
        if user_id:
            user_prefs = await self._load_user_preferences_dict(user_id)

            # Model-specific user params
            model_key = f"{model_name}.temperature"
            if model_key in user_prefs.get("model_params", {}):
                params["temperature"] = user_prefs["model_params"][model_key].get(
                    "value", params["temperature"]
                )

            # Task-specific user params
            if task_type:
                task_key = f"{task_type}.temperature"
                if task_key in user_prefs.get("task_params", {}):
                    params["temperature"] = user_prefs["task_params"][task_key].get(
                        "value", params["temperature"]
                    )

            # Global user params (lowest priority for user layer)
            for key in ["temperature", "max_tokens", "top_p"]:
                if key in user_prefs.get("model_params", {}):
                    # Only apply if not already set by more specific preference
                    if f"{model_name}.{key}" not in user_prefs.get("model_params", {}):
                        params[key] = user_prefs["model_params"][key].get(
                            "value", params.get(key)
                        )

        return params

    async def get_available_options(self, user_id: UUID) -> AvailableOptionsResponse:
        """
        Get available configuration options for a user.

        WHY: Frontend needs this to build dynamic settings UI.
        Shows what users can configure and what constraints apply.
        """
        allowed = await self.get_allowed_overrides(active_only=True)
        resolved = await self.get_resolved_config(user_id)

        options = []
        for a in allowed:
            constraints = a.constraints.copy() if a.constraints else None
            # Populate allowed_values for select constraints that need dynamic values
            if (
                a.config_key == "default_model"
                and constraints
                and constraints.get("type") == "select"
                and "allowed_values" not in constraints
            ):
                constraints["allowed_values"] = resolved.available_models

            options.append(
                AllowedOverrideResponse(
                    id=a.id,
                    config_key=a.config_key,
                    constraints=constraints,
                    allowed_scopes=(
                        a.allowed_scopes.get("scopes") if a.allowed_scopes else None
                    ),
                    display_name=a.display_name,
                    description=a.description,
                    is_active=a.is_active,
                    created_at=a.created_at,
                    updated_at=a.updated_at,
                )
            )

        return AvailableOptionsResponse(
            options=options,
            available_models=resolved.available_models,
            available_tasks=list(resolved.tasks.keys()),
            available_agents=list(resolved.agents.keys()),
        )

    async def get_config_with_overrides(self, user_id: Optional[UUID]) -> LLMConfig:
        """
        Get an LLMConfig instance with all admin and user overrides applied.

        WHY: To be passed to services (like AgentService) that need a full
        configuration object respecting user preferences.
        """
        # Load admin overrides
        admin_overrides = await self._load_admin_overrides_dict()

        # Load user preferences
        user_prefs = {}
        if user_id:
            user_prefs_dict = await self._load_user_preferences_dict(user_id)

            # Transform user prefs structure to overrides structure if needed
            # Currently user_prefs returns {type: {key: value_dict}}
            # We need to extract the raw values for LLMConfig.with_overrides

            # 1. Model Params (User prefs usually map to models or defaults)
            # The current with_overrides expects: model_overrides, task_overrides, agent_overrides
            # But user prefs are often "defaults.temperature" or "completion.temperature"
            # Logic here needs to map user prefs to the override structure expected by with_overrides

            # Implementation simplification: For now, we only map 'default_model' and basic params
            # to global defaults, as deeper structural overrides might be complex.
            # However, if we want to allow users to override AGENT tools, we need to map 'agent_params'

            if "agent_params" in user_prefs_dict:
                # Map agent params to agent_overrides
                # user_prefs: {"agent_params": {"my_agent.allowed_tools": {"value": [...]}}}
                agent_overrides = {}
                for key, val_dict in user_prefs_dict["agent_params"].items():
                    # key = "agent_name.param"
                    if "." in key:
                        agent_name, param = key.split(".", 1)
                        if agent_name not in agent_overrides:
                            agent_overrides[agent_name] = {}
                        agent_overrides[agent_name][param] = val_dict["value"]

                # We'll merge this with admin agent overrides
                for ag_name, overrides in agent_overrides.items():
                    if ag_name not in admin_overrides["agent"]:
                        admin_overrides["agent"][ag_name] = {}
                    admin_overrides["agent"][ag_name].update(overrides)

        # Construct overrides dicts from admin_overrides (which now includes merged user agent prefs)

        final_agent_overrides = copy.deepcopy(admin_overrides.get("agent", {}))

        return self._base_config.with_overrides(
            model_overrides=admin_overrides.get("model"),
            task_overrides=admin_overrides.get("task"),
            agent_overrides=final_agent_overrides,
            global_overrides=admin_overrides.get("global"),
        )


# =========================================================
# Factory / Dependency
# =========================================================


def get_llm_config_service(db: AsyncSession) -> LLMConfigService:
    """Factory function for LLMConfigService dependency injection."""
    return LLMConfigService(db)
