"""
LLM Configuration API Router

Endpoints for managing dynamic LLM configuration:
- User preferences (authenticated users)
- Admin overrides (superusers only)
- Available options and resolved config
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from inference_core.core.dependecies import (
    get_current_active_user,
    get_current_superuser,
    get_llm_config_service,
)
from inference_core.schemas.llm_config import (
    AdminOverrideCreate,
    AdminOverrideListResponse,
    AdminOverrideResponse,
    AdminOverrideUpdate,
    AllowedOverrideCreate,
    AllowedOverrideListResponse,
    AllowedOverrideResponse,
    AvailableOptionsResponse,
    ConfigScopeEnum,
    PreferenceListResponse,
    PreferenceTypeEnum,
    ResolvedConfigResponse,
    UserPreferenceBulkUpdate,
    UserPreferenceCreate,
    UserPreferenceResponse,
)
from inference_core.services.llm_config_service import (
    ConfigValidationError,
    LLMConfigService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["LLM Configuration"])


# =========================================================
# User Preferences Endpoints
# =========================================================


@router.get(
    "/preferences",
    response_model=PreferenceListResponse,
    summary="List user preferences",
)
async def list_user_preferences(
    preference_type: Optional[PreferenceTypeEnum] = None,
    current_user: dict = Depends(get_current_active_user),
    config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    List all LLM preferences for the current user.

    Optionally filter by preference type (default_model, model_params, etc.).
    """
    user_id = UUID(current_user["id"])
    preferences = await config_service.get_user_preferences(
        user_id=user_id,
        preference_type=preference_type,
    )

    return PreferenceListResponse(
        preferences=[
            UserPreferenceResponse(
                id=p.id,
                user_id=p.user_id,
                preference_type=p.preference_type,
                preference_key=p.preference_key,
                preference_value=p.preference_value,
                is_active=p.is_active,
                created_at=p.created_at,
                updated_at=p.updated_at,
            )
            for p in preferences
        ],
        total=len(preferences),
    )


@router.post(
    "/preferences",
    response_model=UserPreferenceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or update preference",
)
async def create_user_preference(
    preference: UserPreferenceCreate,
    current_user: dict = Depends(get_current_active_user),
    config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    Create or update a user preference.

    If a preference with the same type/key already exists, it will be updated.
    Preferences are validated against the allowed overrides list.
    """
    try:
        user_id = UUID(current_user["id"])
        result = await config_service.create_user_preference(
            user_id=user_id,
            preference_type=preference.preference_type,
            preference_key=preference.preference_key,
            preference_value=preference.preference_value,
        )

        return UserPreferenceResponse(
            id=result.id,
            user_id=result.user_id,
            preference_type=result.preference_type,
            preference_key=result.preference_key,
            preference_value=result.preference_value,
            is_active=result.is_active,
            created_at=result.created_at,
            updated_at=result.updated_at,
        )
    except ConfigValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation failed for '{e.key}': {e.message}",
        )


@router.put(
    "/preferences/bulk",
    response_model=PreferenceListResponse,
    summary="Bulk update preferences",
)
async def bulk_update_preferences(
    bulk_update: UserPreferenceBulkUpdate,
    current_user: dict = Depends(get_current_active_user),
    config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    Bulk create or update multiple preferences at once.

    Useful for saving an entire settings form in one request.
    """
    user_id = UUID(current_user["id"])
    results = []
    errors = []

    for pref in bulk_update.preferences:
        try:
            result = await config_service.create_user_preference(
                user_id=user_id,
                preference_type=pref.preference_type,
                preference_key=pref.preference_key,
                preference_value=pref.preference_value,
            )
            results.append(result)
        except ConfigValidationError as e:
            errors.append(f"{pref.preference_key}: {e.message}")

    if errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Some preferences failed validation", "errors": errors},
        )

    return PreferenceListResponse(
        preferences=[
            UserPreferenceResponse(
                id=r.id,
                user_id=r.user_id,
                preference_type=r.preference_type,
                preference_key=r.preference_key,
                preference_value=r.preference_value,
                is_active=r.is_active,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in results
        ],
        total=len(results),
    )


@router.delete(
    "/preferences/{preference_key:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete preference",
)
async def delete_user_preference(
    preference_key: str,
    current_user: dict = Depends(get_current_active_user),
    config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    Delete a user preference by key.

    The preference_key can include dots (e.g., 'chat.temperature').
    """
    user_id = UUID(current_user["id"])
    deleted = await config_service.delete_user_preference(
        user_id=user_id,
        preference_key=preference_key,
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preference '{preference_key}' not found",
        )


# =========================================================
# Resolved Configuration Endpoints
# =========================================================


@router.get(
    "/resolved",
    response_model=ResolvedConfigResponse,
    summary="Get resolved configuration",
)
async def get_resolved_config(
    current_user: dict = Depends(get_current_active_user),
    config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    Get the fully resolved LLM configuration for the current user.

    Merges: YAML base config → Admin DB overrides → User preferences.
    Returns effective models, tasks, and default parameters.
    """
    user_id = UUID(current_user["id"])
    return await config_service.get_resolved_config(user_id=user_id)


@router.get(
    "/available-options",
    response_model=AvailableOptionsResponse,
    summary="Get available configuration options",
)
async def get_available_options(
    current_user: dict = Depends(get_current_active_user),
    config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    Get available configuration options for the current user.

    Returns list of configurable parameters with their constraints,
    available models, and available tasks. Used by frontend to build
    dynamic settings UI.
    """
    user_id = UUID(current_user["id"])
    return await config_service.get_available_options(user_id=user_id)


# =========================================================
# Admin Override Endpoints (Superuser only)
# =========================================================


@router.get(
    "/admin/overrides",
    response_model=AdminOverrideListResponse,
    summary="List admin overrides",
)
async def list_admin_overrides(
    scope: Optional[ConfigScopeEnum] = None,
    scope_key: Optional[str] = None,
    include_inactive: bool = False,
    admin_user: dict = Depends(get_current_superuser),
    config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    List all admin configuration overrides.

    Superuser only. Optionally filter by scope and scope_key.
    """
    overrides = await config_service.get_admin_overrides(
        scope=scope,
        scope_key=scope_key,
        active_only=not include_inactive,
    )

    return AdminOverrideListResponse(
        overrides=[
            AdminOverrideResponse(
                id=o.id,
                scope=ConfigScopeEnum(o.scope),
                scope_key=o.scope_key,
                config_key=o.config_key,
                config_value=o.config_value,
                priority=o.priority,
                description=o.description,
                expires_at=o.expires_at,
                is_active=o.is_active,
                created_by_id=o.created_by_id,
                created_at=o.created_at,
                updated_at=o.updated_at,
            )
            for o in overrides
        ],
        total=len(overrides),
    )


@router.post(
    "/admin/overrides",
    response_model=AdminOverrideResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create admin override",
)
async def create_admin_override(
    override: AdminOverrideCreate,
    admin_user: dict = Depends(get_current_superuser),
    config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    Create a new admin configuration override.

    Superuser only. Overrides are applied to all users based on scope.
    """
    result = await config_service.create_admin_override(
        scope=override.scope,
        scope_key=override.scope_key,
        config_key=override.config_key,
        config_value=override.config_value,
        priority=override.priority,
        description=override.description,
        expires_at=override.expires_at,
        created_by_id=UUID(admin_user["id"]),
    )

    return AdminOverrideResponse(
        id=result.id,
        scope=ConfigScopeEnum(result.scope),
        scope_key=result.scope_key,
        config_key=result.config_key,
        config_value=result.config_value,
        priority=result.priority,
        description=result.description,
        expires_at=result.expires_at,
        is_active=result.is_active,
        created_by_id=result.created_by_id,
        created_at=result.created_at,
        updated_at=result.updated_at,
    )


@router.patch(
    "/admin/overrides/{override_id}",
    response_model=AdminOverrideResponse,
    summary="Update admin override",
)
async def update_admin_override(
    override_id: UUID,
    update: AdminOverrideUpdate,
    admin_user: dict = Depends(get_current_superuser),
    config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    Update an existing admin configuration override.

    Superuser only. Only provided fields will be updated.
    """
    result = await config_service.update_admin_override(
        override_id=override_id,
        **update.model_dump(exclude_none=True),
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Override not found",
        )

    return AdminOverrideResponse(
        id=result.id,
        scope=ConfigScopeEnum(result.scope),
        scope_key=result.scope_key,
        config_key=result.config_key,
        config_value=result.config_value,
        priority=result.priority,
        description=result.description,
        expires_at=result.expires_at,
        is_active=result.is_active,
        created_by_id=result.created_by_id,
        created_at=result.created_at,
        updated_at=result.updated_at,
    )


@router.delete(
    "/admin/overrides/{override_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete admin override",
)
async def delete_admin_override(
    override_id: UUID,
    admin_user: dict = Depends(get_current_superuser),
    config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    Delete an admin configuration override.

    Superuser only. Performs soft-delete.
    """
    deleted = await config_service.delete_admin_override(override_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Override not found",
        )


# =========================================================
# Allowed Overrides Management (Superuser only)
# =========================================================


@router.get(
    "/admin/allowed-overrides",
    response_model=AllowedOverrideListResponse,
    summary="List allowed user overrides",
)
async def list_allowed_overrides(
    include_inactive: bool = False,
    admin_user: dict = Depends(get_current_superuser),
    config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    List all allowed user override definitions.

    Superuser only. Shows what configuration keys users can modify
    and the constraints applied to each.
    """
    allowed = await config_service.get_allowed_overrides(
        active_only=not include_inactive
    )

    return AllowedOverrideListResponse(
        allowed_overrides=[
            AllowedOverrideResponse(
                id=a.id,
                config_key=a.config_key,
                constraints=a.constraints,
                allowed_scopes=(
                    a.allowed_scopes.get("scopes") if a.allowed_scopes else None
                ),
                display_name=a.display_name,
                description=a.description,
                is_active=a.is_active,
                created_at=a.created_at,
                updated_at=a.updated_at,
            )
            for a in allowed
        ],
        total=len(allowed),
    )


@router.post(
    "/admin/allowed-overrides",
    response_model=AllowedOverrideResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create allowed override definition",
)
async def create_allowed_override(
    allowed: AllowedOverrideCreate,
    admin_user: dict = Depends(get_current_superuser),
    config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    Create a new allowed override definition.

    Superuser only. Defines what configuration keys users can modify
    and the constraints applied to each (min/max values, allowed options, etc.).
    """
    result = await config_service.create_allowed_override(
        config_key=allowed.config_key,
        constraints=allowed.constraints.model_dump() if allowed.constraints else None,
        allowed_scopes=(
            [s.value for s in allowed.allowed_scopes]
            if allowed.allowed_scopes
            else None
        ),
        display_name=allowed.display_name,
        description=allowed.description,
    )

    return AllowedOverrideResponse(
        id=result.id,
        config_key=result.config_key,
        constraints=result.constraints,
        allowed_scopes=(
            result.allowed_scopes.get("scopes") if result.allowed_scopes else None
        ),
        display_name=result.display_name,
        description=result.description,
        is_active=result.is_active,
        created_at=result.created_at,
        updated_at=result.updated_at,
    )
