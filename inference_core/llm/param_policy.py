"""
LLM Provider Parameter Policy

Centralized parameter normalization and validation for all LLM providers.
Prevents runtime errors from provider-specific parameter incompatibilities.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from .config import ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class ProviderParamPolicy:
    """Parameter policy for a specific LLM provider or model.

    If used at model level it represents the *effective* merged policy.
    """

    # Parameters that are accepted by the provider/model (after renaming applied)
    allowed: Set[str] = field(default_factory=set)

    # Parameter mappings: old_name -> new_name
    renamed: Dict[str, str] = field(default_factory=dict)

    # Parameters that should be silently dropped (never forwarded)
    dropped: Set[str] = field(default_factory=set)

    # Prefixes for experimental passthrough parameters (e.g. x_, ext_)
    passthrough_prefixes: Set[str] = field(default_factory=set)

    def clone(self) -> "ProviderParamPolicy":
        return ProviderParamPolicy(
            allowed=set(self.allowed),
            renamed=dict(self.renamed),
            dropped=set(self.dropped),
            passthrough_prefixes=set(self.passthrough_prefixes),
        )

    def apply_patch(self, patch: Dict[str, Any]):
        """Patch (add/merge) collections in-place."""
        if not patch:
            return
        if isinstance(patch.get("allowed"), list):
            self.allowed.update(patch["allowed"])
        if isinstance(patch.get("dropped"), list):
            self.dropped.update(patch["dropped"])
        if isinstance(patch.get("renamed"), dict):
            self.renamed.update(patch["renamed"])
        if isinstance(patch.get("passthrough_prefixes"), list):
            self.passthrough_prefixes.update(patch["passthrough_prefixes"])

    def apply_replace(self, replace: Dict[str, Any]):
        """Replace entire collections if provided."""
        if not replace:
            return
        if "allowed" in replace:
            self.allowed = set(replace.get("allowed") or [])
        if "dropped" in replace:
            self.dropped = set(replace.get("dropped") or [])
        if "renamed" in replace:
            self.renamed = dict(replace.get("renamed") or {})
        if "passthrough_prefixes" in replace:
            self.passthrough_prefixes = set(replace.get("passthrough_prefixes") or [])


# Provider-specific parameter policies
POLICIES: Dict[ModelProvider, ProviderParamPolicy] = {
    ModelProvider.OPENAI: ProviderParamPolicy(
        allowed={
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "request_timeout",
        },
        renamed={},
        dropped=set(),
        passthrough_prefixes=set(),
    ),
    ModelProvider.CUSTOM_OPENAI_COMPATIBLE: ProviderParamPolicy(
        allowed={
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "request_timeout",
        },
        renamed={},
        dropped=set(),
        passthrough_prefixes=set(),
    ),
    ModelProvider.GEMINI: ProviderParamPolicy(
        allowed={"temperature", "max_output_tokens", "top_p"},
        renamed={"max_tokens": "max_output_tokens"},
        dropped={"frequency_penalty", "presence_penalty", "request_timeout"},
        passthrough_prefixes=set(),
    ),
    ModelProvider.CLAUDE: ProviderParamPolicy(
        allowed={"temperature", "max_tokens", "top_p", "timeout"},
        renamed={"request_timeout": "timeout"},
        dropped={"frequency_penalty", "presence_penalty"},
        passthrough_prefixes=set(),
    ),
}

# Dynamic override state
_DYNAMIC_LOADED = False
_EFFECTIVE_MODEL_POLICIES: Dict[str, ProviderParamPolicy] = {}
_GLOBAL_PASSTHROUGH_PREFIXES: Set[str] = set()
STREAMING_PARAMS = set(["streaming", "callbacks"])


def _merge_policy(
    base: ProviderParamPolicy, override: Dict[str, Any]
) -> ProviderParamPolicy:
    """Return a new policy with overrides applied (patch/replace semantics)."""
    merged = base.clone()
    if not override:
        return merged
    replace = override.get("replace") or {}
    patch = override.get("patch") or {}
    merged.apply_replace(replace)
    merged.apply_patch(patch)
    return merged


def _ensure_dynamic_loaded():
    global _DYNAMIC_LOADED, _GLOBAL_PASSTHROUGH_PREFIXES
    if _DYNAMIC_LOADED:
        return
    try:
        from .config import (  # Local import to avoid circular at module load
            get_llm_config,
        )

        llm_config = get_llm_config()

        yaml_cfg = getattr(llm_config, "_yaml_config", None)
        if not yaml_cfg:
            _DYNAMIC_LOADED = True
            return
        param_policies = yaml_cfg.get("param_policies", {}) or {}
        # Settings-level passthrough prefixes
        settings_cfg = param_policies.get("settings", {}) or {}
        prefixes = settings_cfg.get("passthrough_prefixes") or []
        if isinstance(prefixes, list):
            _GLOBAL_PASSTHROUGH_PREFIXES.update(prefixes)

        # Provider overrides
        provider_overrides = param_policies.get("providers", {}) or {}
        for provider_name, override in provider_overrides.items():
            try:
                provider_enum = ModelProvider(provider_name)
            except ValueError:
                logger.warning(
                    f"Param policy override ignored for unknown provider '{provider_name}'"
                )
                continue
            base_policy = POLICIES.get(provider_enum)
            if not base_policy:
                logger.warning(
                    f"No base policy found for provider '{provider_name}' (should not happen)"
                )
                continue
            POLICIES[provider_enum] = _merge_policy(base_policy, override)

        # Model overrides
        model_overrides = param_policies.get("models", {}) or {}
        for model_name, override in model_overrides.items():
            provider_str: Optional[str] = override.get("provider")
            provider_enum: Optional[ModelProvider] = None
            if provider_str:
                try:
                    provider_enum = ModelProvider(provider_str)
                except ValueError:
                    logger.warning(
                        f"Model-level param policy override for '{model_name}' has unknown provider '{provider_str}'"
                    )
                    continue
            # If provider not explicitly stated we'll attempt to resolve later at access time.
            _EFFECTIVE_MODEL_POLICIES[model_name] = (
                ProviderParamPolicy()
            )  # Placeholder storing override only via custom attribute
            # Instead of merging now, store raw override for later resolution
            setattr(_EFFECTIVE_MODEL_POLICIES[model_name], "_raw_override", override)
            setattr(_EFFECTIVE_MODEL_POLICIES[model_name], "_provider", provider_enum)

    except Exception as e:
        logger.error(f"Failed loading dynamic param policy overrides: {e}")
    finally:
        _DYNAMIC_LOADED = True


def _resolve_model_policy(
    model_name: str, provider: ModelProvider
) -> ProviderParamPolicy:
    """Compute (and cache) effective model policy for model_name."""
    _ensure_dynamic_loaded()
    # If we already computed a concrete policy (no _raw_override attr) return it
    existing = _EFFECTIVE_MODEL_POLICIES.get(model_name)
    if existing and not hasattr(existing, "_raw_override"):
        return existing
    if existing and hasattr(existing, "_raw_override"):
        raw_override = getattr(existing, "_raw_override")
        # Determine provider: override may specify provider explicitly
        override_provider = getattr(existing, "_provider", None) or provider
        base_policy = POLICIES.get(override_provider, ProviderParamPolicy())
        effective = _merge_policy(base_policy, raw_override)
        # Cache resolved effective policy (remove attrs)
        _EFFECTIVE_MODEL_POLICIES[model_name] = effective
        return effective
    # No model-specific override: return provider effective policy
    return POLICIES.get(provider, ProviderParamPolicy())


def get_effective_policy(
    provider: ModelProvider, model_name: Optional[str] = None
) -> ProviderParamPolicy:
    """Get effective policy (provider base + overrides + model overrides)."""
    _ensure_dynamic_loaded()
    if model_name:
        return _resolve_model_policy(model_name, provider)
    return POLICIES.get(provider, ProviderParamPolicy())


def normalize_params(
    provider: ModelProvider,
    raw_params: Dict[str, Any],
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize parameters for a specific provider.

    Args:
        provider: The LLM provider
        raw_params: Raw parameter dictionary

    Returns:
        Normalized parameter dictionary safe for the provider

    Raises:
        ValueError: If provider is not supported
    """
    if provider not in POLICIES:
        raise ValueError(f"Unsupported provider: {provider}")

    policy = get_effective_policy(provider, model_name=model_name)
    normalized = {}

    for param_name, param_value in raw_params.items():
        # Skip None values
        if param_value is None:
            continue

        # Check if parameter is a streaming-related parameter
        if param_name in STREAMING_PARAMS:
            continue  # Allow streaming params to pass through

        # Check if parameter should be renamed
        if param_name in policy.renamed:
            new_name = policy.renamed[param_name]
            normalized[new_name] = param_value
            logger.debug(
                f"Parameter renamed for {provider}: {param_name} -> {new_name}"
            )
            continue

        # Check if parameter should be dropped
        if param_name in policy.dropped:
            logger.debug(
                f"Parameter dropped for {provider}: {param_name} "
                f"(value: {param_value})"
            )
            continue

        # Check if parameter is allowed as-is
        if param_name in policy.allowed:
            normalized[param_name] = param_value
            continue

        # Parameter not recognized: allow passthrough if it matches experimental prefixes
        if any(
            param_name.startswith(pref)
            for pref in (policy.passthrough_prefixes | _GLOBAL_PASSTHROUGH_PREFIXES)
        ):
            normalized[param_name] = param_value
            logger.debug(
                f"Experimental passthrough parameter for {provider}: {param_name}"
            )
            continue

        logger.warning(
            f"Unknown parameter for {provider}: {param_name} (value: {param_value}) - dropping"
        )

    return normalized


def get_provider_policy(provider: ModelProvider) -> ProviderParamPolicy:
    """
    Get the parameter policy for a provider.

    Args:
        provider: The LLM provider

    Returns:
        Provider parameter policy

    Raises:
        ValueError: If provider is not supported
    """
    if provider not in POLICIES:
        raise ValueError(f"Unsupported provider: {provider}")
    _ensure_dynamic_loaded()
    return POLICIES[provider]


def get_model_policy(model_name: str, provider: ModelProvider) -> ProviderParamPolicy:
    """Expose effective model policy (for debug endpoint)."""
    return get_effective_policy(provider, model_name=model_name)


def get_supported_providers() -> Set[ModelProvider]:
    """Get set of providers with defined parameter policies."""
    return set(POLICIES.keys())


def validate_provider_params(provider: ModelProvider, params: Dict[str, Any]) -> bool:
    """
    Validate that all parameters are supported by the provider.

    Args:
        provider: The LLM provider
        params: Parameters to validate

    Returns:
        True if all parameters are supported, False otherwise
    """
    if provider not in POLICIES:
        return False

    policy = POLICIES[provider]

    for param_name in params:
        if params[param_name] is None:
            continue

        if (
            param_name not in policy.allowed
            and param_name not in policy.renamed
            and param_name not in policy.dropped
        ):
            return False

    return True
