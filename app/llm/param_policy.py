"""
LLM Provider Parameter Policy

Centralized parameter normalization and validation for all LLM providers.
Prevents runtime errors from provider-specific parameter incompatibilities.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Set

from .config import ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class ProviderParamPolicy:
    """Parameter policy for a specific LLM provider"""
    
    # Parameters that are accepted by the provider
    allowed: Set[str]
    
    # Parameter mappings: old_name -> new_name
    renamed: Dict[str, str]
    
    # Parameters that should be silently dropped
    dropped: Set[str]


# Provider-specific parameter policies
POLICIES: Dict[ModelProvider, ProviderParamPolicy] = {
    ModelProvider.OPENAI: ProviderParamPolicy(
        allowed={
            "temperature", "max_tokens", "top_p", "frequency_penalty", 
            "presence_penalty", "request_timeout"
        },
        renamed={},
        dropped=set()
    ),
    
    ModelProvider.CUSTOM_OPENAI_COMPATIBLE: ProviderParamPolicy(
        allowed={
            "temperature", "max_tokens", "top_p", "frequency_penalty", 
            "presence_penalty", "request_timeout"
        },
        renamed={},
        dropped=set()
    ),
    
    ModelProvider.GEMINI: ProviderParamPolicy(
        allowed={
            "temperature", "max_output_tokens", "top_p"
        },
        renamed={
            "max_tokens": "max_output_tokens"
        },
        dropped={
            "frequency_penalty", "presence_penalty", "request_timeout"
        }
    ),
    
    ModelProvider.CLAUDE: ProviderParamPolicy(
        allowed={
            "temperature", "max_tokens", "top_p", "timeout"
        },
        renamed={
            "request_timeout": "timeout"
        },
        dropped={
            "frequency_penalty", "presence_penalty"
        }
    )
}


def normalize_params(provider: ModelProvider, raw_params: Dict[str, Any]) -> Dict[str, Any]:
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
    
    policy = POLICIES[provider]
    normalized = {}
    
    for param_name, param_value in raw_params.items():
        # Skip None values
        if param_value is None:
            continue
            
        # Check if parameter should be renamed
        if param_name in policy.renamed:
            new_name = policy.renamed[param_name]
            normalized[new_name] = param_value
            logger.debug(
                f"Parameter renamed for {provider.value}: {param_name} -> {new_name}"
            )
            continue
            
        # Check if parameter should be dropped
        if param_name in policy.dropped:
            logger.debug(
                f"Parameter dropped for {provider.value}: {param_name} "
                f"(value: {param_value})"
            )
            continue
            
        # Check if parameter is allowed as-is
        if param_name in policy.allowed:
            normalized[param_name] = param_value
            continue
            
        # Parameter is not recognized - log warning and drop
        logger.warning(
            f"Unknown parameter for {provider.value}: {param_name} "
            f"(value: {param_value}) - dropping"
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
    
    return POLICIES[provider]


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
            
        if (param_name not in policy.allowed and 
            param_name not in policy.renamed and 
            param_name not in policy.dropped):
            return False
    
    return True