# Batch Provider System

This directory contains the batch processing provider system for LLM operations.

## Overview

The batch provider system provides a common interface for batch processing across different LLM providers. It consists of:

- **BaseBatchProvider**: Abstract base class defining the provider interface
- **BatchProviderRegistry**: Registry for dynamic provider lookup
- **DTOs**: Data transfer objects for provider operations
- **Exceptions**: Specific error types for batch operations
- **Enums**: Standardized enumerations for status and processing modes

## Core Components

### BaseBatchProvider

All batch providers must inherit from `BaseBatchProvider` and implement:

- `supports_model(model: str) -> bool`: Check if model is supported
- `prepare_payloads(...)`: Convert requests to provider format
- `submit(...)`: Submit batch to provider
- `poll_status(...)`: Check batch status
- `fetch_results(...)`: Retrieve completed results
- `cancel(...)`: Cancel a running batch

### Enums and Status Normalization

The system uses standardized enums for consistency:

- **BatchMode**: Processing modes (CHAT, COMPLETION, EMBEDDING, CUSTOM)
- **BatchStatus**: Normalized status values (VALIDATING, QUEUED, IN_PROGRESS, etc.)

Provider-specific status values are normalized to internal enums using `normalize_provider_status()`.

### Usage Information

The system now provides structured usage information via the `UsageInfo` model:

```python
from app.llm.batch.dto import UsageInfo

usage = UsageInfo(
    prompt_tokens=10,
    completion_tokens=20,
    total_tokens=30
)
```

### Registry Usage

```python
from app.llm.batch import batch_provider_registry, BatchMode
from app.llm.batch.providers.openai_provider import OpenAIBatchProvider

# Register a provider
batch_provider_registry.register(OpenAIBatchProvider)

# Get a provider
provider_class = batch_provider_registry.get("openai")
provider = provider_class(config={"api_key": "your-key"})

# Use with enums
if provider.supports_model("gpt-4"):
    prepared = provider.prepare_payloads(
        batch_id=uuid4(),
        model="gpt-4", 
        mode=BatchMode.CHAT,
        requests=[{"messages": [{"role": "user", "content": "Hello"}]}]
    )
```

### Creating a New Provider

1. Create a new file in `app/llm/batch/providers/`
2. Inherit from `BaseBatchProvider`
3. Define `PROVIDER_NAME` constant
4. Implement all abstract methods using proper enums
5. Use status normalization for consistent status handling
6. Register the provider using the registry

## Future Enhancements (TODOs)

### Issue #002 Integration

Several features are planned for better YAML-driven configuration:

1. **Config-driven Model Support**: Replace hard-coded model lists with lookups from `llm_config.yaml`
2. **Chunking Support**: Handle large batches that exceed provider limits
3. **Provider Status Mappings**: Move status mappings to YAML configuration

## Error Handling

The system defines two types of errors:

- **ProviderTransientError**: Temporary issues that can be retried (rate limits, network timeouts)
- **ProviderPermanentError**: Permanent issues that should not be retried (invalid API key, unsupported model)

## Testing

Unit tests are provided to verify:

- Interface contract compliance
- Registry functionality
- Error handling
- Provider registration and retrieval
- Enum usage and status normalization
- Structured usage information