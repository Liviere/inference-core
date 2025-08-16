# Batch Provider System

This directory contains the batch processing provider system for LLM operations.

## Overview

The batch provider system provides a common interface for batch processing across different LLM providers. It consists of:

- **BaseBatchProvider**: Abstract base class defining the provider interface
- **BatchProviderRegistry**: Registry for dynamic provider lookup
- **DTOs**: Data transfer objects for provider operations
- **Exceptions**: Specific error types for batch operations

## Core Components

### BaseBatchProvider

All batch providers must inherit from `BaseBatchProvider` and implement:

- `supports_model(model: str) -> bool`: Check if model is supported
- `prepare_payloads(...)`: Convert requests to provider format
- `submit(...)`: Submit batch to provider
- `poll_status(...)`: Check batch status
- `fetch_results(...)`: Retrieve completed results
- `cancel(...)`: Cancel a running batch

### Registry Usage

```python
from app.llm.batch import batch_provider_registry
from app.llm.batch.providers.openai_provider import OpenAIBatchProvider

# Register a provider
batch_provider_registry.register(OpenAIBatchProvider)

# Get a provider
provider_class = batch_provider_registry.get("openai")
provider = provider_class(config={"api_key": "your-key"})

# List all providers
providers = batch_provider_registry.list()
```

### Creating a New Provider

1. Create a new file in `app/llm/batch/providers/`
2. Inherit from `BaseBatchProvider`
3. Define `PROVIDER_NAME` constant
4. Implement all abstract methods
5. Register the provider using the registry

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