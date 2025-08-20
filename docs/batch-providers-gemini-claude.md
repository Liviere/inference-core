# Gemini & Claude Batch Providers

This document explains how to configure and use the newly added Google Gemini and Anthropic Claude batch providers in the backend-template.

## Overview

The batch processing framework now supports three providers:

- **OpenAI** (`openai`) - GPT models with /v1/batches API
- **Google Gemini** (`gemini`) - Gemini models with native batch mode
- **Anthropic Claude** (`claude`) - Claude models with message batch API

All providers implement the same `BaseBatchProvider` interface, providing consistent batch processing capabilities across different LLM providers.

## Configuration

### Environment Variables

Set API keys for the providers you want to use:

```bash
# OpenAI (existing)
OPENAI_API_KEY=your_openai_api_key

# Google Gemini (new)
GOOGLE_GENAI_API_KEY=your_gemini_api_key

# Anthropic Claude (new)
ANTHROPIC_API_KEY=your_claude_api_key
```

### Provider Registration

Providers are automatically registered when the batch module is imported. You can check which providers are available:

```python
from app.llm.batch import registry

# List all registered providers
print(registry.list())  # ['openai', 'gemini', 'claude']

# Create a provider instance
provider = registry.create_provider("gemini", {"api_key": "your_key"})
```

## Supported Models

### Gemini Provider (`gemini`)

**Supported Models:**

- `gemini-2.0-flash`
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-pro`
- `gemini-flash`

**Supported Modes:**

- `chat` - Conversational interactions

**Input Formats:**

```python
# Messages format (preferred)
{
    "messages": [
        {"role": "user", "content": "Hello Gemini"}
    ]
}

# Direct content format
{
    "content": "Hello Gemini"
}

# Text format
{
    "text": "Hello Gemini"
}
```

### Claude Provider (`claude`)

**Supported Models:**

- `claude-3.5-sonnet`
- `claude-3-opus`
- `claude-3-haiku`
- `claude-sonnet-4`
- Any model containing "claude-3", "claude-4", "sonnet", "opus", or "haiku"

**Supported Modes:**

- `chat` - Conversational interactions

**Input Formats:**

```python
# Messages format (preferred)
{
    "messages": [
        {"role": "user", "content": "Hello Claude"}
    ],
    "max_tokens": 1024,
    "temperature": 0.7
}

# Direct content format
{
    "content": "Hello Claude",
    "max_tokens": 512
}

# With optional parameters
{
    "content": "Hello Claude",
    "temperature": 0.8,
    "top_p": 0.9,
    "max_tokens": 1024,
    "system": "You are a helpful assistant"
}
```

## Usage Examples

### Basic Batch Processing

```python
from app.llm.batch import registry

# Create provider
provider = registry.create_provider("gemini", {"api_key": "your_key"})

# Check model support
if provider.supports_model("gemini-2.0-flash", "chat"):
    print("Model supported!")

# Prepare batch items
batch_items = [
    {
        "id": "item_1",
        "input_payload": {
            "messages": [{"role": "user", "content": "Explain AI"}]
        }
    },
    {
        "id": "item_2",
        "input_payload": {
            "content": "What is machine learning?"
        }
    }
]

# Prepare payloads
prepared = provider.prepare_payloads(batch_items, "gemini-2.0-flash", "chat")

# Submit batch
result = provider.submit(prepared)
print(f"Batch ID: {result.provider_batch_id}")
print(f"Status: {result.status}")
print(f"Items: {result.item_count}")
```

### Polling and Results

```python
import time

# Poll status until completion
while True:
    status = provider.poll_status(result.provider_batch_id)
    print(f"Status: {status.normalized_status}")

    if status.normalized_status in ["completed", "failed", "cancelled"]:
        break

    time.sleep(30)  # Wait 30 seconds

# Fetch results when completed
if status.normalized_status == "completed":
    results = provider.fetch_results(result.provider_batch_id)

    for result_row in results:
        print(f"Item {result_row.custom_id}:")
        if result_row.is_success:
            print(f"  Output: {result_row.output_text}")
        else:
            print(f"  Error: {result_row.error_message}")
```

### Provider-Specific Examples

#### Gemini with Generation Config

```python
batch_items = [
    {
        "id": "creative_task",
        "input_payload": {
            "messages": [{"role": "user", "content": "Write a creative story"}],
            "generation_config": {
                "temperature": 0.9,
                "max_output_tokens": 500
            }
        }
    }
]
```

#### Claude with System Prompt

```python
batch_items = [
    {
        "id": "analysis_task",
        "input_payload": {
            "messages": [{"role": "user", "content": "Analyze this data"}],
            "system": "You are an expert data analyst",
            "temperature": 0.3,
            "max_tokens": 2048
        }
    }
]
```

## Error Handling

Both providers follow consistent error handling patterns:

### Transient Errors (Retryable)

- Rate limits
- Temporary service unavailability
- Network timeouts
- HTTP 429, 502, 503 errors

```python
from app.llm.batch.exceptions import ProviderTransientError

try:
    result = provider.submit(prepared)
except ProviderTransientError as e:
    print(f"Temporary error: {e}")
    print(f"Retry after: {e.retry_after} seconds")
    # Implement retry logic
```

### Permanent Errors (Non-retryable)

- Invalid API keys
- Unsupported models
- Malformed requests
- HTTP 401, 403, 404 errors

```python
from app.llm.batch.exceptions import ProviderPermanentError

try:
    result = provider.submit(prepared)
except ProviderPermanentError as e:
    print(f"Permanent error: {e}")
    # Fix the issue (API key, model, etc.) before retrying
```

## Status Mapping

### Gemini Status Mapping

| Gemini Status         | Internal Status |
| --------------------- | --------------- |
| `JOB_STATE_QUEUED`    | `submitted`     |
| `JOB_STATE_PENDING`   | `submitted`     |
| `JOB_STATE_RUNNING`   | `in_progress`   |
| `JOB_STATE_SUCCEEDED` | `completed`     |
| `JOB_STATE_FAILED`    | `failed`        |
| `JOB_STATE_CANCELLED` | `cancelled`     |

### Claude Status Mapping

| Claude Status | Internal Status |
| ------------- | --------------- |
| `in_progress` | `in_progress`   |
| `ended`       | `completed`     |
| `errored`     | `failed`        |
| `expired`     | `failed`        |
| `canceling`   | `cancelled`     |

## Best Practices

### Model Selection

- Use `gemini-2.0-flash` for fast, cost-effective Gemini processing
- Use `claude-3.5-sonnet` for high-quality Claude responses
- Check model support before submitting batches

### Batch Size Considerations

- **Gemini**: Supports large batches with inlined requests
- **Claude**: Efficient with moderate batch sizes (50% cost savings)
- **General**: Start with smaller batches (10-50 items) for testing

### API Key Management

- Store API keys securely in environment variables
- Use different keys for development and production
- Implement key rotation practices

### Error Recovery

```python
def submit_with_retry(provider, prepared, max_retries=3):
    for attempt in range(max_retries):
        try:
            return provider.submit(prepared)
        except ProviderTransientError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = e.retry_after or (2 ** attempt)
            time.sleep(wait_time)
        except ProviderPermanentError:
            # Don't retry permanent errors
            raise
```

## Configuration Integration

The providers can be configured through the existing LLM configuration system:

```yaml
# llm_config.yaml
batch:
  providers:
    gemini:
      enabled: true
      models:
        - name: 'gemini-2.0-flash'
          mode: 'chat'
          max_batch_size: 1000
    claude:
      enabled: true
      models:
        - name: 'claude-3.5-sonnet'
          mode: 'chat'
          max_batch_size: 500
```

## Testing

The implementation includes comprehensive test coverage:

- **Unit Tests**: 27 test cases covering both providers
- **Integration Tests**: 12 test cases with mocked HTTP requests
- **Error Scenarios**: Transient and permanent error handling
- **Cross-Provider**: Compatibility and consistency testing

Run tests with:

```bash
# Unit tests
poetry run pytest tests/unit/llm/test_gemini_claude_providers.py

# Integration tests
poetry run pytest tests/integration/test_gemini_claude_batch_integration.py

# All batch-related tests
poetry run pytest tests/unit/llm/test_batch* tests/integration/test_*batch*
```

## Troubleshooting

### Common Issues

1. **Provider not found**: Ensure dependencies are installed

   ```bash
   poetry install  # Installs google-genai and anthropic SDKs
   ```

2. **API key errors**: Check environment variables and permissions

   ```python
   import os
   print(os.getenv('GOOGLE_GENAI_API_KEY'))  # Should not be None
   ```

3. **Model not supported**: Verify model name and provider compatibility

   ```python
   print(provider.supports_model("your-model", "chat"))
   ```

4. **Batch submission fails**: Check API key and model availability
   ```python
   # Test with minimal payload first
   test_items = [{"id": "test", "input_payload": {"content": "Hello"}}]
   ```

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('app.llm.batch.providers').setLevel(logging.DEBUG)
```

## Migration from Single Requests

If you're currently using single requests, here's how to migrate to batch processing:

```python
# Before: Single requests
responses = []
for item in items:
    response = llm_client.chat(item["content"])
    responses.append(response)

# After: Batch processing
batch_items = [
    {"id": f"item_{i}", "input_payload": {"content": item["content"]}}
    for i, item in enumerate(items)
]

provider = registry.create_provider("claude", {"api_key": "your_key"})
prepared = provider.prepare_payloads(batch_items, "claude-3.5-sonnet", "chat")
result = provider.submit(prepared)

# Poll and fetch results...
```

This migration provides significant cost savings (up to 50% for Claude, similar for Gemini) and better throughput for large-scale processing tasks.

## External Resources

- [Gemini Batch API Interface](https://ai.google.dev/api/batch-mode)
