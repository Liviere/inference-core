# OpenAI Batch Provider

This document describes the OpenAI Batch Provider implementation for the batch processing framework.

## Overview

The OpenAI Batch Provider (`OpenAIBatchProvider`) implements the `BaseBatchProvider` interface to provide batch processing capabilities using OpenAI's native `/v1/batches` API. This allows for cost-efficient, large-scale asynchronous processing of chat completion requests.

## Features

- **JSONL Construction**: Automatically formats batch items into OpenAI's JSONL format with `custom_id` mapping
- **File Management**: Handles file upload to OpenAI's Files API for batch input
- **Batch Submission**: Submits batches to OpenAI with proper configuration
- **Status Polling**: Maps OpenAI status to internal status representation
- **Result Retrieval**: Fetches and parses both successful results and error files
- **Error Handling**: Distinguishes between transient and permanent errors for proper retry logic

## Supported Models

The provider supports OpenAI chat completion models including:
- `gpt-3.5-turbo` and variants
- `gpt-4` and variants
- `gpt-4o` and variants  
- `gpt-5` and variants

Only `chat` mode is currently supported for batch processing.

## Configuration

### Environment Variables

Set the following environment variable:
- `OPENAI_API_KEY`: Your OpenAI API key

### Batch Configuration

The provider is configured in `llm_config.yaml` under the `batch.providers.openai` section:

```yaml
batch:
  providers:
    openai:
      enabled: true
      models:
        - name: gpt-5-mini
          mode: chat
          max_prompts_per_batch: 20
          pricing_tier: batch
```

## Usage

### Basic Usage

```python
from app.llm.batch import registry

# Create provider instance
config = {"api_key": "your-openai-api-key"}
provider = registry.create_provider("openai", config)

# Prepare batch items
batch_items = [
    {
        "id": "request-1",
        "input_payload": {
            "messages": [{"role": "user", "content": "What is AI?"}],
            "max_tokens": 100
        }
    },
    {
        "id": "request-2", 
        "input_payload": {
            "messages": [{"role": "user", "content": "Explain quantum computing"}],
            "max_tokens": 150
        }
    }
]

# Process through the provider
prepared = provider.prepare_payloads(batch_items, "gpt-4", "chat")
submit_result = provider.submit(prepared)
provider_batch_id = submit_result.provider_batch_id

# Poll for status
status = provider.poll_status(provider_batch_id)

# Fetch results when completed
if status.normalized_status == "completed":
    results = provider.fetch_results(provider_batch_id)
```

### Batch Item Format

Each batch item must have:
- `id`: Unique identifier for the item (used as `custom_id`)
- `input_payload`: The request data containing:
  - `messages`: Array of message objects for chat completion
  - Additional OpenAI parameters (optional): `max_tokens`, `temperature`, etc.

Example:
```python
batch_item = {
    "id": "unique-request-id", 
    "input_payload": {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }
}
```

## Status Mapping

OpenAI batch statuses are mapped to internal statuses:

| OpenAI Status | Internal Status | Description |
|---------------|-----------------|-------------|
| `validating` | `submitted` | Batch is being validated |
| `in_progress` | `in_progress` | Batch is being processed |
| `finalizing` | `in_progress` | Batch is being finalized |
| `completed` | `completed` | Batch completed successfully |
| `failed` | `failed` | Batch failed |
| `expired` | `failed` | Batch expired before completion |
| `cancelled` | `cancelled` | Batch was cancelled |

## Error Handling

The provider distinguishes between two types of errors:

### Transient Errors (Retry Recommended)
- Rate limiting (429 status codes)
- Network timeouts
- Temporary service unavailability (503, 502)
- General network issues

### Permanent Errors (No Retry)
- Invalid API key (401, 403)
- Unsupported model
- Malformed requests (400)
- Batch not found (404)

## Result Processing

Results are returned as `ProviderResultRow` objects containing:
- `custom_id`: Original batch item ID
- `output_text`: Extracted response content (for successful requests)
- `output_data`: Full OpenAI response data
- `raw_metadata`: Complete OpenAI response metadata
- `error_message`: Error details (for failed requests)
- `is_success`: Boolean indicating success/failure

### Partial Failures

The provider properly handles scenarios where some items succeed and others fail:
- Successful results are parsed from the output file
- Failed results are parsed from the error file
- Each result maintains its original `custom_id` for tracking

## Implementation Details

### JSONL Format

Batch items are converted to OpenAI's JSONL format:
```json
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4", "messages": [...]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4", "messages": [...]}}
```

### File Management

- Input files are uploaded with `purpose="batch"`
- File content is automatically cleaned up by OpenAI
- Result files are downloaded and parsed when batches complete

### Retry Logic

The provider automatically determines retry behavior based on error types:
- Transient errors include optional `retry_after` hints
- Rate limit errors default to 60-second retry delay
- Permanent errors are not retried

## Testing

The provider includes comprehensive test coverage:

### Unit Tests
- All provider methods with mocked OpenAI responses
- Error handling scenarios
- JSONL formatting validation
- Status mapping verification

### Integration Tests  
- End-to-end flow with ≥3 prompts
- Partial failure scenarios
- Status transition testing
- Registry integration

## Context7 Sources

Implementation based on:
- **OpenAI Batch API**: `/v1/batches` endpoints and JSONL format requirements
- **Files API**: `/v1/files` for upload and content retrieval