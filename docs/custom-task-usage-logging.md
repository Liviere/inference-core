# Custom LLM Tasks Usage/Cost Logging

This guide explains how to use the generic usage/cost logging abstraction for custom LLM tasks beyond the built-in `explain` and `conversation` tasks.

## Overview

The `inference_core.llm.custom_task` module provides reusable helpers that enable consistent usage and cost logging for any custom LLM task (extraction, summarization, classification, etc.) without duplicating boilerplate code.

### Features

- **Automatic usage tracking**: Token counts and costs are automatically logged
- **Error handling**: Proper session finalization on success and failure
- **Streaming support**: Both sync and streaming execution modes
- **Pricing integration**: Respects pricing config from `llm_config.yaml`
- **Opt-in**: Works seamlessly with existing usage logging configuration

## Quick Start

### Basic Sync Execution

```python
from inference_core.llm.custom_task import run_with_usage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Create your custom chain (LCEL)
extraction_prompt = ChatPromptTemplate.from_template(
    "Extract named entities from the following text:\n\n{text}"
)
model = # ... your LangChain model
chain = extraction_prompt | model | StrOutputParser()

# Execute with usage tracking
result = await run_with_usage(
    task_type="extraction",  # Custom task type for logging
    runnable=chain,
    input={"text": "Apple Inc. announced new products in California."},
    model_name="openai/gpt-4o-mini",
    request_mode="sync",
    session_id=f"user:{user_id}",
    request_id="unique-request-id",
)

# result contains the extraction output
print(result)
```

### Streaming Execution

```python
from inference_core.llm.custom_task import stream_with_usage

# Create your streaming chain
summarization_prompt = ChatPromptTemplate.from_template(
    "Summarize the following document:\n\n{document}"
)
streaming_model = # ... your LangChain model with streaming enabled
chain = summarization_prompt | streaming_model | StrOutputParser()

# Execute with streaming and usage tracking
async for chunk in stream_with_usage(
    task_type="summarization",
    runnable=chain,
    input={"document": long_document_text},
    model_name="openai/gpt-4o-mini",
    session_id=f"user:{user_id}",
):
    print(chunk, end="", flush=True)
```

## API Reference

### `run_with_usage`

Execute a LangChain Runnable with automatic usage/cost logging (sync mode).

**Parameters:**

- `task_type` (str, required): Task identifier (e.g., "extraction", "summarization", "classification")
- `runnable` (Any, required): LangChain Runnable/Chain/Model to execute
- `input` (Dict[str, Any], required): Input payload for `runnable.ainvoke()`
- `model_name` (str, required): Model name for pricing lookup
- `request_mode` (str, optional): "sync" or "streaming" (default: "sync")
- `session_id` (str, optional): Session identifier for grouping related requests
- `user_id` (str, optional): User UUID string (converted to UUID internally)
- `request_id` (str, optional): Unique request identifier
- `extra_callbacks` (list, optional): Additional app-specific callbacks to include

**Returns:** Result from `runnable.ainvoke()`

**Raises:** Any exception from runnable execution is re-raised after finalizing the usage session

### `stream_with_usage`

Execute a LangChain Runnable in streaming mode with usage/cost logging.

**Parameters:**

- `task_type` (str, required): Task identifier
- `runnable` (Any, required): LangChain Runnable with streaming support
- `input` (Dict[str, Any], required): Input payload for `runnable.astream()`
- `model_name` (str, required): Model name for pricing lookup
- `session_id` (str, optional): Session identifier
- `user_id` (str, optional): User UUID string
- `request_id` (str, optional): Unique request identifier
- `extra_callbacks` (list, optional): Additional callbacks

**Yields:** Chunks from `runnable.astream()`

**Raises:** Any exception from runnable execution is re-raised after finalizing the usage session

## Usage Examples

### Example 1: Entity Extraction Task

```python
from inference_core.llm.custom_task import run_with_usage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from inference_core.llm.models import get_model_factory

# Create extraction chain
factory = get_model_factory()
model = factory.create_model("openai/gpt-4o-mini")

extraction_prompt = ChatPromptTemplate.from_template("""
Extract entities from the text and return a JSON object with keys:
- people: list of person names
- organizations: list of organization names
- locations: list of location names

Text: {text}
""")

chain = extraction_prompt | model | JsonOutputParser()

# Execute with usage logging
entities = await run_with_usage(
    task_type="entity_extraction",
    runnable=chain,
    input={"text": "Elon Musk announced Tesla's new factory in Austin, Texas."},
    model_name="openai/gpt-4o-mini",
    user_id=str(current_user.id),
    request_id=f"extraction-{uuid.uuid4()}",
)

# entities = {
#     "people": ["Elon Musk"],
#     "organizations": ["Tesla"],
#     "locations": ["Austin", "Texas"]
# }
```

### Example 2: Document Summarization (Streaming)

```python
from inference_core.llm.custom_task import stream_with_usage

# Create summarization chain with streaming
summarization_prompt = ChatPromptTemplate.from_template("""
Provide a concise summary of the following document:

{document}

Summary:
""")

streaming_model = factory.create_model(
    "openai/gpt-4o-mini",
    streaming=True,
    max_tokens=500,
)

chain = summarization_prompt | streaming_model | StrOutputParser()

# Stream summary with usage tracking
full_summary = ""
async for chunk in stream_with_usage(
    task_type="summarization",
    runnable=chain,
    input={"document": long_document},
    model_name="openai/gpt-4o-mini",
    session_id=f"doc-summary-{document_id}",
):
    full_summary += chunk
    # Stream to client in real-time
    await websocket.send_text(chunk)
```

### Example 3: Sentiment Analysis

```python
from pydantic import BaseModel

# Define structured output
class SentimentAnalysis(BaseModel):
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float  # 0.0 to 1.0
    reasoning: str

# Create sentiment analysis chain with structured output
sentiment_prompt = ChatPromptTemplate.from_template("""
Analyze the sentiment of the following text:

{text}

Provide your analysis in JSON format.
""")

structured_model = model.with_structured_output(SentimentAnalysis)
chain = sentiment_prompt | structured_model

# Execute with usage logging
analysis = await run_with_usage(
    task_type="sentiment_analysis",
    runnable=chain,
    input={"text": "This product exceeded my expectations!"},
    model_name="openai/gpt-4o-mini",
    request_id=f"sentiment-{review_id}",
)

# analysis.sentiment = "positive"
# analysis.confidence = 0.95
```

### Example 4: Using Extra Callbacks

```python
from langchain_core.callbacks import BaseCallbackHandler

# Define custom callback for additional tracking
class CustomMetricsCallback(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        # Track custom metrics
        metrics.increment("custom_task.started")
    
    def on_llm_end(self, *args, **kwargs):
        metrics.increment("custom_task.completed")

# Use with extra callbacks
custom_callback = CustomMetricsCallback()

result = await run_with_usage(
    task_type="classification",
    runnable=classification_chain,
    input={"text": "Classify this text"},
    model_name="openai/gpt-4o-mini",
    extra_callbacks=[custom_callback],
)
```

## Configuration

### Pricing Configuration

The helpers automatically read pricing from `llm_config.yaml`:

```yaml
models:
  openai/gpt-4o-mini:
    provider: 'openai'
    pricing:
      currency: USD
      input:
        cost_per_1k: 0.00015  # $0.15 per 1M tokens
      output:
        cost_per_1k: 0.0006   # $0.60 per 1M tokens
```

### Usage Logging Configuration

Control usage logging via settings in `llm_config.yaml`:

```yaml
settings:
  usage_logging:
    enabled: true              # Enable/disable usage logging
    base_currency: USD         # Base currency for costs
    fail_open: true            # Continue on logging errors
    default_rounding_decimals: 6  # Decimal places for cost rounding
```

Or via environment variables:

```bash
export LLM_USAGE_LOGGING_ENABLED=true
export LLM_USAGE_FAIL_OPEN=true
```

## Database Schema

Usage logs are stored in the `llm_request_logs` table with the following custom task fields:

- `task_type`: Your custom task identifier (e.g., "extraction", "summarization")
- `request_mode`: "sync" or "streaming"
- `model_name`: Model used for execution
- `provider`: Provider (e.g., "openai", "anthropic")
- `session_id`: Optional session grouping
- `request_id`: Optional unique request identifier
- `input_tokens`, `output_tokens`, `total_tokens`: Token usage
- `cost_input_usd`, `cost_output_usd`, `cost_total_usd`: Cost breakdown
- `success`: Whether execution succeeded
- `error_type`, `error_message`: Error details if failed
- `streamed`: Whether response was streamed
- `partial`: Whether streaming was interrupted

## Best Practices

### 1. Use Descriptive Task Types

Choose clear, consistent task type names:

```python
# Good
task_type="entity_extraction"
task_type="document_summarization"
task_type="sentiment_analysis"

# Avoid
task_type="task1"
task_type="custom"
```

### 2. Include Request IDs for Tracing

Use unique request IDs to correlate logs with application traces:

```python
import uuid

request_id = f"extraction-{uuid.uuid4()}"
result = await run_with_usage(
    task_type="extraction",
    runnable=chain,
    input=input_data,
    model_name=model_name,
    request_id=request_id,
)
```

### 3. Group Related Requests with Session IDs

Use session IDs for multi-turn or related operations:

```python
session_id = f"user:{user_id}:document:{doc_id}"

# All operations on this document use the same session
summary = await run_with_usage(..., session_id=session_id)
entities = await run_with_usage(..., session_id=session_id)
```

### 4. Handle Errors Gracefully

The helpers re-raise errors after logging, so handle them in your application:

```python
try:
    result = await run_with_usage(
        task_type="extraction",
        runnable=chain,
        input=input_data,
        model_name=model_name,
    )
except Exception as e:
    logger.error(f"Extraction failed: {e}")
    # Return fallback or error response
    return {"error": "Extraction service unavailable"}
```

### 5. Use Streaming for Long Responses

For tasks with long outputs, use streaming to improve user experience:

```python
async def stream_response(request):
    """Stream summarization to client"""
    async for chunk in stream_with_usage(
        task_type="summarization",
        runnable=summary_chain,
        input={"document": request.document},
        model_name="openai/gpt-4o-mini",
    ):
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
```

## Troubleshooting

### Usage Not Being Logged

1. **Check configuration**: Ensure `usage_logging.enabled` is `true` in `llm_config.yaml`
2. **Verify model config**: Confirm the model exists in `llm_config.yaml`
3. **Check database**: Ensure database migrations are up to date
4. **Review logs**: Look for errors in application logs (fail_open mode swallows DB errors)

### Incorrect Cost Calculations

1. **Verify pricing config**: Check that pricing is defined for the model in `llm_config.yaml`
2. **Check model name**: Ensure exact match between `model_name` parameter and config
3. **Review token usage**: Verify callback is receiving usage metadata from LangChain

### Missing Token Counts

The callback relies on LangChain models emitting usage metadata:

1. **Check model support**: Not all LangChain models emit usage metadata
2. **Verify LangChain version**: Ensure you're using a recent version
3. **Test with OpenAI**: OpenAI models consistently emit usage metadata

## Migration from Built-in Tasks

If you've implemented custom tasks that duplicate usage logging code, migrate to use these helpers:

**Before:**

```python
# Duplicated boilerplate
usage_logger = UsageLogger(config.usage_logging)
session = usage_logger.start_session(...)
callback = LLMUsageCallbackHandler(session, pricing)
try:
    result = await chain.ainvoke(input, config={"callbacks": [callback]})
    await session.finalize(success=True, ...)
    return result
except Exception as e:
    await session.finalize(success=False, error=e, ...)
    raise
```

**After:**

```python
# Clean, reusable helper
result = await run_with_usage(
    task_type="extraction",
    runnable=chain,
    input=input,
    model_name=model_name,
)
return result
```

## See Also

- [Usage Logging Configuration](configuration.md#usage-logging)
- [LLM Models Documentation](models.md)
- [LangChain Callbacks](https://python.langchain.com/docs/modules/callbacks/)
