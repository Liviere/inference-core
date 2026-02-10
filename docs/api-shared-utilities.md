# Shared API Utilities

This document describes the shared utilities available for API route development in `inference_core/api/v1/shared/`.

## Overview

The shared utilities module provides common functionality used across API routes to reduce code duplication, improve consistency, and make the codebase more maintainable.

## Available Utilities

### Error Handling

#### `@handle_api_errors`
Decorator for consistent error handling across API endpoints.

**Features:**
- Catches all exceptions and converts them to appropriate HTTP responses
- Re-raises `HTTPException` as-is to preserve status codes and details
- Converts unexpected exceptions to 500 Internal Server Error
- Logs unexpected errors with full stack traces

**Usage:**
```python
from inference_core.api.v1.shared import handle_api_errors

@router.get("/endpoint")
@handle_api_errors
async def my_endpoint():
    # Your endpoint logic
    # Any unexpected exceptions will be handled automatically
    return {"status": "success"}
```

### Response Utilities

#### `get_iso_timestamp()`
Get current UTC timestamp in ISO 8601 format.

**Returns:** String timestamp (e.g., `"2024-01-15T10:30:00.123456+00:00"`)

**Usage:**
```python
from inference_core.api.v1.shared import get_iso_timestamp

timestamp = get_iso_timestamp()
return {"created_at": timestamp}
```

**Replaces:**
```python
# Old pattern (now deprecated in routes)
str(datetime.now(UTC).isoformat())
```

#### `create_pagination_response(items, total_count, offset=0, limit=10)`
Create a standardized pagination response.

**Parameters:**
- `items`: List of items for the current page
- `total_count`: Total number of items available
- `offset`: Current offset (default: 0)
- `limit`: Items per page (default: 10)

**Returns:** Dictionary with pagination metadata

**Usage:**
```python
from inference_core.api.v1.shared import create_pagination_response

items = await service.get_items(offset=0, limit=10)
total = await service.count_items()

return create_pagination_response(
    items=items,
    total_count=total,
    offset=0,
    limit=10
)
# Returns:
# {
#     "items": [...],
#     "total": 100,
#     "offset": 0,
#     "limit": 10,
#     "has_more": True
# }
```

#### `serialize_model(model, exclude_unset=False, exclude_none=False)`
Serialize a Pydantic model to a dictionary.

**Parameters:**
- `model`: Pydantic model to serialize
- `exclude_unset`: Exclude fields that were not explicitly set
- `exclude_none`: Exclude fields with None values

**Returns:** Dictionary representation of the model

### Authentication Utilities

#### `get_user_id_from_context(current_user)`
Extract and convert user ID from the current user context.

**Parameters:**
- `current_user`: Dictionary containing user information from JWT/auth

**Returns:** UUID representation of the user ID

**Raises:** `HTTPException(401)` if user ID is missing or invalid

**Usage:**
```python
from inference_core.api.v1.shared import get_user_id_from_context

@router.post("/resource")
async def create_resource(
    current_user: dict = Depends(get_current_active_user),
):
    user_id = get_user_id_from_context(current_user)
    # user_id is now a UUID object
    resource = await service.create(created_by=user_id)
    return resource
```

**Replaces:**
```python
# Old pattern (now deprecated in routes)
user_id = UUID(current_user["id"])
```

#### `verify_resource_ownership(resource_owner_id, current_user_id, resource_type="Resource", resource_id=None)`
Verify that the current user owns the specified resource.

**Parameters:**
- `resource_owner_id`: UUID of the resource owner
- `current_user_id`: UUID of the current user
- `resource_type`: Type of resource for error message (default: "Resource")
- `resource_id`: Optional resource ID for error message

**Raises:** `HTTPException(404)` if ownership verification fails

**Usage:**
```python
from inference_core.api.v1.shared import (
    get_user_id_from_context,
    verify_resource_ownership
)

@router.get("/jobs/{job_id}")
async def get_job(
    job_id: UUID,
    current_user: dict = Depends(get_current_active_user),
    service: Service = Depends(get_service),
):
    job = await service.get_job(job_id)
    user_id = get_user_id_from_context(current_user)
    
    if not job:
        raise HTTPException(404, "Job not found")
    
    verify_resource_ownership(
        resource_owner_id=job.created_by,
        current_user_id=user_id,
        resource_type="Job",
        resource_id=str(job_id),
    )
    
    return job
```

**Replaces:**
```python
# Old pattern (now deprecated in routes)
user_id = UUID(current_user["id"])
if not job or job.created_by != user_id:
    raise HTTPException(404, f"Job {job_id} not found")
```

## Benefits

### Code Reduction
- **20+ instances** of duplicate code patterns eliminated across routes
- Average **2-3 lines reduced** per usage site

### Consistency
- Uniform error messages and status codes
- Standardized timestamp formats
- Consistent authentication patterns

### Maintainability
- Single source of truth for common functionality
- Easier to update behavior across all routes
- Better testability through unit tests

### Type Safety
- Proper UUID type conversion with validation
- Type hints on all utility functions
- Clear error messages for invalid inputs

## Migration Guide

When creating new routes or updating existing ones:

1. **Import shared utilities:**
   ```python
   from inference_core.api.v1.shared import (
       get_iso_timestamp,
       get_user_id_from_context,
       verify_resource_ownership,
   )
   ```

2. **Replace timestamp generation:**
   ```python
   # Before
   timestamp = str(datetime.now(UTC).isoformat())
   
   # After
   timestamp = get_iso_timestamp()
   ```

3. **Replace user ID extraction:**
   ```python
   # Before
   user_id = UUID(current_user["id"])
   
   # After
   user_id = get_user_id_from_context(current_user)
   ```

4. **Replace ownership verification:**
   ```python
   # Before
   if not resource or resource.owner_id != user_id:
       raise HTTPException(404, "Resource not found")
   
   # After
   if not resource:
       raise HTTPException(404, "Resource not found")
   verify_resource_ownership(
       resource_owner_id=resource.owner_id,
       current_user_id=user_id,
       resource_type="Resource",
       resource_id=str(resource_id),
   )
   ```

## Testing

All shared utilities have comprehensive unit tests in `tests/unit/api/v1/shared/test_shared_utils.py`.

Run tests:
```bash
poetry run pytest tests/unit/api/v1/shared/ -v
```

## Current Usage

The following routes currently use shared utilities:

- ✅ `health.py` - timestamp generation
- ✅ `batch.py` - user ID extraction, ownership verification
- ✅ `agent_instances.py` - user ID extraction
- ✅ `auth.py` - timestamp generation
- ✅ `config.py` - user ID extraction
