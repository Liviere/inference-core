## Test Suite Documentation

### Test Structure

The test suite is organized as follows:

- `app/tests/conftest.py` - Test configuration and shared fixtures
- `app/tests/integration/` - Integration tests for API endpoints and database operations

### Running Tests

To run all tests:

```bash
poetry run pytest
```

More specific test commands can be obtained by running:

```bash
poetry run pytest --help
```

### Test Fixtures

The project provides several useful fixtures:

#### Database Fixtures

- **`test_settings`** - Provides application settings configured for testing (sets `ENVIRONMENT=testing`)
- **`async_engine`** - Creates a temporary database engine with proper cleanup
- **`async_session_with_engine`** - Provides a database session with automatic table creation and cleanup
- **`async_test_client`** - HTTP test client with database dependency injection

#### Automatic Table Creation

The test fixtures automatically create database tables if they don't exist, ensuring tests can run on a clean database without manual setup.

### Example Test Usage

```python
import pytest
from sqlalchemy import text

@pytest.mark.asyncio
async def test_database_operations(async_session_with_engine):
    """Test database operations with automatic table creation."""
    session, engine = async_session_with_engine

    # Tables are automatically created by the fixture
    result = await session.execute(text("SELECT 1"))
    assert result.scalar() == 1

@pytest.mark.asyncio
async def test_api_endpoint(async_test_client):
    """Test API endpoints with test client."""
    client = async_test_client

    response = await client.get("/api/v1/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### Test Database Configuration

Tests use the same database configuration as the main application but with:

- Environment set to "testing"
- Automatic table creation before each test module
- Proper connection cleanup to prevent resource leaks
- Isolated database sessions for each test
