import uuid

import pytest
from sqlalchemy import select

from inference_core.database.sql.models.user import User

# Constant test UUIDs
TEST_USER_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
OTHER_USER_ID = uuid.UUID("87654321-4321-8765-4321-876543218765")


@pytest.fixture
async def auth_client(async_test_client_factory, async_engine):
    """
    Creates an authenticated client linked to TEST_USER_ID.
    Also seeds the database with the test user.
    """
    # 1. Seed the test user into the DB
    # We must do this because UserAgentInstance fk references users.id
    from inference_core.database.sql.connection import get_non_singleton_session_maker

    session_maker = get_non_singleton_session_maker(engine=async_engine)

    async with session_maker() as session:
        # Check if user exists (metadata is created by factory, but tables are empty)
        # Note: factory creates tables inside the generator, but we need
        # to inject data *after* tables are created.
        # This is tricky because async_test_client_factory manages its own cleanup.
        # However, async_test_client_factory yields the client.
        # We need a way to insert data inside the context of the running app/tables.
        pass

    # Better approach: Use dependency injection to grab the session from the app override?
    # Or just use the fact that async_test_client_factory re-creates metadata each time.

    # Let's write a wrapper fixture in the test function instead or inside here.

    # We will override get_current_active_user
    async def mock_get_current_user():
        return {
            "id": str(TEST_USER_ID),
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True,
            "is_superuser": False,
        }

    # Helper to insert user
    async def _setup_data(session):
        user = User(
            id=TEST_USER_ID,
            email="test@example.com",
            username="testuser",
            hashed_password="fake",
            is_active=True,
            is_superuser=False,
            is_verified=True,
        )
        session.add(user)
        # Add another user for isolation tests
        other = User(
            id=OTHER_USER_ID,
            email="other@example.com",
            username="otheruser",
            hashed_password="fake",
            is_active=True,
        )
        session.add(other)
        await session.commit()

    pass


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_templates_list(async_test_client_factory):
    """Test listing available agent templates."""

    from inference_core.core.dependecies import get_current_active_user

    # We don't strictly need a DB user for this endpoint as it just reads config,
    # assuming get_current_active_user doesn't query DB validation (it usually just decodes JWT).
    # But for safety, we mock the user dependency.

    async def mock_user():
        return {"id": str(TEST_USER_ID), "is_active": True}

    async for client in async_test_client_factory(
        dependency_overrides={get_current_active_user: mock_user}
    ):
        resp = await client.get("/api/v1/agent-instances/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        assert "available_models" in data
        assert len(data["available_models"]) > 0
        # Check structure
        t = data["templates"][0]
        assert "agent_name" in t
        assert "primary_model" in t


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_instance_crud_flow(async_test_client_factory):
    """Full CRUD flow test for agent instances."""

    from httpx import ASGITransport, AsyncClient

    from inference_core.core.config import get_settings
    from inference_core.core.dependecies import get_current_active_user, get_db
    from inference_core.database.sql.connection import (
        Base,
        create_database_engine,
        get_non_singleton_session_maker,
    )
    from inference_core.main_factory import create_application

    # Setup DB
    engine = create_database_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = get_non_singleton_session_maker(engine=engine)

    # Seed User
    async with session_maker() as session:
        user = User(
            id=TEST_USER_ID,
            email="test@example.com",
            username="test",
            hashed_password="x",
        )
        session.add(user)
        await session.commit()

    # Override Deps
    async def override_get_db():
        async with session_maker() as s:
            yield s

    async def override_user():
        return {
            "id": str(TEST_USER_ID),
            "username": "test",
            "email": "test@example.com",
        }

    app = create_application()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_active_user] = override_user

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:

        # 1. List Templates to get a valid base name
        t_resp = await client.get("/api/v1/agent-instances/templates")
        templates = t_resp.json()["templates"]
        base_agent = templates[0]["agent_name"]

        # 2. Create Instance
        payload = {
            "instance_name": "my-cool-agent",
            "display_name": "My Cool Agent",
            "base_agent_name": base_agent,
            "description": "Testing agent",
            "system_prompt_append": "Speak like a pirate.",
        }
        create_resp = await client.post("/api/v1/agent-instances", json=payload)
        assert create_resp.status_code == 201
        instance_data = create_resp.json()
        instance_id = instance_data["id"]
        assert instance_data["instance_name"] == "my-cool-agent"
        assert instance_data["system_prompt_append"] == "Speak like a pirate."

        # 3. List Instances
        list_resp = await client.get("/api/v1/agent-instances")
        assert list_resp.status_code == 200
        l_data = list_resp.json()
        assert l_data["total"] == 1
        assert l_data["instances"][0]["id"] == instance_id

        # 4. Get Instance
        get_resp = await client.get(f"/api/v1/agent-instances/{instance_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == instance_id

        # 5. Update Instance
        patch_payload = {"display_name": "Updated Name", "is_default": True}
        patch_resp = await client.patch(
            f"/api/v1/agent-instances/{instance_id}", json=patch_payload
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["display_name"] == "Updated Name"
        assert patch_resp.json()["is_default"] is True

        # 6. Delete Instance
        del_resp = await client.delete(f"/api/v1/agent-instances/{instance_id}")
        assert del_resp.status_code == 204

        # Verify it's gone from list (active_only=True by default)
        list_again = await client.get("/api/v1/agent-instances")
        assert list_again.json()["total"] == 0

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_instance_validation(async_test_client_factory):
    # Same setup as above, abbreviated
    from httpx import ASGITransport, AsyncClient

    from inference_core.core.dependecies import get_current_active_user, get_db
    from inference_core.database.sql.connection import (
        Base,
        create_database_engine,
        get_non_singleton_session_maker,
    )
    from inference_core.main_factory import create_application

    engine = create_database_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = get_non_singleton_session_maker(engine=engine)

    async with session_maker() as session:
        user = User(
            id=TEST_USER_ID,
            email="test@example.com",
            username="test",
            hashed_password="x",
        )
        session.add(user)
        await session.commit()

    async def override_get_db():
        async with session_maker() as s:
            yield s

    async def override_user():
        return {"id": str(TEST_USER_ID)}

    app = create_application()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_active_user] = override_user

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Invalid base agent
        payload = {
            "instance_name": "bad-agent",
            "display_name": "Bad",
            "base_agent_name": "non_existent_agent_123",
        }
        resp = await client.post("/api/v1/agent-instances", json=payload)
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"]

        # Invalid slug
        payload2 = {
            "instance_name": "Bad Name with Spaces",
            "display_name": "Bad",
            "base_agent_name": "marketing_agent",  # valid agent usually
        }
        resp2 = await client.post("/api/v1/agent-instances", json=payload2)
        assert resp2.status_code == 422  # Pydantic validation error

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
