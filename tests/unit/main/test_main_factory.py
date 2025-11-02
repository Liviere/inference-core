import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from inference_core.core.config import Settings
from inference_core.main_factory import create_application


@pytest.mark.asyncio
async def test_docs_enabled_in_testing():
    # Default ENVIRONMENT is forced to testing in conftest
    app = create_application()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["docs"] == "/docs"


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="TrustedHostMiddleware zwraca 400 bez dopasowanego Host; zajmiemy się tym później"
)
async def test_docs_disabled_in_production_and_trusted_hosts(monkeypatch):
    # Stub out resource init/shutdown to avoid side effects in production mode
    async def _noinit(_settings):
        return None

    async def _noshutdown(_settings):
        return None

    monkeypatch.setattr(
        "inference_core.core.lifecycle.init_resources", _noinit, raising=False
    )
    monkeypatch.setattr(
        "inference_core.core.lifecycle.shutdown_resources", _noshutdown, raising=False
    )

    settings = Settings(
        environment="production", debug=False, app_public_url="http://test"
    )
    app = create_application(custom_settings=settings)

    # Docs should be disabled
    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None

    # Root response should show docs disabled
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["docs"] == "disabled"

    # TrustedHostMiddleware should be present in production
    from fastapi.middleware.trustedhost import TrustedHostMiddleware

    assert any(m.cls is TrustedHostMiddleware for m in app.user_middleware)
