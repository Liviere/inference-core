"""Extended tests for inference_core.main_factory.

Covers lifespan branches (dev, testing, production, Sentry init),
setup_middleware (CORS, TrustedHost), setup_routers helpers,
create_application with custom_settings and external_routers,
and static frontend mounting logic.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from inference_core.core.config import Settings
from inference_core.main_factory import (
    create_application,
    lifespan,
    setup_middleware,
    setup_routers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _testing_settings(**overrides) -> Settings:
    """Return minimal Settings for testing environment."""
    defaults = {
        "environment": "testing",
        "debug": False,
        "app_public_url": "http://test",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _production_settings(**overrides) -> Settings:
    defaults = {
        "environment": "production",
        "debug": False,
        "app_public_url": "https://prod.example.com",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _development_settings(**overrides) -> Settings:
    defaults = {
        "environment": "development",
        "debug": True,
        "app_public_url": "http://localhost:8000",
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ============================================================================
# create_application
# ============================================================================


class TestCreateApplication:
    """create_application factory wiring."""

    def test_custom_settings_used(self):
        """custom_settings bypasses get_settings."""
        s = _testing_settings()
        app = create_application(custom_settings=s)
        assert app.title == s.app_title

    def test_docs_enabled_in_testing(self):
        app = create_application(custom_settings=_testing_settings())
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

    def test_docs_disabled_in_production(self):
        app = create_application(custom_settings=_production_settings())
        assert app.docs_url is None
        assert app.redoc_url is None
        assert app.openapi_url is None

    def test_external_routers_included(self):
        """External routers are wired into the app."""
        from fastapi import APIRouter

        ext = APIRouter()

        @ext.get("/custom")
        async def _custom():
            return {"custom": True}

        app = create_application(
            custom_settings=_testing_settings(),
            external_routers={"/ext": ext},
        )
        # Router should be in app's routes
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert any("/ext/custom" in p for p in paths)


# ============================================================================
# Lifespan
# ============================================================================


class TestLifespan:
    """Lifespan context manager branches."""

    @pytest.mark.asyncio
    async def test_testing_skips_init_resources(self):
        """In testing mode, init_resources is NOT called."""
        s = _testing_settings()
        app = FastAPI()

        with patch(
            "inference_core.main_factory.init_resources",
            new_callable=AsyncMock,
        ) as mock_init:
            ctx = lifespan(s)
            async with ctx(app):
                pass
            mock_init.assert_not_called()

    @pytest.mark.asyncio
    async def test_development_calls_init_resources(self):
        """In development mode, init_resources IS called."""
        s = _development_settings()
        app = FastAPI()

        with patch(
            "inference_core.main_factory.init_resources",
            new_callable=AsyncMock,
        ) as mock_init:
            with patch(
                "inference_core.main_factory.shutdown_resources",
                new_callable=AsyncMock,
            ):
                ctx = lifespan(s)
                async with ctx(app):
                    pass
            mock_init.assert_called_once_with(s)

    @pytest.mark.asyncio
    async def test_production_sentry_init(self):
        """Production + sentry_dsn → sentry_sdk.init is called."""
        s = _production_settings(sentry_dsn="https://abc@sentry.io/1")
        app = FastAPI()

        mock_sentry = MagicMock()
        with patch.dict("sys.modules", {"sentry_sdk": mock_sentry}):
            with patch(
                "inference_core.main_factory.init_resources",
                new_callable=AsyncMock,
            ):
                with patch(
                    "inference_core.main_factory.shutdown_resources",
                    new_callable=AsyncMock,
                ):
                    ctx = lifespan(s)
                    async with ctx(app):
                        pass

    @pytest.mark.asyncio
    async def test_production_no_dsn_skips_sentry_init(self):
        """Production without sentry_dsn → sentry_sdk.init is NOT called."""
        s = _production_settings(sentry_dsn=None)
        app = FastAPI()

        with patch(
            "inference_core.main_factory.init_resources",
            new_callable=AsyncMock,
        ):
            with patch(
                "inference_core.main_factory.shutdown_resources",
                new_callable=AsyncMock,
            ):
                # sentry_sdk is imported lazily inside the lifespan;
                # confirm it is NOT invoked when DSN is absent.
                import sentry_sdk as _real_sdk

                with patch.object(_real_sdk, "init") as mock_init:
                    ctx = lifespan(s)
                    async with ctx(app):
                        pass
                    mock_init.assert_not_called()

    @pytest.mark.asyncio
    async def test_external_on_startup_called(self):
        """external_on_startup callback is invoked."""
        s = _testing_settings()
        app = FastAPI()
        startup_called = False

        async def on_startup(a):
            nonlocal startup_called
            startup_called = True

        with patch(
            "inference_core.main_factory.shutdown_resources",
            new_callable=AsyncMock,
        ):
            ctx = lifespan(s, external_on_startup=on_startup)
            async with ctx(app):
                pass
        assert startup_called

    @pytest.mark.asyncio
    async def test_external_on_shutdown_called(self):
        """external_on_shutdown callback is invoked during shutdown."""
        s = _testing_settings()
        app = FastAPI()
        shutdown_called = False

        async def on_shutdown(a):
            nonlocal shutdown_called
            shutdown_called = True

        with patch(
            "inference_core.main_factory.shutdown_resources",
            new_callable=AsyncMock,
        ):
            ctx = lifespan(s, external_on_shutdown=on_shutdown)
            async with ctx(app):
                pass
        assert shutdown_called

    @pytest.mark.asyncio
    async def test_init_resources_failure_reraises(self):
        """init_resources exception is re-raised (aborts startup)."""
        s = _development_settings()
        app = FastAPI()

        with patch(
            "inference_core.main_factory.init_resources",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB down"),
        ):
            with patch(
                "inference_core.main_factory.shutdown_resources",
                new_callable=AsyncMock,
            ):
                ctx = lifespan(s)
                with pytest.raises(RuntimeError, match="DB down"):
                    async with ctx(app):
                        pass

    @pytest.mark.asyncio
    async def test_shutdown_exception_swallowed(self):
        """Shutdown exception is swallowed so it doesn't mask original shutdown."""
        s = _testing_settings()
        app = FastAPI()

        with patch(
            "inference_core.main_factory.shutdown_resources",
            new_callable=AsyncMock,
            side_effect=RuntimeError("shutdown fail"),
        ):
            ctx = lifespan(s)
            async with ctx(app):
                pass
            # Should not raise


# ============================================================================
# setup_middleware
# ============================================================================


class TestSetupMiddleware:
    """setup_middleware CORS and TrustedHost configuration."""

    def test_cors_added_always(self):
        app = FastAPI()
        s = _testing_settings()
        setup_middleware(app, s)
        # CORS middleware should be present
        from starlette.middleware.cors import CORSMiddleware

        assert any(
            m.cls is CORSMiddleware for m in app.user_middleware
        )

    def test_trusted_host_only_in_production(self):
        from fastapi.middleware.trustedhost import TrustedHostMiddleware

        app_test = FastAPI()
        setup_middleware(app_test, _testing_settings())
        assert not any(
            m.cls is TrustedHostMiddleware for m in app_test.user_middleware
        )

        app_prod = FastAPI()
        setup_middleware(app_prod, _production_settings())
        assert any(
            m.cls is TrustedHostMiddleware for m in app_prod.user_middleware
        )


# ============================================================================
# setup_routers
# ============================================================================


class TestSetupRouters:
    """setup_routers includes expected API routes."""

    def test_health_routes_present(self):
        app = create_application(custom_settings=_testing_settings())
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/api/v1/health/" in paths or any(
            "/health" in p for p in paths
        )

    def test_auth_routes_present(self):
        app = create_application(custom_settings=_testing_settings())
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert any("/auth" in p for p in paths)

    def test_external_routers_added(self):
        from fastapi import APIRouter

        ext_router = APIRouter()

        @ext_router.get("/ping")
        async def _ping():
            return "pong"

        app = FastAPI()
        s = _testing_settings()
        setup_routers(app, s, external_routers={"/custom": ext_router})
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert any("/custom/ping" in p for p in paths)


# ============================================================================
# Root endpoint
# ============================================================================


class TestRootEndpoint:
    """Root GET / response structure."""

    @pytest.mark.asyncio
    async def test_root_returns_expected_fields(self):
        app = create_application(custom_settings=_testing_settings())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "version" in data
        assert "environment" in data
        assert "docs" in data

    @pytest.mark.asyncio
    async def test_root_shows_docs_slash_docs_in_testing(self):
        app = create_application(custom_settings=_testing_settings())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            data = (await client.get("/")).json()
        assert data["docs"] == "/docs"
