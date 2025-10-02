import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api.v1.routes import auth, batch, health, llm, metrics, tasks, vector
from .core.config import Settings, get_settings
from .core.lifecycle import init_resources, shutdown_resources
from .core.logging_config import setup_logging

###################################
#            Functions            #
###################################


def lifespan(settings: Settings):
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """
        Application lifespan handler

        Handles startup and shutdown events for the FastAPI application.
        """

        # Startup
        setup_logging()
        logging.info("🚀 Starting up FastAPI application...")
        if settings.debug:
            logging.debug("Debug mode is enabled")

        if settings.is_development:
            logging.info("Running in development mode")

        elif settings.is_testing:
            logging.info("Running in testing mode")

        elif settings.is_production:
            logging.info("Running in production mode")

            if settings.sentry_dsn:
                import sentry_sdk

                sentry_sdk.init(
                    dsn=settings.sentry_dsn,
                    # Set traces_sample_rate to capture performance traces
                    # Adjust this value in production based on your traffic volume
                    traces_sample_rate=settings.sentry_traces_sample_rate,
                    # Set profiles_sample_rate to profile a subset of transactions
                    profiles_sample_rate=settings.sentry_profiles_sample_rate,
                    # Set environment to distinguish between different stages
                    environment=settings.environment,
                    # Set release to track deployments
                    release=settings.app_version,
                    # Send default PII (personally identifiable information)
                    # Be careful with this in production - consider data privacy requirements
                    send_default_pii=True,
                    # Enable auto session tracking
                    auto_session_tracking=True,
                    # Attach stack traces to all messages
                    attach_stacktrace=True,
                )
                logging.info(
                    f"Sentry initialized for environment: {settings.environment}"
                )
            else:
                logging.warning("Sentry DSN not configured for production environment")

        # --- Resource Warm-Up & Health Probes (Startup) ---
        if not settings.is_testing:
            try:
                await init_resources(settings)
            except Exception:
                # init_resources already logged critical failure; re-raise to abort startup
                raise

        yield
        # Shutdown
        logging.info("🛑 Shutting down FastAPI application...")

        try:
            await shutdown_resources(settings)
        except Exception:
            # shutdown_resources is defensive; swallow to not mask original shutdown reasons
            pass

    return _lifespan


def create_application(
    external_routers: Dict[str, APIRouter] = None, custom_settings: Settings = None
) -> FastAPI:
    """
    Create and configure FastAPI application

    Returns:
        Configured FastAPI application
    """

    if not custom_settings:
        get_settings.cache_clear()
        settings = get_settings()
    else:
        settings = custom_settings
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_title,
        description=settings.app_description,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan(settings),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    # Add middleware
    setup_middleware(app, settings)

    ###################################
    #             Routes              #
    ###################################

    # Configure API routers explicitly
    setup_routers(app, external_routers)

    # Mount simple static test assets (only in debug/development) for LLM streaming manual QA
    # These assets provide a lightweight in-browser UI to exercise SSE streaming endpoints.
    test_frontend_url = None
    try:
        if settings.debug:
            frontend_dir = Path(__file__).parent / "frontend"
            app.mount(
                "/static/frontend/",
                StaticFiles(directory=str(frontend_dir), html=True),
                name="static",
            )
            logging.info("✅ Mounted static test assets at /static/frontend/")
            # Decide what the public test frontend URL should be.
            index_exists = (frontend_dir / "index.html").exists()
            stream_exists = (frontend_dir / "stream.html").exists()

            if index_exists:
                test_frontend_url = f"{settings.app_public_url}/static/frontend/"
            elif stream_exists:
                # If there is no index.html but there is stream.html, expose a small
                # redirect so `/static/frontend/` opens the actual page.
                @app.get("/static/frontend/", include_in_schema=False)
                async def _frontend_index_redirect():
                    return RedirectResponse("/static/frontend/stream.html")

                test_frontend_url = (
                    f"{settings.app_public_url}/static/frontend/stream.html"
                )
            else:
                # No obvious entry point found
                test_frontend_url = None
    except Exception as e:
        logging.warning(f"Could not mount static test assets: {e}")

    # Add root endpoint
    @app.get("/", tags=["Root"])
    async def root(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
        """Root endpoint"""
        return {
            "message": f"Welcome to {settings.app_title}",
            "app_name": settings.app_name,
            "app_description": settings.app_description,
            "app_title": settings.app_title,
            "app_public_url": settings.app_public_url,
            "version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug,
            "docs": "/docs" if not settings.is_production else "disabled",
            "test_frontend": test_frontend_url,
        }

    return app


def setup_routers(app: FastAPI, external_routers: Dict[str, APIRouter] = None) -> None:
    """
    Setup all application routers explicitly

    Args:
        app: FastAPI application
    """
    # Create API v1 router
    api_v1 = APIRouter(prefix="/api/v1")

    # Include all v1 endpoints with explicit configuration
    api_v1.include_router(health.router)
    api_v1.include_router(auth.router)
    api_v1.include_router(tasks.router)
    api_v1.include_router(llm.router)
    api_v1.include_router(batch.router)
    api_v1.include_router(vector.router)

    # Include main API router
    app.include_router(api_v1)

    # Add metrics endpoint at root level (outside API versioning)
    app.include_router(metrics.router)

    # Future: Add other API versions here
    # api_v2 = APIRouter(prefix="/api/v2")
    # app.include_router(api_v2)

    # Include any external routers passed to the function
    if external_routers:
        for prefix, router in external_routers.items():
            app.include_router(router, prefix=prefix)


def setup_middleware(app: FastAPI, settings) -> None:
    """
    Setup application middleware

    Args:
        app: FastAPI application
        settings: Application settings
    """
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )

    # Trusted host middleware for production
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.get_effective_allowed_hosts(),
        )


# Create app instance for ASGI servers
app = create_application()
