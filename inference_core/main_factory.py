import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

from .api.v1.routes import auth, batch, health, llm, metrics, tasks, vector
from .core.config import get_settings
from .core.logging_config import setup_logging
from .database.sql.connection import close_database, create_tables

###################################
#            Functions            #
###################################


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler

    Handles startup and shutdown events for the FastAPI application.
    """

    settings = get_settings()

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
            logging.info(f"Sentry initialized for environment: {settings.environment}")
        else:
            logging.warning("Sentry DSN not configured for production environment")

    # Create database tables
    if not settings.is_testing:
        try:
            await create_tables()
            logging.info("✅ Database tables created successfully")
        except Exception as e:
            logging.error(f"❌ Failed to create database tables: {e}")
            raise

    yield
    # Shutdown
    logging.info("🛑 Shutting down FastAPI application...")

    if not settings.is_testing:
        try:
            await close_database()
            logging.info("✅ Database connections closed successfully")
        except Exception as e:
            logging.error(f"❌ Failed to close database connections: {e}")


def create_application(external_routers: Dict[str, APIRouter] = None) -> FastAPI:
    """
    Create and configure FastAPI application

    Returns:
        Configured FastAPI application
    """
    settings = get_settings()

    # Create FastAPI app
    app = FastAPI(
        title=settings.app_title,
        description=settings.app_description,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
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
    try:
        if settings.debug:
            app.mount(
                "/static",
                StaticFiles(directory="app/frontend", html=True),
                name="static",
            )
    except Exception as e:
        logging.warning(f"Could not mount static test assets: {e}")

    # Add root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> Dict[str, Any]:
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
