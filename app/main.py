import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .api.v1.routes import health
from .core.config import get_settings

###################################
#            Functions            #
###################################


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler

    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logging.info("🚀 Starting up FastAPI application...")

    yield
    # Shutdown
    logging.info("🛑 Shutting down FastAPI application...")


def create_application() -> FastAPI:
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
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
    )

    # Add middleware
    setup_middleware(app, settings)

    ###################################
    #             Routes              #
    ###################################

    # Configure API routers explicitly
    setup_routers(app)

    # Add root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> Dict[str, Any]:
        """Root endpoint"""
        return {
            "message": f"Welcome to {settings.app_title}",
            "app_name": settings.app_name,
            "app_description": settings.app_description,
            "app_title": settings.app_title,
            "version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug,
            "docs": "/docs" if settings.is_development else "disabled",
        }

    return app


def setup_routers(app: FastAPI) -> None:
    """
    Setup all application routers explicitly

    Args:
        app: FastAPI application
    """
    # Create API v1 router
    api_v1 = APIRouter(prefix="/api/v1")

    # Include all v1 endpoints with explicit configuration
    api_v1.include_router(health.router)

    # Include main API router
    app.include_router(api_v1)

    # Future: Add other API versions here
    # api_v2 = APIRouter(prefix="/api/v2")
    # app.include_router(api_v2)


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
            allowed_hosts=(
                settings.cors_origins if settings.cors_origins != ["*"] else ["*"]
            ),
        )


# Create app instance
app = create_application()
