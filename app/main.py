import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI

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

    ###################################
    #            Routes               #
    ###################################

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


# Create app instance
app = create_application()
