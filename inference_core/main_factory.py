import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .api.v1.routes import (
    agent_instances,
    auth,
    batch,
    config,
    embeddings,
    health,
    metrics,
    tasks,
    vector,
)
from .core.config import Settings, get_settings
from .core.dependecies import (
    get_current_active_user,
    get_current_superuser,
    get_optional_current_user,
)
from .core.lifecycle import init_resources, shutdown_resources
from .core.logging_config import setup_logging, shutdown_logging

###################################
#            Functions            #
###################################


def lifespan(settings: Settings, external_on_startup=None, external_on_shutdown=None):
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """
        Application lifespan handler

        Handles startup and shutdown events for the FastAPI application.
        """

        # Startup
        setup_logging(service="api")
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

                from inference_core.observability.sentry_scrubbing import (
                    privacy_sentry_options,
                )

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
                    # Enable auto session tracking
                    auto_session_tracking=True,
                    # Attach stack traces to all messages
                    attach_stacktrace=True,
                    # Privacy hardening: send_default_pii=False, no local
                    # variables, before_send/before_breadcrumb PII scrubbers.
                    **privacy_sentry_options(),
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

        if external_on_startup:
            await external_on_startup(app)

        yield
        # Shutdown
        logging.info("🛑 Shutting down FastAPI application...")

        try:
            await shutdown_resources(settings)
            if external_on_shutdown:
                await external_on_shutdown(app)
        except Exception:
            # shutdown_resources is defensive; swallow to not mask original shutdown reasons
            pass

        shutdown_logging()

    return _lifespan


def create_application(
    external_routers: Dict[str, APIRouter] = None,
    custom_settings: Settings = None,
    external_on_startup=None,
    external_on_shutdown=None,
    task_access_checker: Optional[tasks.TaskAccessChecker] = None,
) -> FastAPI:
    """
    Create and configure FastAPI application

    Args:
        task_access_checker: Optional ``(task_id, user) -> bool`` (sync or
            async) asked before a task's status or result is returned. A
            refusal answers 404. ``user`` is None for an anonymous caller.
            Without it, any caller allowed by ``tasks_api_access_mode`` may
            read any task.

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
        lifespan=lifespan(settings, external_on_startup, external_on_shutdown),
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
    setup_routers(app, settings, external_routers, task_access_checker)

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
        }

    return app


def _include_task_routers(
    api_v1: APIRouter,
    settings: Settings,
    task_access_checker: Optional[tasks.TaskAccessChecker],
) -> None:
    """Include the task endpoints with the access ``tasks_api_access_mode`` asks.

    WHY one setting for both groups: the admin endpoints (cancel, active
    tasks, worker stats, task health) are not about a task of the caller's own
    — active tasks lists every task with its arguments, cancel stops anyone's
    task. Opening them to every user would undo the guard on status and
    result, so outside 'public' they always require an active superuser.
    """
    mode = settings.tasks_api_access_mode
    if mode == "public" and task_access_checker is None:
        api_v1.include_router(tasks.router)
        return

    if mode == "public":
        read_user = get_optional_current_user
        read_deps = []
    elif mode == "user":
        read_user = get_current_active_user
        read_deps = [Depends(get_current_active_user)]
    else:
        read_user = get_current_superuser
        read_deps = [Depends(get_current_active_user), Depends(get_current_superuser)]
    if task_access_checker is not None:
        read_deps.append(
            Depends(tasks.task_visibility_dependency(task_access_checker, read_user))
        )

    admin_deps = (
        []
        if mode == "public"
        else [Depends(get_current_active_user), Depends(get_current_superuser)]
    )

    api_v1.include_router(
        tasks.read_router,
        prefix=tasks.TASKS_PREFIX,
        tags=tasks.TASKS_TAGS,
        dependencies=read_deps,
    )
    api_v1.include_router(
        tasks.admin_router,
        prefix=tasks.TASKS_PREFIX,
        tags=tasks.TASKS_TAGS,
        dependencies=admin_deps,
    )


def setup_routers(
    app: FastAPI,
    settings: Settings,
    external_routers: Dict[str, APIRouter] = None,
    task_access_checker: Optional[tasks.TaskAccessChecker] = None,
) -> None:
    """
    Setup all application routers explicitly

    Args:
        app: FastAPI application
        task_access_checker: see ``create_application``
    """
    # Create API v1 router
    api_v1 = APIRouter(prefix="/api/v1")

    # Compute LLM access control dependencies based on provided settings once for this app instance.
    def _llm_deps_for_settings():
        from fastapi import (
            Depends,  # local import to avoid circular deps in type checkers
        )

        mode = settings.llm_api_access_mode
        if mode == "public":
            return []
        elif mode == "user":
            return [Depends(get_current_active_user)]
        else:
            return [Depends(get_current_superuser)]

    # Include all v1 endpoints with explicit configuration
    api_v1.include_router(health.router)
    api_v1.include_router(auth.router)
    _include_task_routers(api_v1, settings, task_access_checker)
    api_v1.include_router(config.router)  # LLM configuration management
    api_v1.include_router(agent_instances.router)  # User agent instance management
    # Apply same access control to batch endpoints (public/user/superuser) derived from settings
    api_v1.include_router(batch.router, dependencies=_llm_deps_for_settings())

    api_v1.include_router(vector.router, dependencies=_llm_deps_for_settings())
    api_v1.include_router(embeddings.router, dependencies=_llm_deps_for_settings())

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
