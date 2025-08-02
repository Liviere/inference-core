import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI

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


app = FastAPI(lifespan=lifespan)

###################################
#            Routes               #
###################################


@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "message": f"Welcome to FastAPI!   🚀",
    }
