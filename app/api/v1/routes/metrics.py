"""
Metrics endpoint for Prometheus scraping.

Provides /metrics endpoint that exposes Prometheus metrics for monitoring
batch processing operations and general application health.
"""

import os

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    generate_latest,
)

try:
    from prometheus_client import multiprocess
except Exception:  # pragma: no cover
    multiprocess = None  # type: ignore

router = APIRouter(tags=["Metrics"])


@router.get("/metrics")
async def get_metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping by Prometheus server.

    Returns:
        Response with metrics data in Prometheus format
    """
    multiproc_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    if multiproc_dir and multiprocess:  # multi-process mode
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
