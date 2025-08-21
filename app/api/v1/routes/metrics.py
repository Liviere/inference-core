"""
Metrics endpoint for Prometheus scraping.

Provides /metrics endpoint that exposes Prometheus metrics for monitoring
batch processing operations and general application health.
"""

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

router = APIRouter(tags=["Metrics"])


@router.get("/metrics")
async def get_metrics() -> Response:
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format for scraping by Prometheus server.
    
    Returns:
        Response with metrics data in Prometheus format
    """
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)