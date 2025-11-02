import pytest
from httpx import ASGITransport, AsyncClient

from inference_core.main_factory import create_application


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_text():
    app = create_application()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/metrics")
        assert r.status_code == 200
        # Prometheus text exposition format content type
        assert r.headers.get("content-type", "").startswith("text/plain; version=0.0.4")
        # Basic sanity: response contains HELP/TYPE lines or metrics
        body = r.text
        assert "# HELP" in body or "# TYPE" in body or len(body) > 0
