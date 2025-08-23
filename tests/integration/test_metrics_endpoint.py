"""
Integration tests for metrics endpoint.

Tests that the /metrics endpoint is properly exposed and contains the expected
batch processing metrics families.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.observability.metrics import (
    record_job_status_change,
    record_provider_latency,
    record_job_duration,
    reset_metrics
)


class TestMetricsEndpoint:
    """Test the /metrics endpoint integration."""
    
    def setup_method(self):
        """Setup test environment."""
        # Reset metrics before each test
        reset_metrics()
        self.client = TestClient(app)
    
    def test_metrics_endpoint_available(self):
        """Test that the /metrics endpoint is available and returns 200."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
    
    def test_metrics_endpoint_contains_batch_metrics(self):
        """Test that the /metrics endpoint contains batch processing metrics."""
        # Generate some test metrics
        record_job_status_change("test_provider", "completed")
        record_provider_latency("test_provider", "submit", 1.5)
        record_job_duration("test_provider", "completed", 120.0)
        
        response = self.client.get("/metrics")
        assert response.status_code == 200
        
        content = response.text
        
        # Check for batch metrics families
        assert "batch_jobs_total" in content
        assert "batch_provider_latency_seconds" in content
        assert "batch_job_duration_seconds" in content
        
        # Check for provider labels
        assert 'provider="test_provider"' in content
        
        # Check metric types
        assert "# TYPE batch_jobs_total counter" in content
        assert "# TYPE batch_provider_latency_seconds histogram" in content
        assert "# TYPE batch_job_duration_seconds histogram" in content
    
    def test_metrics_endpoint_prometheus_format(self):
        """Test that the /metrics endpoint returns valid Prometheus format."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        
        content = response.text
        lines = content.strip().split('\n')
        
        # Should have HELP and TYPE lines for metrics
        help_lines = [line for line in lines if line.startswith("# HELP")]
        type_lines = [line for line in lines if line.startswith("# TYPE")]
        
        # Should have at least some metrics documentation
        assert len(help_lines) > 0
        assert len(type_lines) > 0
        
        # Find batch-related help lines
        batch_help_lines = [line for line in help_lines if "batch_" in line]
        assert len(batch_help_lines) > 0
    
    def test_metrics_endpoint_with_no_data(self):
        """Test that the /metrics endpoint works even with no metrics data."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        
        # Should still contain metric definitions even without data
        content = response.text
        assert "batch_jobs_total" in content or "# TYPE" in content
    
    def test_metrics_endpoint_content_type(self):
        """Test that the /metrics endpoint returns correct content type."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        
        # Prometheus expects text/plain with version parameter
        content_type = response.headers["content-type"]
        assert content_type.startswith("text/plain")
        # Prometheus client usually adds version parameter
        assert "version=" in content_type or content_type == "text/plain; charset=utf-8"