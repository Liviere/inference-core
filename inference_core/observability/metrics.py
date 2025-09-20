"""Prometheus metrics collection for batch processing.

Supports both single-process (default) and multi-process collection.

Multi-process mode (Celery worker + FastAPI app) is enabled by setting the
environment variable ``PROMETHEUS_MULTIPROC_DIR`` in both containers and mounting
the same directory. The FastAPI metrics endpoint will automatically switch to a
``MultiProcessCollector`` when it detects the variable.

IMPORTANT: Gauges require an aggregation mode in multi-process setups. We use
``livesum`` so values are summed across processes.

Cleanup: Only the API container should set ``PROMETHEUS_MULTIPROC_CLEANUP=1`` so
that on cold start it clears stale metric shard files once. Celery workers must
NOT perform cleanup, otherwise they could delete shards while another process is
writing.  (Source: Prometheus Python client multiprocess docs – 2025-08)
"""

import glob
import os
import time
from functools import wraps
from typing import Any, Dict, Optional

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

# ----------------------------------------------------------------------------
# Multiprocess cleanup (only if explicitly enabled)
# ----------------------------------------------------------------------------
_multiproc_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR")
if os.getenv("PROMETHEUS_MULTIPROC_CLEANUP") == "1" and _multiproc_dir:
    try:
        for f in glob.glob(os.path.join(_multiproc_dir, "*")):
            try:
                os.remove(f)
            except FileNotFoundError:  # race-safe
                pass
    except Exception as _e:  # non-fatal
        # We keep this silent to avoid noisy logs on startup; logging initialized later.
        pass

# Batch job metrics
batch_jobs_total = Counter(
    "batch_jobs_total", "Total number of batch jobs by status", ["provider", "status"]
)

batch_items_total = Counter(
    "batch_items_total", "Total number of batch items by status", ["provider", "status"]
)

batch_job_duration_seconds = Histogram(
    "batch_job_duration_seconds",
    "Time taken to complete batch jobs",
    ["provider", "status"],
    buckets=[1, 5, 15, 30, 60, 300, 900, 1800, 3600, 7200, 14400],  # 1s to 4h
)

batch_poll_cycle_seconds = Histogram(
    "batch_poll_cycle_seconds",
    "Time taken for each batch polling cycle",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120],  # 100ms to 2 minutes
)

batch_provider_latency_seconds = Histogram(
    "batch_provider_latency_seconds",
    "Latency of provider API calls",
    ["provider", "operation"],
    buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],  # 100ms to 30s
)

# Additional operational metrics
batch_jobs_in_progress = Gauge(
    "batch_jobs_in_progress",
    "Number of batch jobs currently in progress",
    ["provider"],
    multiprocess_mode="livesum",
)

batch_retry_attempts_total = Counter(
    "batch_retry_attempts_total",
    "Total number of retry attempts",
    ["provider", "task_type", "retry_reason"],
)

batch_errors_total = Counter(
    "batch_errors_total",
    "Total number of batch processing errors",
    ["provider", "error_type", "operation"],
)


def record_job_status_change(provider: str, new_status: str) -> None:
    """Record a batch job status change in metrics.

    Args:
        provider: LLM provider name (openai, anthropic, etc.)
        new_status: New job status
    """
    batch_jobs_total.labels(provider=provider, status=new_status).inc()


def record_item_status_change(
    provider: str, old_status: Optional[str], new_status: str, count: int = 1
) -> None:
    """Record batch item status changes in metrics.

    Args:
        provider: LLM provider name
        old_status: Previous item status (None for new items)
        new_status: New item status
        count: Number of items changing status (for bulk updates)
    """
    batch_items_total.labels(provider=provider, status=new_status).inc(count)


def record_job_duration(provider: str, status: str, duration_seconds: float) -> None:
    """Record the duration of a completed batch job.

    Args:
        provider: LLM provider name
        status: Final job status (completed, failed, cancelled)
        duration_seconds: Total time from submission to completion
    """
    batch_job_duration_seconds.labels(provider=provider, status=status).observe(
        duration_seconds
    )


def record_provider_latency(
    provider: str, operation: str, latency_seconds: float
) -> None:
    """Record latency for provider API operations.

    Args:
        provider: LLM provider name
        operation: Type of operation (submit, poll, fetch, cancel)
        latency_seconds: Time taken for the API call
    """
    batch_provider_latency_seconds.labels(
        provider=provider, operation=operation
    ).observe(latency_seconds)


def record_poll_cycle_duration(duration_seconds: float) -> None:
    """Record the duration of a batch polling cycle.

    Args:
        duration_seconds: Total time taken for the polling cycle
    """
    batch_poll_cycle_seconds.observe(duration_seconds)


def update_jobs_in_progress(provider: str, delta: int) -> None:
    """Update the gauge for jobs currently in progress.

    Args:
        provider: LLM provider name
        delta: Change in number of in-progress jobs (+1 for started, -1 for completed)
    """
    if delta > 0:
        batch_jobs_in_progress.labels(provider=provider).inc(delta)
    elif delta < 0:
        batch_jobs_in_progress.labels(provider=provider).dec(abs(delta))


def record_retry_attempt(provider: str, task_type: str, retry_reason: str) -> None:
    """Record a retry attempt.

    Args:
        provider: LLM provider name
        task_type: Type of task being retried (submit, poll, fetch)
        retry_reason: Reason for retry (transient_error, timeout, rate_limit)
    """
    batch_retry_attempts_total.labels(
        provider=provider, task_type=task_type, retry_reason=retry_reason
    ).inc()


def record_error(provider: str, error_type: str, operation: str) -> None:
    """Record a batch processing error.

    Args:
        provider: LLM provider name
        error_type: Type of error (permanent, transient, validation)
        operation: Operation that failed (submit, poll, fetch)
    """
    batch_errors_total.labels(
        provider=provider, error_type=error_type, operation=operation
    ).inc()


def time_provider_operation(provider: str, operation: str):
    """Decorator to automatically time and record provider operation latency.

    Args:
        provider: LLM provider name
        operation: Type of operation being performed

    Example:
        @time_provider_operation("openai", "submit")
        def submit_batch():
            # API call here
            pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                record_provider_latency(provider, operation, duration)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                record_provider_latency(provider, operation, duration)

        # Return the appropriate wrapper based on whether the function is a coroutine
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator


def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of current metrics for debugging/monitoring.

    Returns:
        Dictionary containing current metric values
    """
    summary = {}

    # Collect samples from the registry
    for metric_family in REGISTRY.collect():
        metric_name = metric_family.name
        if metric_name.startswith("batch_"):
            summary[metric_name] = {
                "type": metric_family.type,
                "help": metric_family.documentation,
                "samples": [],
            }
            for sample in metric_family.samples:
                summary[metric_name]["samples"].append(
                    {
                        "name": sample.name,
                        "labels": sample.labels,
                        "value": sample.value,
                    }
                )

    return summary


def reset_metrics() -> None:
    """Reset all batch-related metrics.

    WARNING: This should only be used in testing environments.
    """
    # Get all collector instances - only reset counters which support clear()
    collectors_to_reset = [
        batch_jobs_total,
        batch_items_total,
        batch_retry_attempts_total,
        batch_errors_total,
    ]

    for collector in collectors_to_reset:
        try:
            collector.clear()
        except Exception:
            # Some metric types don't support clear() - that's OK for testing
            pass

    # Reset gauges to 0 - they don't have clear() method
    try:
        # This is a bit hacky but necessary for testing
        for metric_family in REGISTRY.collect():
            if metric_family.name == "batch_jobs_in_progress":
                for sample in metric_family.samples:
                    if hasattr(batch_jobs_in_progress, "_value"):
                        # Reset the underlying metric values
                        batch_jobs_in_progress.labels(**sample.labels).set(0)
    except Exception:
        # If reset fails, it's not critical for normal operation
        pass


# ----------------------------------------------------------------------------
# Vector Store Metrics
# ----------------------------------------------------------------------------

# Vector store operation metrics
vector_similarity_search_seconds = Histogram(
    "vector_similarity_search_seconds",
    "Time spent on vector similarity search",
    ["backend", "collection"],
)

vector_ingest_batch_seconds = Histogram(
    "vector_ingest_batch_seconds",
    "Time spent on batch document ingestion",
    ["backend", "collection"],
)

vector_documents_ingested_total = Counter(
    "vector_documents_ingested_total",
    "Total number of documents ingested",
    ["backend", "collection"],
)

vector_query_requests_total = Counter(
    "vector_query_requests_total",
    "Total number of vector query requests",
    ["backend", "collection", "status"],
)

vector_collections_total = Gauge(
    "vector_collections_total",
    "Total number of vector collections",
    ["backend"],
    multiprocess_mode="livesum",
)

vector_documents_total = Gauge(
    "vector_documents_total",
    "Total number of documents in all collections",
    ["backend", "collection"],
    multiprocess_mode="livesum",
)


def record_vector_search(backend: str, collection: str, duration_seconds: float, success: bool = True) -> None:
    """Record vector similarity search metrics.
    
    Args:
        backend: Vector store backend (qdrant, memory, etc.)
        collection: Collection name
        duration_seconds: Time taken for the search
        success: Whether the search was successful
    """
    vector_similarity_search_seconds.labels(backend=backend, collection=collection).observe(duration_seconds)
    status = "success" if success else "error"
    vector_query_requests_total.labels(backend=backend, collection=collection, status=status).inc()


def record_vector_ingestion(backend: str, collection: str, document_count: int, duration_seconds: float) -> None:
    """Record vector document ingestion metrics.
    
    Args:
        backend: Vector store backend (qdrant, memory, etc.)
        collection: Collection name
        document_count: Number of documents ingested
        duration_seconds: Time taken for ingestion
    """
    vector_ingest_batch_seconds.labels(backend=backend, collection=collection).observe(duration_seconds)
    vector_documents_ingested_total.labels(backend=backend, collection=collection).inc(document_count)


def update_vector_collection_stats(backend: str, collection: str, document_count: int) -> None:
    """Update vector collection statistics.
    
    Args:
        backend: Vector store backend (qdrant, memory, etc.)
        collection: Collection name
        document_count: Total number of documents in the collection
    """
    vector_documents_total.labels(backend=backend, collection=collection).set(document_count)


def time_vector_operation(backend: str, collection: str, operation: str):
    """Decorator to automatically time vector store operations.
    
    Args:
        backend: Vector store backend name
        collection: Collection name
        operation: Type of operation (search, ingest)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if operation == "search":
                    record_vector_search(backend, collection, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                if operation == "search":
                    record_vector_search(backend, collection, duration, success=False)
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                if operation == "search":
                    record_vector_search(backend, collection, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                if operation == "search":
                    record_vector_search(backend, collection, duration, success=False)
                raise

        # Return the appropriate wrapper based on whether the function is a coroutine
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator
