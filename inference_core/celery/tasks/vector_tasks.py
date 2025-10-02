"""
Vector Store Celery Tasks

Asynchronous tasks for vector store operations.
"""

import logging
from typing import Any, Dict, List, Optional

from celery import current_task

from inference_core.celery.async_utils import run_in_worker_loop
from inference_core.celery.celery_main import celery_app
from inference_core.services.vector_store_service import get_vector_store_service

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="vector.ingest_documents")
def ingest_documents_task(
    self,
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Asynchronous task for ingesting documents into vector store.

    Args:
        texts: List of texts to ingest
        metadatas: Optional metadata for each text
        ids: Optional IDs for each text
        collection: Optional collection name

    Returns:
        Dict with ingestion results
    """
    task_id = self.request.id
    logger.info(f"Starting vector ingestion task {task_id} with {len(texts)} documents")

    try:
        current_task.update_state(
            state="PROCESSING",
            meta={
                "current": 0,
                "total": len(texts),
                "status": "Starting ingestion...",
                "collection": collection,
            },
        )

        service = get_vector_store_service()
        if not service.is_available:
            raise RuntimeError("Vector store is not available")

        current_task.update_state(
            state="PROCESSING",
            meta={
                "current": 0,
                "total": len(texts),
                "status": "Generating embeddings and storing documents...",
                "collection": collection,
            },
        )

        async def _ingest():
            # Process documents
            document_ids_local = await service.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                collection=collection,
            )

            actual_collection_local = (
                collection or service.provider.get_default_collection()
            )
            stats_local = await service.get_collection_stats(actual_collection_local)
            return document_ids_local, actual_collection_local, stats_local

        document_ids, actual_collection, stats = run_in_worker_loop(_ingest())

        result = {
            "success": True,
            "document_ids": document_ids,
            "collection": actual_collection,
            "count": len(document_ids),
            "total_in_collection": stats.count,
            "dimension": stats.dimension,
            "distance_metric": stats.distance_metric,
            "message": f"Successfully ingested {len(document_ids)} documents",
        }

        logger.info(
            f"Completed vector ingestion task {task_id}: {len(document_ids)} documents"
        )
        return result

    except Exception as e:
        logger.error(f"Vector ingestion task {task_id} failed: {e}", exc_info=True)
        current_task.update_state(
            state="FAILURE",
            meta={
                "current": 0,
                "total": len(texts),
                "status": f"Ingestion failed: {str(e)}",
                "collection": collection,
                "error": str(e),
            },
        )
        raise


@celery_app.task(bind=True, name="vector.health_check")
def health_check_task(self) -> Dict[str, Any]:
    """
    Asynchronous health check for vector store.

    Returns:
        Dict with health status
    """
    task_id = self.request.id
    logger.debug(f"Starting vector store health check task {task_id}")

    try:
        service = get_vector_store_service()

        async def _hc():
            return await service.health_check()

        health_info = run_in_worker_loop(_hc())
        logger.debug(f"Completed vector store health check task {task_id}")
        return health_info

    except Exception as e:
        logger.error(
            f"Vector store health check task {task_id} failed: {e}", exc_info=True
        )
        return {"status": "unhealthy", "error": str(e), "backend": None}


@celery_app.task(bind=True, name="vector.cleanup_collection")
def cleanup_collection_task(self, collection: str) -> Dict[str, Any]:
    """
    Asynchronous task for cleaning up a collection.

    Args:
        collection: Collection name to delete

    Returns:
        Dict with cleanup results
    """
    task_id = self.request.id
    logger.info(
        f"Starting vector collection cleanup task {task_id} for collection: {collection}"
    )

    try:
        service = get_vector_store_service()
        if not service.is_available:
            raise RuntimeError("Vector store is not available")

        async def _cleanup():
            return await service.delete_collection(collection)

        deleted = run_in_worker_loop(_cleanup())
        result = {
            "success": True,
            "collection": collection,
            "deleted": deleted,
            "message": f"Collection '{collection}' {'deleted' if deleted else 'did not exist'}",
        }

        logger.info(
            f"Completed vector collection cleanup task {task_id}: {result['message']}"
        )
        return result

    except Exception as e:
        logger.error(
            f"Vector collection cleanup task {task_id} failed: {e}", exc_info=True
        )
        raise
