"""
Vector Store API Endpoints

FastAPI endpoints for vector store operations including document ingestion,
similarity search, and collection management.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from inference_core.core.dependecies import get_llm_router_dependencies
from inference_core.schemas.vector import (
    CollectionStatsResponse,
    ErrorResponse,
    IngestRequest,
    IngestTaskResponse,
    ListRequest,
    ListResponse,
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    VectorStoreHealthResponse,
)
from inference_core.services.vector_store_service import get_vector_store_service

logger = logging.getLogger(__name__)

# Create router with same access control as LLM endpoints
router = APIRouter(
    prefix="/vector",
    tags=["Vector Store"],
    dependencies=get_llm_router_dependencies(),
    responses={
        503: {"model": ErrorResponse, "description": "Vector store unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


async def get_vector_service():
    """Dependency to get vector store service and check availability"""
    service = get_vector_store_service()
    if not service.is_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not configured or unavailable",
        )
    return service


@router.get("/health", response_model=VectorStoreHealthResponse)
async def get_vector_store_health():
    """
    Get vector store health status.

    Returns health information about the vector store backend,
    including connectivity status and available collections.
    """
    try:
        service = get_vector_store_service()
        health_info = await service.health_check()

        return VectorStoreHealthResponse(
            status=health_info.get("status", "unknown"),
            backend=health_info.get("backend"),
            message=health_info.get("message"),
            collections=health_info.get("collections"),
            details={
                k: v
                for k, v in health_info.items()
                if k not in ("status", "backend", "message", "collections")
            },
        )
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}", exc_info=True)
        return VectorStoreHealthResponse(
            status="error",
            message=f"Health check failed: {str(e)}",
        )


@router.post("/ingest", response_model=IngestTaskResponse)
async def ingest_documents(
    request: IngestRequest,
    service=Depends(get_vector_service),
):
    """
    Ingest documents into vector store.

    Accepts a list of texts with optional metadata and IDs.
    Documents are processed asynchronously by default, returning a task ID for tracking.

    For synchronous processing (not recommended for large batches),
    set async_mode=False in the request.
    """
    try:
        if request.async_mode:
            # Asynchronous processing via Celery
            from inference_core.celery.tasks.vector_tasks import ingest_documents_task

            # Submit task to Celery
            task = ingest_documents_task.delay(
                texts=request.texts,
                metadatas=request.metadatas,
                ids=request.ids,
                collection=request.collection,
            )

            actual_collection = (
                request.collection or service.provider.get_default_collection()
            )

            return IngestTaskResponse(
                task_id=task.id,
                message="Document ingestion task submitted successfully",
                collection=actual_collection,
                estimated_count=len(request.texts),
            )

        else:
            # Synchronous processing (not recommended for large batches)
            if len(request.texts) > 100:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Synchronous processing is limited to 100 documents. Use async_mode=True for larger batches.",
                )

            document_ids = await service.add_texts(
                texts=request.texts,
                metadatas=request.metadatas,
                ids=request.ids,
                collection=request.collection,
            )

            actual_collection = (
                request.collection or service.provider.get_default_collection()
            )

            return IngestTaskResponse(
                task_id="synchronous",
                message="Documents ingested successfully",
                collection=actual_collection,
                estimated_count=len(document_ids),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    service=Depends(get_vector_service),
):
    """
    Query documents using similarity search.

    Finds documents similar to the provided query text and returns them
    ranked by similarity score.
    """
    try:
        # Perform similarity search
        documents = await service.similarity_search(
            query=request.query,
            k=request.k,
            collection=request.collection,
            filters=request.filters,
        )

        # Convert to response format
        retrieved_docs = [
            RetrievedDocument(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                score=doc.score,
            )
            for doc in documents
        ]

        actual_collection = (
            request.collection or service.provider.get_default_collection()
        )

        # Get collection stats for total count (optional)
        total_in_collection = None
        try:
            stats = await service.get_collection_stats(actual_collection)
            total_in_collection = stats.count
        except Exception:
            # Don't fail the query if we can't get stats
            pass

        return QueryResponse(
            documents=retrieved_docs,
            query=request.query,
            collection=actual_collection,
            count=len(retrieved_docs),
            total_in_collection=total_in_collection,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document query failed: {str(e)}",
        )


@router.post("/list", response_model=ListResponse)
async def list_documents(
    request: ListRequest,
    service=Depends(get_vector_service),
):
    """
    List documents by metadata filters without a text query.
    
    Retrieves documents based on metadata filters with pagination support.
    Useful for listing all documents in a session, by user, or other metadata criteria.
    """
    try:
        # Perform filter-only listing
        documents, total_count = await service.list_documents(
            collection=request.collection,
            filters=request.filters,
            limit=request.limit,
            offset=request.offset,
            order_by=request.order_by,
            order=request.order,
            include_scores=False,
        )

        # Convert to response format
        retrieved_docs = [
            RetrievedDocument(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                score=doc.score,
            )
            for doc in documents
        ]

        actual_collection = (
            request.collection or service.provider.get_default_collection()
        )

        return ListResponse(
            documents=retrieved_docs,
            count=len(retrieved_docs),
            total=total_count,
            collection=actual_collection,
            limit=request.limit,
            offset=request.offset,
        )

    except ValueError as e:
        # Validation errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document listing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document listing failed: {str(e)}",
        )


@router.get("/collections/{collection}/stats", response_model=CollectionStatsResponse)
async def get_collection_stats(
    collection: str,
    service=Depends(get_vector_service),
):
    """
    Get statistics for a specific collection.

    Returns information about the collection including document count,
    vector dimension, and distance metric.
    """
    try:
        stats = await service.get_collection_stats(collection)

        return CollectionStatsResponse(
            name=stats.name,
            count=stats.count,
            dimension=stats.dimension,
            distance_metric=stats.distance_metric,
        )

    except ValueError as e:
        # Collection doesn't exist
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection stats: {str(e)}",
        )


@router.delete("/collections/{collection}")
async def delete_collection(
    collection: str,
    service=Depends(get_vector_service),
):
    """
    Delete a collection and all its documents.

    **Warning**: This operation is irreversible and will delete all documents
    in the specified collection.
    """
    try:
        deleted = await service.delete_collection(collection)

        if deleted:
            return {"message": f"Collection '{collection}' deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection}' not found",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete collection: {str(e)}",
        )


# Optional endpoints for development/debugging


@router.post("/collections/{collection}/ensure")
async def ensure_collection(
    collection: str,
    service=Depends(get_vector_service),
):
    """
    Ensure a collection exists, creating it if necessary.

    This endpoint is mainly for development and testing purposes.
    """
    try:
        created = await service.ensure_collection(collection)

        if created:
            return {"message": f"Collection '{collection}' created successfully"}
        else:
            return {"message": f"Collection '{collection}' already exists"}

    except Exception as e:
        logger.error(f"Failed to ensure collection: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ensure collection: {str(e)}",
        )
