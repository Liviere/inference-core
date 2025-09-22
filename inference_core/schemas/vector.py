"""
Vector Store API Schemas

Pydantic schemas for vector store API requests and responses.
"""

from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    """Request schema for ingesting documents into vector store"""

    texts: List[str] = Field(
        ...,
        description="List of texts to ingest",
        min_length=1,
        max_length=1000,
    )
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional metadata for each text",
    )

    ids: Optional[List[Union[int, UUID]]] = Field(
        default=None,
        description="Opcjonalne identyfikatory dla każdego tekstu (tylko UUID lub int; str powoduje błąd)",
    )
    collection: Optional[str] = Field(
        default=None,
        description="Collection name (uses default if not provided)",
        max_length=100,
    )
    async_mode: bool = Field(
        default=True,
        description="Whether to process ingestion asynchronously",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        """Validate text inputs"""
        if not v:
            raise ValueError("At least one text must be provided")

        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string")
            if len(text.strip()) == 0:
                raise ValueError(f"Text at index {i} cannot be empty")
            if len(text) > 50000:  # Reasonable limit for single document
                raise ValueError(
                    f"Text at index {i} is too long (max 50,000 characters)"
                )

        return v

    @field_validator("metadatas")
    @classmethod
    def validate_metadatas(cls, v, info):
        """Validate metadata inputs"""
        if v is not None:
            texts = info.data.get("texts", [])
            if len(v) != len(texts):
                raise ValueError("Number of metadatas must match number of texts")

            for i, metadata in enumerate(v):
                if not isinstance(metadata, dict):
                    raise ValueError(f"Metadata at index {i} must be a dictionary")

        return v

    @field_validator("ids")
    @classmethod
    def validate_ids(cls, v, info):
        """Walidacja identyfikatorów: tylko UUID lub int"""
        if v is not None:
            texts = info.data.get("texts", [])
            if len(v) != len(texts):
                raise ValueError(
                    "Liczba identyfikatorów musi odpowiadać liczbie tekstów"
                )
            if len(set(v)) != len(v):
                raise ValueError("Wszystkie identyfikatory muszą być unikalne")
            for i, doc_id in enumerate(v):
                if isinstance(doc_id, int):
                    continue
                if isinstance(doc_id, UUID):
                    continue
                raise ValueError(
                    f"ID na pozycji {i} musi być typu int lub UUID (str powoduje błąd)"
                )
        return v


class IngestResponse(BaseModel):
    """Response schema for synchronous ingestion"""

    success: bool = Field(..., description="Whether ingestion was successful")
    document_ids: List[str] = Field(..., description="List of ingested document IDs")
    collection: str = Field(..., description="Collection name used")
    count: int = Field(..., description="Number of documents ingested")
    message: Optional[str] = Field(default=None, description="Additional information")


class IngestTaskResponse(BaseModel):
    """Response schema for asynchronous ingestion"""

    task_id: str = Field(..., description="Celery task ID for tracking")
    message: str = Field(..., description="Status message")
    collection: str = Field(..., description="Collection name used")
    estimated_count: int = Field(..., description="Number of documents to be processed")


class QueryRequest(BaseModel):
    """Request schema for querying vector store"""

    query: str = Field(
        ...,
        description="Query text for similarity search",
        min_length=1,
        max_length=10000,
    )
    k: int = Field(
        default=4,
        description="Number of results to return",
        ge=1,
        le=100,
    )
    collection: Optional[str] = Field(
        default=None,
        description="Collection name (uses default if not provided)",
        max_length=100,
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters to apply to search results",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate query text"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class RetrievedDocument(BaseModel):
    """Schema for a retrieved document"""

    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )
    score: Optional[float] = Field(default=None, description="Similarity score")


class QueryResponse(BaseModel):
    """Response schema for vector store queries"""

    documents: List[RetrievedDocument] = Field(..., description="Retrieved documents")
    query: str = Field(..., description="Original query")
    collection: str = Field(..., description="Collection name used")
    count: int = Field(..., description="Number of documents returned")
    total_in_collection: Optional[int] = Field(
        default=None,
        description="Total number of documents in collection",
    )


class CollectionStatsResponse(BaseModel):
    """Response schema for collection statistics"""

    name: str = Field(..., description="Collection name")
    count: int = Field(..., description="Number of documents")
    dimension: int = Field(..., description="Vector dimension")
    distance_metric: str = Field(..., description="Distance metric used")


class VectorStoreHealthResponse(BaseModel):
    """Response schema for vector store health check"""

    status: str = Field(..., description="Health status (healthy, unhealthy, disabled)")
    backend: Optional[str] = Field(default=None, description="Backend type")
    message: Optional[str] = Field(
        default=None, description="Additional status information"
    )
    collections: Optional[List[str]] = Field(
        default=None, description="Available collections"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional health details"
    )


class ErrorResponse(BaseModel):
    """Standard error response schema"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )
