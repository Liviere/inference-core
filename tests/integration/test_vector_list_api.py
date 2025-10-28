"""
Integration tests for vector store list API endpoint.
"""

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
async def enable_vector_store_for_tests():
    """Enable in-memory vector store for integration tests"""
    from inference_core.vectorstores.factory import clear_provider_cache, get_vector_store_provider
    from inference_core.services.vector_store_service import get_vector_store_service
    
    # Clear any cached provider
    clear_provider_cache()
    
    # Mock settings to enable vector store
    with patch('inference_core.vectorstores.factory.get_settings') as mock_get_settings:
        mock_settings = type('MockSettings', (), {
            'is_vector_store_enabled': True,
            'vector_backend': 'memory',
            'vector_collection_default': 'test_collection',
            'vector_dim': 384,
            'vector_distance': 'cosine',
            'vector_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'vector_ingest_max_batch_size': 1000,
        })()
        mock_get_settings.return_value = mock_settings
        
        # Force re-initialization of provider
        provider = get_vector_store_provider()
        service = get_vector_store_service()
        service._provider = provider
        
        yield
        
        # Clean up
        clear_provider_cache()


@pytest.mark.asyncio
async def test_list_documents_endpoint_requires_auth(async_test_client):
    """Test that list endpoint requires authentication"""
    response = await async_test_client.post(
        "/api/v1/vector/list",
        json={"limit": 10},
    )
    # Should require authentication
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_list_documents_basic(public_access_async_client):
    """Test basic list documents functionality"""
    # First, ingest some test documents synchronously
    ingest_response = await public_access_async_client.post(
        "/api/v1/vector/ingest",
        json={
            "texts": [
                "First document",
                "Second document",
                "Third document",
            ],
            "metadatas": [
                {"session_id": "s-123", "index": 1},
                {"session_id": "s-123", "index": 2},
                {"session_id": "s-456", "index": 3},
            ],
            "async_mode": False,  # Synchronous processing
        },
    )
    assert ingest_response.status_code == 200
    
    # Now list all documents
    list_response = await public_access_async_client.post(
        "/api/v1/vector/list",
        json={"limit": 10},
    )
    
    assert list_response.status_code == 200
    data = list_response.json()
    
    assert "documents" in data
    assert "count" in data
    assert "total" in data
    assert "collection" in data
    assert "limit" in data
    assert "offset" in data
    
    assert data["count"] == 3
    assert data["total"] == 3
    assert len(data["documents"]) == 3


@pytest.mark.asyncio
async def test_list_documents_with_filters(public_access_async_client):
    """Test listing documents with metadata filters"""
    # Ingest test documents
    await public_access_async_client.post(
        "/api/v1/vector/ingest",
        json={
            "texts": [
                "Doc A1",
                "Doc A2",
                "Doc B1",
                "Doc A3",
            ],
            "metadatas": [
                {"session_id": "session-A", "type": "note"},
                {"session_id": "session-A", "type": "task"},
                {"session_id": "session-B", "type": "note"},
                {"session_id": "session-A", "type": "note"},
            ],
            "async_mode": False,
        },
    )
    
    # List documents for session-A only
    response = await public_access_async_client.post(
        "/api/v1/vector/list",
        json={
            "filters": {"session_id": "session-A"},
            "limit": 10,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["count"] == 3
    assert data["total"] == 3
    
    # Verify all returned documents have session_id = "session-A"
    for doc in data["documents"]:
        assert doc["metadata"]["session_id"] == "session-A"


@pytest.mark.asyncio
async def test_list_documents_pagination(public_access_async_client):
    """Test pagination in list endpoint"""
    # Ingest 5 documents
    await public_access_async_client.post(
        "/api/v1/vector/ingest",
        json={
            "texts": [f"Document {i}" for i in range(1, 6)],
            "metadatas": [{"index": i} for i in range(1, 6)],
            "async_mode": False,
        },
    )
    
    # Get first page (limit=2, offset=0)
    page1 = await public_access_async_client.post(
        "/api/v1/vector/list",
        json={"limit": 2, "offset": 0},
    )
    assert page1.status_code == 200
    data1 = page1.json()
    assert data1["count"] == 2
    assert data1["total"] == 5
    assert data1["limit"] == 2
    assert data1["offset"] == 0
    
    # Get second page (limit=2, offset=2)
    page2 = await public_access_async_client.post(
        "/api/v1/vector/list",
        json={"limit": 2, "offset": 2},
    )
    assert page2.status_code == 200
    data2 = page2.json()
    assert data2["count"] == 2
    assert data2["total"] == 5
    
    # Get third page (limit=2, offset=4) - should have 1 document
    page3 = await public_access_async_client.post(
        "/api/v1/vector/list",
        json={"limit": 2, "offset": 4},
    )
    assert page3.status_code == 200
    data3 = page3.json()
    assert data3["count"] == 1
    assert data3["total"] == 5


@pytest.mark.asyncio
async def test_list_documents_validation(public_access_async_client):
    """Test validation of list parameters"""
    # Invalid limit (too high)
    response = await public_access_async_client.post(
        "/api/v1/vector/list",
        json={"limit": 2000},
    )
    assert response.status_code == 422  # Validation error
    
    # Invalid limit (too low)
    response = await public_access_async_client.post(
        "/api/v1/vector/list",
        json={"limit": 0},
    )
    assert response.status_code == 422
    
    # Invalid offset (negative)
    response = await public_access_async_client.post(
        "/api/v1/vector/list",
        json={"offset": -1},
    )
    assert response.status_code == 422
    
    # Invalid order
    response = await public_access_async_client.post(
        "/api/v1/vector/list",
        json={"order": "invalid"},
    )
    assert response.status_code == 422  # Pydantic validation catches this


@pytest.mark.asyncio
async def test_list_documents_empty_collection(public_access_async_client):
    """Test listing from an empty collection"""
    response = await public_access_async_client.post(
        "/api/v1/vector/list",
        json={"limit": 10},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["count"] == 0
    assert data["documents"] == []
    # Total might be None or 0 depending on provider
    assert data["total"] is not None or data["total"] == 0


@pytest.mark.asyncio
async def test_list_documents_with_ordering(public_access_async_client):
    """Test listing documents with ordering"""
    # Ingest documents with timestamps
    await public_access_async_client.post(
        "/api/v1/vector/ingest",
        json={
            "texts": ["Third", "First", "Second"],
            "metadatas": [
                {"timestamp": "2024-01-03", "index": 3},
                {"timestamp": "2024-01-01", "index": 1},
                {"timestamp": "2024-01-02", "index": 2},
            ],
            "async_mode": False,
        },
    )
    
    # List with ascending order
    response = await public_access_async_client.post(
        "/api/v1/vector/list",
        json={
            "order_by": "index",
            "order": "asc",
            "limit": 10,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["documents"]) == 3
    
    # Verify ordering (should be 1, 2, 3)
    indices = [doc["metadata"]["index"] for doc in data["documents"]]
    assert indices == [1, 2, 3]
    
    # List with descending order
    response = await public_access_async_client.post(
        "/api/v1/vector/list",
        json={
            "order_by": "index",
            "order": "desc",
            "limit": 10,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify ordering (should be 3, 2, 1)
    indices = [doc["metadata"]["index"] for doc in data["documents"]]
    assert indices == [3, 2, 1]
