"""
Test Vector Store Service
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from inference_core.services.vector_store_service import VectorStoreService, get_vector_store_service
from inference_core.vectorstores.base import VectorStoreDocument, CollectionStats


class TestVectorStoreService:
    """Test VectorStoreService"""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock vector store provider"""
        provider = AsyncMock()
        provider.get_default_collection.return_value = "default_collection"
        provider.config = {"max_batch_size": 1000}
        return provider

    @pytest.fixture
    def service_with_provider(self, mock_provider):
        """Create service with mocked provider"""
        service = VectorStoreService()
        service._provider = mock_provider
        return service

    def test_service_initialization(self):
        """Test service initialization"""
        service = VectorStoreService()
        assert service._provider is None
        assert service._retrievers_cache == {}

    def test_is_available_with_provider(self, service_with_provider):
        """Test is_available when provider exists"""
        assert service_with_provider.is_available is True

    @patch('inference_core.services.vector_store_service.get_vector_store_provider')
    def test_is_available_without_provider(self, mock_get_provider):
        """Test is_available when no provider"""
        mock_get_provider.return_value = None
        service = VectorStoreService()
        assert service.is_available is False

    @pytest.mark.asyncio
    async def test_ensure_collection(self, service_with_provider, mock_provider):
        """Test ensuring a collection exists"""
        mock_provider.ensure_collection.return_value = True
        
        result = await service_with_provider.ensure_collection("test_collection")
        
        assert result is True
        mock_provider.ensure_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_ensure_collection_no_provider(self):
        """Test ensure_collection without provider"""
        service = VectorStoreService()
        
        with pytest.raises(RuntimeError, match="Vector store is not available"):
            await service.ensure_collection("test_collection")

    @pytest.mark.asyncio
    async def test_add_texts(self, service_with_provider, mock_provider):
        """Test adding texts"""
        texts = ["Document 1", "Document 2"]
        metadatas = [{"type": "doc"}, {"type": "doc"}]
        mock_provider.add_texts.return_value = ["id1", "id2"]
        mock_provider.ensure_collection.return_value = False
        
        with patch('inference_core.services.vector_store_service.time.time') as mock_time, \
             patch('inference_core.services.vector_store_service.record_vector_ingestion') as mock_record:
            mock_time.side_effect = [0, 1.5, 1.5]  # Start time, end time, and any additional calls
            
            result = await service_with_provider.add_texts(
                texts=texts,
                metadatas=metadatas,
                collection="test_collection",
            )
        
        assert result == ["id1", "id2"]
        mock_provider.ensure_collection.assert_called_once_with("test_collection")
        mock_provider.add_texts.assert_called_once()
        mock_record.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_texts_validation(self, service_with_provider):
        """Test validation when adding texts"""
        # Empty texts
        with pytest.raises(ValueError, match="No texts provided"):
            await service_with_provider.add_texts(texts=[])
        
        # Batch size too large
        large_texts = ["text"] * 1001
        with pytest.raises(ValueError, match="Batch size .* exceeds maximum"):
            await service_with_provider.add_texts(texts=large_texts)

    @pytest.mark.asyncio
    async def test_add_texts_no_provider(self):
        """Test add_texts without provider"""
        service = VectorStoreService()
        
        with pytest.raises(RuntimeError, match="Vector store is not available"):
            await service.add_texts(texts=["test"])

    @pytest.mark.asyncio
    async def test_similarity_search(self, service_with_provider, mock_provider):
        """Test similarity search"""
        mock_documents = [
            VectorStoreDocument(
                id="1",
                content="Test document 1",
                metadata={"source": "test"},
                score=0.95,
            ),
            VectorStoreDocument(
                id="2",
                content="Test document 2",
                metadata={"source": "test"},
                score=0.85,
            ),
        ]
        mock_provider.similarity_search.return_value = mock_documents
        
        with patch('inference_core.services.vector_store_service.time.time') as mock_time, \
             patch('inference_core.services.vector_store_service.record_vector_search') as mock_record:
            mock_time.side_effect = [0, 0.5, 0.5]  # Start time, end time, and any additional calls
            
            result = await service_with_provider.similarity_search(
                query="test query",
                k=2,
                collection="test_collection",
            )
        
        assert len(result) == 2
        assert result[0].id == "1"
        assert result[1].id == "2"
        mock_provider.similarity_search.assert_called_once()
        mock_record.assert_called_once()

    @pytest.mark.asyncio
    async def test_similarity_search_validation(self, service_with_provider):
        """Test validation in similarity search"""
        # Empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await service_with_provider.similarity_search(query="")
        
        # Invalid k
        with pytest.raises(ValueError, match="k must be positive"):
            await service_with_provider.similarity_search(query="test", k=0)

    @pytest.mark.asyncio
    async def test_similarity_search_no_provider(self):
        """Test similarity_search without provider"""
        service = VectorStoreService()
        
        with pytest.raises(RuntimeError, match="Vector store is not available"):
            await service.similarity_search(query="test")

    @pytest.mark.asyncio
    async def test_get_retriever(self, service_with_provider, mock_provider):
        """Test getting a retriever"""
        mock_retriever = MagicMock()
        mock_provider.as_retriever.return_value = mock_retriever
        
        result = await service_with_provider.get_retriever(collection="test_collection")
        
        assert result == mock_retriever
        # Should be cached
        result2 = await service_with_provider.get_retriever(collection="test_collection")
        assert result2 == mock_retriever
        # Provider should only be called once due to caching
        mock_provider.as_retriever.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, service_with_provider, mock_provider):
        """Test getting collection stats"""
        mock_stats = CollectionStats(
            name="test_collection",
            count=100,
            dimension=384,
            distance_metric="cosine",
        )
        mock_provider.collection_stats.return_value = mock_stats
        
        result = await service_with_provider.get_collection_stats("test_collection")
        
        assert result == mock_stats
        mock_provider.collection_stats.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_delete_collection(self, service_with_provider, mock_provider):
        """Test deleting a collection"""
        mock_provider.delete_collection.return_value = True
        
        result = await service_with_provider.delete_collection("test_collection")
        
        assert result is True
        mock_provider.delete_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_health_check_with_provider(self, service_with_provider, mock_provider):
        """Test health check with provider"""
        health_info = {"status": "healthy", "backend": "test"}
        mock_provider.health_check.return_value = health_info
        
        result = await service_with_provider.health_check()
        
        assert result == health_info
        mock_provider.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_no_provider(self):
        """Test health check without provider"""
        service = VectorStoreService()
        
        result = await service.health_check()
        
        assert result["status"] == "disabled"
        assert result["backend"] is None

    def test_sanitize_metadata(self, service_with_provider):
        """Test metadata sanitization"""
        metadata = {
            "valid_string": "test",
            "valid_int": 123,
            "valid_float": 12.34,
            "valid_bool": True,
            "valid_list": ["a", "b", "c"],
            "valid_dict": {"nested": "value"},
            "_private": "should_be_removed",
            "too_long_key" * 20: "value",
            "long_value": "x" * 2000,  # Too long
            "invalid_type": {"complex": {"nested": "dict"}},
        }
        
        sanitized = service_with_provider._sanitize_metadata(metadata)
        
        assert "valid_string" in sanitized
        assert "valid_int" in sanitized
        assert "valid_float" in sanitized
        assert "valid_bool" in sanitized
        assert "valid_list" in sanitized
        assert "valid_dict" in sanitized
        assert "_private" not in sanitized
        assert len([k for k in sanitized.keys() if "too_long_key" in k]) == 0
        assert len(sanitized["long_value"]) == 1000  # Truncated

    def test_clear_cache(self, service_with_provider):
        """Test clearing cache"""
        # Add something to cache
        service_with_provider._retrievers_cache["test"] = MagicMock()
        
        service_with_provider.clear_cache()
        
        assert service_with_provider._retrievers_cache == {}

    @pytest.mark.asyncio
    async def test_list_documents(self, service_with_provider, mock_provider):
        """Test listing documents with filters"""
        mock_documents = [
            VectorStoreDocument(
                id="1",
                content="Document 1",
                metadata={"session_id": "s-123", "created_at": "2024-01-01"},
                score=None,
            ),
            VectorStoreDocument(
                id="2",
                content="Document 2",
                metadata={"session_id": "s-123", "created_at": "2024-01-02"},
                score=None,
            ),
        ]
        mock_provider.list_documents.return_value = (mock_documents, 2)
        
        with patch('inference_core.services.vector_store_service.time.time') as mock_time, \
             patch('inference_core.services.vector_store_service.record_vector_search') as mock_record:
            mock_time.side_effect = [0, 0.5, 0.5]
            
            documents, total = await service_with_provider.list_documents(
                collection="test_collection",
                filters={"session_id": "s-123"},
                limit=10,
                offset=0,
            )
        
        assert len(documents) == 2
        assert total == 2
        assert documents[0].id == "1"
        assert documents[1].id == "2"
        mock_provider.list_documents.assert_called_once()
        mock_record.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_documents_validation(self, service_with_provider):
        """Test validation in list_documents"""
        # Invalid limit (too low)
        with pytest.raises(ValueError, match="limit must be between 1 and 1000"):
            await service_with_provider.list_documents(limit=0)
        
        # Invalid limit (too high)
        with pytest.raises(ValueError, match="limit must be between 1 and 1000"):
            await service_with_provider.list_documents(limit=1001)
        
        # Invalid offset
        with pytest.raises(ValueError, match="offset must be >= 0"):
            await service_with_provider.list_documents(offset=-1)
        
        # Invalid order
        with pytest.raises(ValueError, match="order must be 'asc' or 'desc'"):
            await service_with_provider.list_documents(order="invalid")

    @pytest.mark.asyncio
    async def test_list_documents_no_provider(self):
        """Test list_documents without provider"""
        service = VectorStoreService()
        
        with pytest.raises(RuntimeError, match="Vector store is not available"):
            await service.list_documents()

    @pytest.mark.asyncio
    async def test_list_documents_pagination(self, service_with_provider, mock_provider):
        """Test listing documents with pagination"""
        mock_documents = [
            VectorStoreDocument(
                id=f"{i}",
                content=f"Document {i}",
                metadata={"index": i},
                score=None,
            )
            for i in range(5)
        ]
        mock_provider.list_documents.return_value = (mock_documents, 100)
        
        with patch('inference_core.services.vector_store_service.time.time') as mock_time, \
             patch('inference_core.services.vector_store_service.record_vector_search'):
            mock_time.side_effect = [0, 0.3, 0.3]
            
            documents, total = await service_with_provider.list_documents(
                collection="test_collection",
                limit=5,
                offset=10,
            )
        
        assert len(documents) == 5
        assert total == 100
        # Check that list_documents was called with correct arguments
        assert mock_provider.list_documents.call_count == 1
        call_kwargs = mock_provider.list_documents.call_args[1]
        assert call_kwargs["collection"] == "test_collection"
        assert call_kwargs["limit"] == 5
        assert call_kwargs["offset"] == 10
        assert call_kwargs["order_by"] is None
        assert call_kwargs["order"] == "desc"
        assert call_kwargs["include_scores"] is False

    @pytest.mark.asyncio
    async def test_list_documents_with_ordering(self, service_with_provider, mock_provider):
        """Test listing documents with ordering"""
        mock_documents = [
            VectorStoreDocument(
                id="1",
                content="Document 1",
                metadata={"created_at": "2024-01-02"},
                score=None,
            ),
            VectorStoreDocument(
                id="2",
                content="Document 2",
                metadata={"created_at": "2024-01-01"},
                score=None,
            ),
        ]
        mock_provider.list_documents.return_value = (mock_documents, 2)
        
        with patch('inference_core.services.vector_store_service.time.time') as mock_time, \
             patch('inference_core.services.vector_store_service.record_vector_search'):
            mock_time.side_effect = [0, 0.4, 0.4]
            
            documents, total = await service_with_provider.list_documents(
                order_by="created_at",
                order="asc",
            )
        
        assert len(documents) == 2
        mock_provider.list_documents.assert_called_once()
        call_kwargs = mock_provider.list_documents.call_args[1]
        assert call_kwargs["order_by"] == "created_at"
        assert call_kwargs["order"] == "asc"


class TestGetVectorStoreService:
    """Test get_vector_store_service function"""

    def test_singleton_behavior(self):
        """Test that get_vector_store_service returns the same instance"""
        service1 = get_vector_store_service()
        service2 = get_vector_store_service()
        
        assert service1 is service2

    def test_returns_vector_store_service(self):
        """Test that function returns VectorStoreService instance"""
        service = get_vector_store_service()
        assert isinstance(service, VectorStoreService)