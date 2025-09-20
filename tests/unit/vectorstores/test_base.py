"""
Test Vector Store Base Classes and In-Memory Provider
"""

import pytest
from unittest.mock import AsyncMock

from inference_core.vectorstores.base import (
    BaseVectorStoreProvider,
    InMemoryVectorStoreProvider,
    VectorStoreDocument,
    CollectionStats,
)


class TestVectorStoreDocument:
    """Test VectorStoreDocument model"""

    def test_create_document(self):
        """Test creating a document"""
        doc = VectorStoreDocument(
            id="test-1",
            content="This is a test document",
            metadata={"source": "test"},
            score=0.95,
        )
        
        assert doc.id == "test-1"
        assert doc.content == "This is a test document"
        assert doc.metadata == {"source": "test"}
        assert doc.score == 0.95

    def test_document_defaults(self):
        """Test document with default values"""
        doc = VectorStoreDocument(
            id="test-2",
            content="Another test",
        )
        
        assert doc.metadata == {}
        assert doc.score is None


class TestCollectionStats:
    """Test CollectionStats model"""

    def test_create_stats(self):
        """Test creating collection stats"""
        stats = CollectionStats(
            name="test_collection",
            count=100,
            dimension=384,
            distance_metric="cosine",
        )
        
        assert stats.name == "test_collection"
        assert stats.count == 100
        assert stats.dimension == 384
        assert stats.distance_metric == "cosine"


class TestInMemoryVectorStoreProvider:
    """Test InMemoryVectorStoreProvider"""

    @pytest.fixture
    def provider(self):
        """Create a test provider"""
        config = {
            "default_collection": "test_collection",
            "dimension": 384,
            "distance": "cosine",
        }
        return InMemoryVectorStoreProvider(config)

    def test_provider_initialization(self, provider):
        """Test provider initialization"""
        assert provider.get_default_collection() == "test_collection"
        assert provider.get_dimension() == 384
        assert provider.get_distance_metric() == "cosine"

    @pytest.mark.asyncio
    async def test_ensure_collection(self, provider):
        """Test collection creation"""
        # Create new collection
        created = await provider.ensure_collection("new_collection")
        assert created is True
        assert "new_collection" in provider._collections
        
        # Try to create again
        created = await provider.ensure_collection("new_collection")
        assert created is False

    @pytest.mark.asyncio
    async def test_add_texts(self, provider):
        """Test adding texts to collection"""
        texts = ["Document 1", "Document 2", "Document 3"]
        metadatas = [{"type": "doc"}, {"type": "doc"}, {"type": "article"}]
        ids = ["1", "2", "3"]
        
        doc_ids = await provider.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            collection="test_collection",
        )
        
        assert doc_ids == ["1", "2", "3"]
        assert len(provider._documents["test_collection"]) == 3
        assert provider._collections["test_collection"]["count"] == 3

    @pytest.mark.asyncio
    async def test_add_texts_without_metadata(self, provider):
        """Test adding texts without metadata"""
        texts = ["Document 1", "Document 2"]
        
        doc_ids = await provider.add_texts(
            texts=texts,
            collection="test_collection",
        )
        
        assert len(doc_ids) == 2
        assert len(provider._documents["test_collection"]) == 2
        
        # Check documents have empty metadata
        for doc in provider._documents["test_collection"]:
            assert doc.metadata == {}

    @pytest.mark.asyncio
    async def test_add_texts_validation(self, provider):
        """Test validation errors when adding texts"""
        texts = ["Document 1", "Document 2"]
        metadatas = [{"type": "doc"}]  # Wrong length
        
        with pytest.raises(ValueError, match="Length of metadatas must match"):
            await provider.add_texts(texts=texts, metadatas=metadatas)

    @pytest.mark.asyncio
    async def test_similarity_search(self, provider):
        """Test similarity search"""
        # Add some test documents
        texts = [
            "Python is a programming language",
            "JavaScript is used for web development",
            "Machine learning uses Python",
            "Web apps use JavaScript frameworks",
        ]
        
        await provider.add_texts(texts=texts, collection="test_collection")
        
        # Search for Python-related documents
        results = await provider.similarity_search(
            query="Python programming",
            k=2,
            collection="test_collection",
        )
        
        assert len(results) <= 2
        # Check that results have scores
        for doc in results:
            assert doc.score is not None
            assert doc.score > 0

    @pytest.mark.asyncio
    async def test_similarity_search_empty_collection(self, provider):
        """Test search in empty collection"""
        results = await provider.similarity_search(
            query="test query",
            collection="empty_collection",
        )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_collection_stats(self, provider):
        """Test getting collection statistics"""
        # Add some documents
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        await provider.add_texts(texts=texts, collection="test_collection")
        
        stats = await provider.collection_stats("test_collection")
        
        assert stats.name == "test_collection"
        assert stats.count == 3
        assert stats.dimension == 384
        assert stats.distance_metric == "cosine"

    @pytest.mark.asyncio
    async def test_collection_stats_nonexistent(self, provider):
        """Test stats for non-existent collection"""
        with pytest.raises(ValueError, match="Collection .* does not exist"):
            await provider.collection_stats("nonexistent")

    @pytest.mark.asyncio
    async def test_delete_collection(self, provider):
        """Test deleting a collection"""
        # Create and populate collection
        await provider.ensure_collection("test_collection")
        await provider.add_texts(["Doc 1"], collection="test_collection")
        
        # Delete collection
        deleted = await provider.delete_collection("test_collection")
        assert deleted is True
        assert "test_collection" not in provider._collections
        assert "test_collection" not in provider._documents
        
        # Try to delete again
        deleted = await provider.delete_collection("test_collection")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_health_check(self, provider):
        """Test health check"""
        # Add some collections and documents
        await provider.add_texts(["Doc 1"], collection="col1")
        await provider.add_texts(["Doc 2", "Doc 3"], collection="col2")
        
        health = await provider.health_check()
        
        assert health["status"] == "healthy"
        assert health["backend"] == "memory"
        assert set(health["collections"]) == {"col1", "col2"}
        assert health["total_documents"] == 3

    @pytest.mark.asyncio
    async def test_as_retriever(self, provider):
        """Test creating a retriever"""
        # Add some test documents
        texts = ["Document about cats", "Document about dogs", "Document about birds"]
        await provider.add_texts(texts=texts, collection="test_collection")
        
        # Create retriever
        retriever = await provider.as_retriever(collection="test_collection")
        
        # Test retriever interface
        assert hasattr(retriever, '_get_relevant_documents')
        
        # This would require running the retriever in a sync context
        # which is complex to test here, so we just verify it's created


class TestBaseVectorStoreProvider:
    """Test BaseVectorStoreProvider abstract class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseVectorStoreProvider cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseVectorStoreProvider({})

    def test_config_access_methods(self):
        """Test configuration access methods"""
        config = {
            "default_collection": "my_collection",
            "dimension": 512,
            "distance": "euclidean",
        }
        
        # Create a concrete implementation for testing
        class TestProvider(BaseVectorStoreProvider):
            async def ensure_collection(self, name, dimension=None):
                pass
            async def add_texts(self, texts, metadatas=None, ids=None, collection=None):
                pass
            async def similarity_search(self, query, k=4, collection=None, filters=None, **kwargs):
                pass
            async def as_retriever(self, collection=None, search_kwargs=None):
                pass
            async def collection_stats(self, collection):
                pass
            async def delete_collection(self, collection):
                pass
            async def health_check(self):
                pass
        
        provider = TestProvider(config)
        
        assert provider.get_default_collection() == "my_collection"
        assert provider.get_dimension() == 512
        assert provider.get_distance_metric() == "euclidean"

    def test_config_defaults(self):
        """Test default configuration values"""
        class TestProvider(BaseVectorStoreProvider):
            async def ensure_collection(self, name, dimension=None):
                pass
            async def add_texts(self, texts, metadatas=None, ids=None, collection=None):
                pass
            async def similarity_search(self, query, k=4, collection=None, filters=None, **kwargs):
                pass
            async def as_retriever(self, collection=None, search_kwargs=None):
                pass
            async def collection_stats(self, collection):
                pass
            async def delete_collection(self, collection):
                pass
            async def health_check(self):
                pass
        
        provider = TestProvider({})
        
        assert provider.get_default_collection() == "default_documents"
        assert provider.get_dimension() == 384
        assert provider.get_distance_metric() == "cosine"