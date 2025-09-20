"""
Base Vector Store Provider

Abstract interface for vector store implementations.
Defines the contract that all vector store providers must follow.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VectorStoreDocument(BaseModel):
    """Document with vector representation"""
    
    id: str
    content: str
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None


class CollectionStats(BaseModel):
    """Statistics about a vector collection"""
    
    name: str
    count: int
    dimension: int
    distance_metric: str


class BaseVectorStoreProvider(ABC):
    """
    Abstract base class for vector store providers.
    
    Provides a unified interface for different vector database backends
    like Qdrant, Milvus, FAISS, etc.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector store provider.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def ensure_collection(self, name: str, dimension: int = None) -> bool:
        """
        Ensure a collection exists, creating it if necessary.
        
        Args:
            name: Collection name
            dimension: Vector dimension (uses default if None)
            
        Returns:
            True if collection was created, False if it already existed
        """
        pass

    @abstractmethod
    async def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
        collection: Optional[str] = None,
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: Sequence of texts to add
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text (auto-generated if None)
            collection: Collection name (uses default if None)
            
        Returns:
            List of document IDs that were added
        """
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[VectorStoreDocument]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            collection: Collection name (uses default if None)
            filters: Optional filters to apply
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents with scores
        """
        pass

    @abstractmethod
    async def as_retriever(
        self,
        collection: Optional[str] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BaseRetriever:
        """
        Create a LangChain retriever interface.
        
        Args:
            collection: Collection name (uses default if None)
            search_kwargs: Additional search parameters
            
        Returns:
            LangChain BaseRetriever instance
        """
        pass

    @abstractmethod
    async def collection_stats(self, collection: str) -> CollectionStats:
        """
        Get statistics about a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Collection statistics
        """
        pass

    @abstractmethod
    async def delete_collection(self, collection: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            True if collection was deleted, False if it didn't exist
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the vector store.
        
        Returns:
            Health status information
        """
        pass

    def get_default_collection(self) -> str:
        """Get the default collection name from config"""
        return self.config.get("default_collection", "default_documents")

    def get_dimension(self) -> int:
        """Get the vector dimension from config"""
        return self.config.get("dimension", 384)

    def get_distance_metric(self) -> str:
        """Get the distance metric from config"""
        return self.config.get("distance", "cosine")


class InMemoryVectorStoreProvider(BaseVectorStoreProvider):
    """
    In-memory vector store implementation for testing and development.
    
    Note: This is not suitable for production use as data is not persisted.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._collections: Dict[str, Dict[str, Any]] = {}
        self._documents: Dict[str, List[VectorStoreDocument]] = {}

    async def ensure_collection(self, name: str, dimension: int = None) -> bool:
        """Create collection in memory if it doesn't exist"""
        if name not in self._collections:
            self._collections[name] = {
                "dimension": dimension or self.get_dimension(),
                "distance": self.get_distance_metric(),
                "count": 0,
            }
            self._documents[name] = []
            self.logger.info(f"Created in-memory collection: {name}")
            return True
        return False

    async def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
        collection: Optional[str] = None,
    ) -> List[str]:
        """Add texts to in-memory storage"""
        collection = collection or self.get_default_collection()
        await self.ensure_collection(collection)

        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts")
        if ids and len(ids) != len(texts):
            raise ValueError("Length of ids must match length of texts")

        doc_ids = []
        for i, text in enumerate(texts):
            doc_id = ids[i] if ids else f"{collection}_{i}_{len(self._documents[collection])}"
            metadata = metadatas[i] if metadatas else {}
            
            doc = VectorStoreDocument(
                id=doc_id,
                content=text,
                metadata=metadata,
            )
            
            self._documents[collection].append(doc)
            self._collections[collection]["count"] += 1
            doc_ids.append(doc_id)

        self.logger.info(f"Added {len(texts)} documents to collection: {collection}")
        return doc_ids

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[VectorStoreDocument]:
        """Simple text matching for in-memory implementation"""
        collection = collection or self.get_default_collection()
        
        if collection not in self._documents:
            return []

        # Simple text similarity (case-insensitive substring matching)
        # In a real implementation, this would use actual vector similarity
        documents = self._documents[collection]
        query_lower = query.lower()
        
        # Score documents based on query term presence
        scored_docs = []
        for doc in documents:
            score = 0.0
            content_lower = doc.content.lower()
            
            # Simple scoring: count query words found in content
            query_words = query_lower.split()
            for word in query_words:
                if word in content_lower:
                    score += 1.0
            
            if score > 0:
                doc_copy = doc.model_copy()
                doc_copy.score = score / len(query_words) if query_words else 0.0
                scored_docs.append(doc_copy)

        # Sort by score (descending) and return top k
        scored_docs.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored_docs[:k]

    async def as_retriever(
        self,
        collection: Optional[str] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BaseRetriever:
        """Create a mock retriever for in-memory implementation"""
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

        class InMemoryRetriever(BaseRetriever):
            """Simple in-memory retriever implementation"""
            
            def __init__(self, provider: "InMemoryVectorStoreProvider", collection: str, search_kwargs: Dict[str, Any]):
                # Don't call super().__init__() to avoid Pydantic validation issues
                # Set attributes directly
                object.__setattr__(self, 'provider', provider)
                object.__setattr__(self, 'collection', collection)  
                object.__setattr__(self, 'search_kwargs', search_kwargs or {})

            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                import asyncio
                
                # Run the async method in a sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(
                        self.provider.similarity_search(
                            query=query,
                            collection=self.collection,
                            **self.search_kwargs
                        )
                    )
                finally:
                    loop.close()
                
                # Convert to LangChain Documents
                return [
                    Document(page_content=doc.content, metadata=doc.metadata)
                    for doc in results
                ]

        search_kwargs = search_kwargs or {}
        collection = collection or self.get_default_collection()
        return InMemoryRetriever(self, collection, search_kwargs)

    async def collection_stats(self, collection: str) -> CollectionStats:
        """Get stats for in-memory collection"""
        if collection not in self._collections:
            raise ValueError(f"Collection '{collection}' does not exist")
        
        col_info = self._collections[collection]
        return CollectionStats(
            name=collection,
            count=col_info["count"],
            dimension=col_info["dimension"],
            distance_metric=col_info["distance"],
        )

    async def delete_collection(self, collection: str) -> bool:
        """Delete in-memory collection"""
        if collection in self._collections:
            del self._collections[collection]
            del self._documents[collection]
            self.logger.info(f"Deleted in-memory collection: {collection}")
            return True
        return False

    async def health_check(self) -> Dict[str, Any]:
        """Return health info for in-memory store"""
        return {
            "status": "healthy",
            "backend": "memory",
            "collections": list(self._collections.keys()),
            "total_documents": sum(col["count"] for col in self._collections.values()),
        }