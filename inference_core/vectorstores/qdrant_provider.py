"""
Qdrant Vector Store Provider

Implementation of vector store provider using Qdrant as the backend.
Supports both local Qdrant instances and Qdrant Cloud.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
from langchain_core.retrievers import BaseRetriever
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sentence_transformers import SentenceTransformer

from .base import BaseVectorStoreProvider, CollectionStats, VectorStoreDocument

logger = logging.getLogger(__name__)


class QdrantProvider(BaseVectorStoreProvider):
    """
    Qdrant implementation of vector store provider.

    Supports both local and cloud Qdrant instances with async operations.
    Uses sentence-transformers for text embedding generation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Extract Qdrant-specific configuration
        self.url = config.get("url", "http://localhost:6333")
        self.api_key = config.get("api_key")
        self.embedding_model_name = config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize clients
        self._async_client = None
        self._sync_client = None
        self._embedding_model = None

        self.logger.info(f"Initialized Qdrant provider with URL: {self.url}")

    async def _get_async_client(self) -> AsyncQdrantClient:
        """Get or create async Qdrant client"""
        if self._async_client is None:
            client_kwargs = {"url": self.url}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key

            self._async_client = AsyncQdrantClient(**client_kwargs)
            self.logger.debug("Created async Qdrant client")

        return self._async_client

    def _get_sync_client(self) -> QdrantClient:
        """Get or create sync Qdrant client (for embedding operations)"""
        if self._sync_client is None:
            client_kwargs = {"url": self.url}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key

            self._sync_client = QdrantClient(**client_kwargs)
            self.logger.debug("Created sync Qdrant client")

        return self._sync_client

    def _get_embedding_model(self) -> SentenceTransformer:
        """Get or create embedding model"""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")

        return self._embedding_model

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        model = self._get_embedding_model()
        embeddings = model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    def _get_distance_metric(self) -> models.Distance:
        """Convert distance string to Qdrant Distance enum"""
        distance_map = {
            "cosine": models.Distance.COSINE,
            "euclidean": models.Distance.EUCLID,
            "dot": models.Distance.DOT,
        }

        distance_str = self.get_distance_metric()
        if distance_str not in distance_map:
            raise ValueError(f"Unsupported distance metric: {distance_str}")

        return distance_map[distance_str]

    async def ensure_collection(self, name: str, dimension: int = None) -> bool:
        """Ensure collection exists in Qdrant"""
        client = await self._get_async_client()

        try:
            # Check if collection exists
            collection_info = await client.get_collection(name)
            self.logger.debug(f"Collection '{name}' already exists")
            return False
        except (ResponseHandlingException, UnexpectedResponse):
            # Collection doesn't exist, create it
            dimension = dimension or self.get_dimension()

            # Get actual dimension from embedding model
            if dimension == self.get_dimension():
                model = self._get_embedding_model()
                # Get dimension by encoding a test text
                test_embedding = model.encode(["test"], convert_to_tensor=False)
                dimension = len(test_embedding[0])
                self.logger.info(f"Auto-detected embedding dimension: {dimension}")

            vectors_config = models.VectorParams(
                size=dimension,
                distance=self._get_distance_metric(),
            )

            await client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
            )

            self.logger.info(
                f"Created Qdrant collection '{name}' with dimension {dimension}"
            )
            return True

    async def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
        collection: Optional[str] = None,
    ) -> List[str]:
        """Add texts to Qdrant collection"""
        collection = collection or self.get_default_collection()
        client = await self._get_async_client()

        # Ensure collection exists
        await self.ensure_collection(collection)

        # Validate inputs
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts")
        if ids and len(ids) != len(texts):
            raise ValueError("Length of ids must match length of texts")

        # Generate embeddings
        embeddings = self._embed_texts(texts)

        # Generate IDs if not provided
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Prepare points for upsert
        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            # Store the original text in metadata for retrieval
            payload = {**metadata, "_text": text}

            point = models.PointStruct(
                id=ids[i],
                vector=embedding,
                payload=payload,
            )
            points.append(point)

        # Upsert points to Qdrant
        await client.upsert(
            collection_name=collection,
            points=points,
            wait=True,  # Wait for operation to complete
        )

        self.logger.info(
            f"Added {len(texts)} documents to Qdrant collection '{collection}'"
        )
        return list(ids)

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[VectorStoreDocument]:
        """Search for similar documents in Qdrant"""
        collection = collection or self.get_default_collection()
        client = await self._get_async_client()

        # Generate query embedding
        query_embedding = self._embed_texts([query])[0]

        # Prepare search filter if provided
        search_filter = None
        if filters:
            # Convert simple dict filters to Qdrant filter format
            # For now, support simple key-value matching
            conditions = []
            for key, value in filters.items():
                if key != "_text":  # Skip internal text field
                    conditions.append(
                        models.FieldCondition(
                            key=key, match=models.MatchValue(value=value)
                        )
                    )

            if conditions:
                search_filter = models.Filter(must=conditions)

        # Perform search
        search_result = await client.search(
            collection_name=collection,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=k,
            with_payload=True,
        )

        # Convert results to VectorStoreDocument
        documents = []
        for point in search_result:
            # Extract text from payload
            content = point.payload.get("_text", "")
            # Create metadata without internal fields
            metadata = {k: v for k, v in point.payload.items() if not k.startswith("_")}

            doc = VectorStoreDocument(
                id=str(point.id),
                content=content,
                metadata=metadata,
                score=point.score,
            )
            documents.append(doc)

        self.logger.debug(
            f"Found {len(documents)} similar documents for query in collection '{collection}'"
        )
        return documents

    async def as_retriever(
        self,
        collection: Optional[str] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BaseRetriever:
        """Create a LangChain retriever using langchain-qdrant"""
        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError:
            raise ImportError(
                "langchain-qdrant is required for retriever functionality. "
                "Install it with: pip install langchain-qdrant"
            )

        collection = collection or self.get_default_collection()
        search_kwargs = search_kwargs or {}

        # Ensure collection exists
        await self.ensure_collection(collection)

        # Create QdrantVectorStore instance
        sync_client = self._get_sync_client()
        embedding_model = self._get_embedding_model()

        # Create a simple embeddings wrapper for langchain
        class SentenceTransformerEmbeddings:
            def __init__(self, model: SentenceTransformer):
                self.model = model

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self.model.encode(texts, convert_to_tensor=False).tolist()

            def embed_query(self, text: str) -> List[float]:
                return self.model.encode([text], convert_to_tensor=False)[0].tolist()

        embeddings = SentenceTransformerEmbeddings(embedding_model)

        # Create QdrantVectorStore
        vector_store = QdrantVectorStore(
            client=sync_client,
            collection_name=collection,
            embeddings=embeddings,
        )

        # Return as retriever
        return vector_store.as_retriever(search_kwargs=search_kwargs)

    async def collection_stats(self, collection: str) -> CollectionStats:
        """Get collection statistics from Qdrant"""
        client = await self._get_async_client()

        try:
            # Get collection info
            collection_info = await client.get_collection(collection)

            # Count points in collection
            count_result = await client.count(collection_name=collection, exact=True)
            count = count_result.count if hasattr(count_result, "count") else 0

            # Extract collection parameters
            vectors_config = collection_info.config.params.vectors

            if not vectors_config:
                raise ValueError(
                    f"No vector configuration found for collection '{collection}'"
                )

            dimension = vectors_config.size
            distance_metric = vectors_config.distance.value

            return CollectionStats(
                name=collection,
                count=count,
                dimension=dimension,
                distance_metric=distance_metric,
            )

        except (ResponseHandlingException, UnexpectedResponse) as e:
            raise ValueError(
                f"Collection '{collection}' does not exist or is not accessible"
            ) from e

    async def delete_collection(self, collection: str) -> bool:
        """Delete collection from Qdrant"""
        client = await self._get_async_client()

        try:
            await client.delete_collection(collection_name=collection)
            self.logger.info(f"Deleted Qdrant collection: {collection}")
            return True
        except (ResponseHandlingException, UnexpectedResponse):
            self.logger.warning(
                f"Collection '{collection}' does not exist or could not be deleted"
            )
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health and connectivity"""
        try:
            client = await self._get_async_client()

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.url}/readyz") as response:
                    if response.status != 200:
                        raise ConnectionError(
                            f"Qdrant readiness check failed with status {response.status}"
                        )
                    ready_state = await response.text()

            # Get collections list
            collections = await client.get_collections()
            collection_names = [col.name for col in collections.collections]

            return {
                "status": "healthy",
                "backend": "qdrant",
                "url": self.url,
                "ready": ready_state,
                "collections": collection_names,
                "embedding_model": self.embedding_model_name,
                "dimension": self.get_dimension(),
                "distance_metric": self.get_distance_metric(),
            }

        except Exception as e:
            self.logger.error(f"Qdrant health check failed: {e}")
            return {
                "status": "unhealthy",
                "backend": "qdrant",
                "url": self.url,
                "error": str(e),
                "embedding_model": self.embedding_model_name,
            }

    async def close(self):
        """Close connections to Qdrant"""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None

        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

        self.logger.debug("Closed Qdrant connections")
