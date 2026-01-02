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

        # Prepare search filter if provided (supports nested dictionaries)
        search_filter = self._build_qdrant_filter(filters) if filters else None
        # Perform search
        search_result = await client.query_points(
            collection_name=collection,
            query=query_embedding,
            query_filter=search_filter,
            limit=k,
            with_payload=True,
        )
        points = search_result.points if hasattr(search_result, "points") else []

        # Convert results to VectorStoreDocument
        documents = []
        for point in points:
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

    async def list_documents(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: Optional[str] = None,
        order: str = "desc",
        include_scores: bool = False,
    ) -> tuple[List[VectorStoreDocument], int]:
        """List documents by metadata filters using Qdrant scroll API"""
        client = await self._get_async_client()

        # Prepare filter if provided (supports nested dictionaries)
        search_filter = self._build_qdrant_filter(filters) if filters else None

        # Get total count with the filter
        try:
            count_result = await client.count(
                collection_name=collection,
                count_filter=search_filter,
                exact=True,
            )
            total_count = count_result.count if hasattr(count_result, "count") else 0
        except (ResponseHandlingException, UnexpectedResponse):
            # Collection doesn't exist
            return ([], 0)

        # Use scroll to get documents with pagination
        # Qdrant scroll doesn't support direct offset, so we need to scroll through
        documents = []
        scroll_offset = None
        current_offset = 0

        # Keep scrolling until we reach the desired offset + limit
        while len(documents) < limit and (
            current_offset < offset + limit or total_count == 0
        ):
            try:
                scroll_result = await client.scroll(
                    collection_name=collection,
                    scroll_filter=search_filter,
                    limit=min(100, offset + limit - current_offset),  # Fetch in batches
                    offset=scroll_offset,
                    with_payload=True,
                    with_vectors=False,  # We don't need vectors for listing
                )

                points, next_offset = scroll_result

                if not points:
                    break

                # Process points and add to results if past offset
                for point in points:
                    if current_offset >= offset:
                        # Extract text from payload
                        content = point.payload.get("_text", "")
                        # Create metadata without internal fields
                        metadata = {
                            k: v
                            for k, v in point.payload.items()
                            if not k.startswith("_")
                        }

                        doc = VectorStoreDocument(
                            id=str(point.id),
                            content=content,
                            metadata=metadata,
                            score=None if not include_scores else 0.0,
                        )
                        documents.append(doc)

                        if len(documents) >= limit:
                            break

                    current_offset += 1

                # Update scroll offset for next iteration
                scroll_offset = next_offset
                if next_offset is None:
                    break

            except (ResponseHandlingException, UnexpectedResponse) as e:
                self.logger.error(f"Error scrolling collection '{collection}': {e}")
                break

        # Note: Qdrant scroll doesn't natively support ordering by metadata fields
        # If order_by is specified, we sort in memory (not ideal for large datasets)
        if order_by and documents:

            def get_sort_key(doc: VectorStoreDocument):
                value = doc.metadata.get(order_by)
                # Handle None values by placing them at the end
                if value is None:
                    return ("", "") if order == "asc" else ("~", "~")
                return value

            try:
                documents = sorted(
                    documents,
                    key=get_sort_key,
                    reverse=(order == "desc"),
                )
            except Exception as e:
                # If sorting fails, keep original order
                self.logger.warning(
                    f"Failed to sort by {order_by}, using insertion order: {e}"
                )

        self.logger.debug(
            f"Listed {len(documents)} documents (total: {total_count}) from Qdrant collection '{collection}'"
        )
        return (documents, total_count)

    # --------------------
    # Helpers
    # --------------------
    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Optional[models.Filter]:
        """Build a Qdrant models.Filter from a (possibly nested) dict.

        Rules:
        - Scalar values (str/int/float/bool) -> FieldCondition(key, MatchValue(value))
        - Empty dict value -> presence check: require key to exist (implemented as Nested with no conditions is not supported; we mimic by not adding a condition to avoid 400s)
        - Dict value with scalars -> Nested(key=<parent>, filter=Filter(must=[FieldCondition(...)]))
        - Deeply nested dict -> recursive Nested for each level

        Note: Qdrant supports Nested(key=..., filter=Filter(...)) for JSON objects.
        """
        if not filters:
            return None

        must_conditions: List[Any] = []

        for key, value in filters.items():
            if key == "_text":
                # never filter on internal field
                continue

            cond = self._build_condition_for_key(key, value)
            if cond is None:
                # Skip unsupported/empty presence checks to avoid validation errors
                continue
            # Support single condition or list of conditions
            if isinstance(cond, list):
                must_conditions.extend(cond)
            else:
                must_conditions.append(cond)

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)

    def _build_condition_for_key(self, key: str, value: Any):
        """Recursively build FieldCondition or Nested for a key/value."""
        # Presence check: treat {} as "key exists" by skipping (no-op filter)
        if isinstance(value, dict) and len(value) == 0:
            return None

        # Scalar -> simple match
        if isinstance(value, (str, int, float, bool)):
            return models.FieldCondition(key=key, match=models.MatchValue(value=value))

        # Dict with scalar sub-keys -> Nested
        if isinstance(value, dict):
            # Range support for numeric fields (gte/lte/gt/lt)
            range_keys = {"gte", "lte", "gt", "lt"}
            if any(k in value for k in range_keys):
                try:
                    rng = models.Range(
                        gte=value.get("gte"),
                        lte=value.get("lte"),
                        gt=value.get("gt"),
                        lt=value.get("lt"),
                    )
                    return models.FieldCondition(key=key, range=rng)
                except Exception:
                    # Fall back to nested handling below if range construction fails
                    pass
            # Split one level: build inner conditions
            inner_must: List[Any] = []
            for sub_key, sub_val in value.items():
                # If nested deeper, recurse by creating nested with combined path
                if isinstance(sub_val, dict):
                    nested_cond = self._build_condition_for_key(
                        f"{key}.{sub_key}", sub_val
                    )
                    if nested_cond is not None:
                        # When we already expanded to a dotted path, it returns FieldCondition/Nested
                        # Wrap as must element directly
                        inner_must.append(nested_cond)
                elif isinstance(sub_val, (str, int, float, bool)):
                    inner_must.append(
                        models.FieldCondition(
                            key=f"{key}.{sub_key}",
                            match=models.MatchValue(value=sub_val),
                        )
                    )

            if not inner_must:
                return None

            # Two implementation options: Nested or dot-path FieldConditions.
            # Using dot-path is sufficient here since we already expanded keys as key.sub_key
            # Return a Filter grouping via should/must is not necessary; caller will wrap in Filter(must=[...]).
            # Therefore, return a models.Filter is not accepted as a condition; we return a models.Nested for clarity.
            try:
                # Prefer NestedCondition when available in the installed qdrant-client
                return models.NestedCondition(
                    key=key, filter=models.Filter(must=inner_must)
                )
            except Exception:
                # Fallback: flatten into top-level FieldConditions with dotted keys
                return inner_must

        # Unsupported value type -> skip
        return None

    async def close(self):
        """Close connections to Qdrant"""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None

        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

        self.logger.debug("Closed Qdrant connections")
