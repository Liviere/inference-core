# Vector Store Guide

This guide explains how to configure and use the vector store functionality in Inference Core for semantic document storage and retrieval.

## Overview

The vector store system provides:
- **Pluggable backends**: Support for different vector databases (Qdrant, in-memory)
- **RESTful API**: Endpoints for document ingestion and similarity search
- **Async processing**: Celery tasks for batch document ingestion
- **Metrics**: Prometheus monitoring of operations
- **Authentication**: Integrated with existing auth system

## Quick Start

### 1. Configuration

Add vector store settings to your `.env` file:

```bash
# Enable vector store with in-memory backend (development)
VECTOR_BACKEND=memory

# Or use Qdrant for production
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-api-key-here

# Collection settings
VECTOR_COLLECTION_DEFAULT=documents
VECTOR_DISTANCE=cosine
VECTOR_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIM=384
```

### 2. Start Qdrant (for production)

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant:latest
```

### 3. Test the API

```bash
# Check health
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/vector/health

# Ingest documents
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  http://localhost:8000/api/v1/vector/ingest \
  -d '{
    "texts": ["Python is great", "Machine learning is awesome"],
    "metadatas": [{"type": "programming"}, {"type": "ai"}],
    "async_mode": false
  }'

# Search for similar documents
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  http://localhost:8000/api/v1/vector/query \
  -d '{
    "query": "programming language",
    "k": 5
  }'
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_BACKEND` | `None` | Backend type: `qdrant`, `memory`, or leave blank to disable |
| `VECTOR_COLLECTION_DEFAULT` | `default_documents` | Default collection name |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | `None` | Qdrant API key (optional) |
| `VECTOR_DISTANCE` | `cosine` | Distance metric: `cosine`, `euclidean`, `dot` |
| `VECTOR_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `VECTOR_DIM` | `384` | Vector dimension (must match model) |
| `VECTOR_INGEST_MAX_BATCH_SIZE` | `1000` | Maximum batch size for ingestion |

### Access Control

Vector store endpoints use the same access control as LLM endpoints:
- `LLM_API_ACCESS_MODE=public`: No authentication required
- `LLM_API_ACCESS_MODE=user`: Requires authenticated user (default for vector store)
- `LLM_API_ACCESS_MODE=superuser`: Requires superuser privileges

## API Reference

### Health Check

```http
GET /api/v1/vector/health
```

Returns vector store status and configuration.

**Response:**
```json
{
  "status": "healthy",
  "backend": "qdrant",
  "collections": ["documents", "knowledge_base"],
  "details": {
    "url": "http://localhost:6333",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

### Document Ingestion

```http
POST /api/v1/vector/ingest
```

**Request:**
```json
{
  "texts": ["Document 1", "Document 2"],
  "metadatas": [{"source": "web"}, {"source": "pdf"}],
  "ids": ["doc1", "doc2"],
  "collection": "my_collection",
  "async_mode": true
}
```

**Response (async):**
```json
{
  "task_id": "abc123",
  "message": "Document ingestion task submitted successfully",
  "collection": "my_collection",
  "estimated_count": 2
}
```

### Similarity Search

```http
POST /api/v1/vector/query
```

**Request:**
```json
{
  "query": "machine learning algorithms",
  "k": 5,
  "collection": "documents",
  "filters": {"source": "web"}
}
```

**Response:**
```json
{
  "documents": [
    {
      "id": "doc1",
      "content": "Machine learning is...",
      "metadata": {"source": "web"},
      "score": 0.95
    }
  ],
  "query": "machine learning algorithms",
  "collection": "documents",
  "count": 1,
  "total_in_collection": 100
}
```

### Collection Stats

```http
GET /api/v1/vector/collections/{collection}/stats
```

**Response:**
```json
{
  "name": "documents",
  "count": 1000,
  "dimension": 384,
  "distance_metric": "cosine"
}
```

## Backends

### In-Memory Backend

For development and testing:

```bash
VECTOR_BACKEND=memory
```

**Features:**
- No external dependencies
- Fast for small datasets
- Data lost on restart
- Simple text-based similarity

**Use cases:**
- Development and testing
- Demos and prototypes
- Unit tests

### Qdrant Backend

For production use:

```bash
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
```

**Features:**
- Production-ready vector database
- Persistent storage
- High performance at scale
- Advanced filtering and search
- Supports clustering and sharding

**Setup:**
```bash
# Local development
docker run -p 6333:6333 qdrant/qdrant:latest

# Production with persistence
docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

## Embedding Models

The system uses Sentence Transformers for text embeddings. Supported models:

| Model | Dimension | Performance | Use Case |
|-------|-----------|-------------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | General purpose |
| `all-mpnet-base-v2` | 768 | Balanced | High quality |
| `all-distilroberta-v1` | 768 | Medium | Roberta-based |

**Custom models:**
```bash
VECTOR_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
VECTOR_DIM=768
```

## Celery Integration

### Async Document Ingestion

Large document batches are processed asynchronously:

```python
from inference_core.celery.tasks.vector_tasks import ingest_documents_task

# Submit task
task = ingest_documents_task.delay(
    texts=["doc1", "doc2"],
    metadatas=[{"type": "doc"}, {"type": "doc"}],
    collection="my_collection"
)

# Check status
result = task.get()
```

### Task Monitoring

Track ingestion progress:

```bash
# Check task status
curl http://localhost:8000/api/v1/tasks/abc123/status

# Get task result
curl http://localhost:8000/api/v1/tasks/abc123/result
```

## Metrics and Monitoring

### Prometheus Metrics

The following metrics are exported:

- `vector_similarity_search_seconds`: Search latency
- `vector_ingest_batch_seconds`: Ingestion latency
- `vector_documents_ingested_total`: Documents ingested
- `vector_query_requests_total`: Query requests
- `vector_collections_total`: Number of collections
- `vector_documents_total`: Total documents per collection

### Example Queries

```promql
# Search latency by backend
histogram_quantile(0.95, vector_similarity_search_seconds_bucket)

# Ingestion rate
rate(vector_documents_ingested_total[5m])

# Error rate
rate(vector_query_requests_total{status="error"}[5m])
```

## Integration Examples

### LangChain Integration

```python
from inference_core.services.vector_store_service import get_vector_store_service

# Get a retriever for RAG
service = get_vector_store_service()
retriever = await service.get_retriever(
    collection="knowledge_base",
    search_kwargs={"k": 5}
)

# Use with LangChain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=your_llm,
    retriever=retriever
)
```

### Custom Processing Pipeline

```python
from inference_core.services.vector_store_service import get_vector_store_service

async def process_documents(documents):
    service = get_vector_store_service()
    
    # Preprocess documents
    texts = [preprocess(doc) for doc in documents]
    metadatas = [extract_metadata(doc) for doc in documents]
    
    # Ingest in batches
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        await service.add_texts(
            texts=batch_texts,
            metadatas=batch_metas,
            collection="processed_docs"
        )
```

## Troubleshooting

### Common Issues

**Vector store not available:**
```bash
# Check configuration
curl http://localhost:8000/api/v1/vector/health

# Verify environment variables
echo $VECTOR_BACKEND
```

**Qdrant connection failed:**
```bash
# Check Qdrant is running
curl http://localhost:6333/collections

# Verify URL in config
echo $QDRANT_URL
```

**Dimension mismatch:**
```bash
# Check embedding model dimension
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(model.get_sentence_embedding_dimension())
"
```

### Debug Mode

Enable debug logging:

```bash
DEBUG=true VECTOR_BACKEND=memory python -m uvicorn inference_core.main_factory:app
```

### Testing

Run vector store tests:

```bash
# Unit tests
poetry run pytest tests/unit/vectorstores/ -v

# Service tests  
poetry run pytest tests/unit/services/test_vector_store_service.py -v

# Integration tests (requires Qdrant)
QDRANT_URL=http://localhost:6333 poetry run pytest tests/integration/test_vector_store.py -v
```

## Performance Considerations

### Batch Size

- **Small batches** (< 100): Low latency, high overhead
- **Medium batches** (100-1000): Balanced performance
- **Large batches** (> 1000): High throughput, use async processing

### Memory Usage

- **In-memory backend**: Linear with document count
- **Qdrant backend**: Configurable memory limits
- **Embedding generation**: ~100MB per model

### Scaling

- **Horizontal**: Multiple Qdrant nodes with sharding
- **Vertical**: Increase memory and CPU for embedding generation
- **Async processing**: Use Celery for non-blocking operations

## Future Enhancements

- **Hybrid search**: Combine vector and keyword search (BM25)
- **Re-ranking**: LLM-based result re-ranking
- **Multi-modal**: Support for image and audio embeddings
- **Advanced filters**: Complex boolean logic for metadata
- **Streaming ingestion**: Real-time document processing
- **Collection management**: APIs for collection lifecycle