"""Embedding Generation Router.

Provides a public endpoint for generating text embeddings via the configured
backend (local Celery worker or remote API provider). Auth follows
``LLM_API_ACCESS_MODE`` — identical to other LLM endpoints.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from inference_core.core.config import get_settings
from inference_core.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/embeddings",
    tags=["Embeddings"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(
        ..., description="Texts to embed", min_length=1, max_length=500
    )


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]] = Field(
        ..., description="Generated embedding vectors"
    )
    dimension: int = Field(..., description="Vector dimensionality")
    count: int = Field(..., description="Number of embeddings generated")
    backend: str = Field(..., description="Backend used (local or remote)")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/generate", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for a list of texts.

    Uses the configured embedding backend (``EMBEDDING_BACKEND`` env var):

    - **local** — Delegates to Celery prefork worker with SentenceTransformer
    - **remote** — Calls external API provider via LangChain embeddings
    """
    try:
        service = get_embedding_service()
        embeddings = await service.aembed_texts(request.texts)
        dimension = len(embeddings[0]) if embeddings else 0
        settings = get_settings()

        return EmbeddingResponse(
            embeddings=embeddings,
            dimension=dimension,
            count=len(embeddings),
            backend=settings.embedding_backend,
        )
    except Exception as e:
        logger.error("Embedding generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {e}",
        )
