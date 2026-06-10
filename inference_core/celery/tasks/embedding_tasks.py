"""Celery tasks for CPU-bound embedding generation.

These tasks run on dedicated prefork workers consuming the 'embeddings' queue.
The SentenceTransformer model is loaded ONLY in this worker process — never in
the API process or the agent/LLM Celery workers.

Worker startup example::

    poetry run celery -A inference_core.celery.celery_main:celery_app worker \
        --loglevel=info --queues=embeddings --pool=prefork --concurrency=2
"""

import logging
from typing import Any

from inference_core.celery.celery_main import celery_app

logger = logging.getLogger(__name__)

# Per-process model cache (SentenceTransformer loaded once per prefork child).
_model_cache: dict[str, Any] = {}

# Per-process cross-encoder cache (reranker models, loaded once per child).
_cross_encoder_cache: dict[str, Any] = {}


def _get_sentence_transformer(model_name: str):
    """Lazy-load and cache a SentenceTransformer model for the current worker process."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer model: %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _get_cross_encoder(model_name: str):
    """Lazy-load and cache a CrossEncoder model for the current worker process."""
    if model_name not in _cross_encoder_cache:
        from sentence_transformers import CrossEncoder

        logger.info("Loading CrossEncoder model: %s", model_name)
        _cross_encoder_cache[model_name] = CrossEncoder(model_name)
    return _cross_encoder_cache[model_name]


@celery_app.task(name="embeddings.generate")
def generate_embeddings(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[list[float]]:
    """Generate embeddings for a batch of texts using SentenceTransformer.

    CPU-bound task designed for prefork workers. The model is loaded once per
    worker process and cached in ``_model_cache``.
    """
    model = _get_sentence_transformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings.tolist()


@celery_app.task(name="embeddings.rerank")
def rerank_documents(
    query: str,
    documents: list[str],
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> list[float]:
    """Score (query, document) relevance pairs with a cross-encoder reranker.

    Returns one raw relevance score per document, in input order. CPU/GPU-bound
    task designed for the dedicated ``embeddings`` workers; the model is loaded
    once per worker process and cached in ``_cross_encoder_cache``.
    """
    if not documents:
        return []
    model = _get_cross_encoder(model_name)
    scores = model.predict([(query, document) for document in documents])
    return [float(score) for score in scores]
