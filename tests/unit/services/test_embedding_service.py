"""Unit tests for EmbeddingService abstraction layer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_core.services.embedding_service import (
    EmbeddingService,
    LocalCeleryBackend,
    RemoteLangChainBackend,
    clear_embedding_service_cache,
    get_embedding_service,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure the singleton is reset before and after each test."""
    clear_embedding_service_cache()
    yield
    clear_embedding_service_cache()


def _mock_settings(**overrides):
    settings = MagicMock()
    settings.embedding_backend = overrides.get("embedding_backend", "local")
    settings.embedding_local_model = overrides.get(
        "embedding_local_model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    settings.embedding_local_timeout = overrides.get("embedding_local_timeout", 60)
    settings.vector_dim = overrides.get("vector_dim", 384)
    return settings


# ---------------------------------------------------------------------------
# LocalCeleryBackend
# ---------------------------------------------------------------------------


class TestLocalCeleryBackend:
    def test_embed_texts_sends_celery_task(self):
        settings = _mock_settings()
        backend = LocalCeleryBackend(settings)

        mock_result = MagicMock()
        mock_result.get.return_value = [[0.1, 0.2], [0.3, 0.4]]

        with patch("inference_core.celery.celery_main.celery_app") as mock_app:
            mock_app.send_task.return_value = mock_result

            result = backend.embed_texts(["hello", "world"])

        mock_app.send_task.assert_called_once_with(
            "embeddings.generate",
            kwargs={
                "texts": ["hello", "world"],
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            },
            queue="embeddings",
        )
        mock_result.get.assert_called_once_with(timeout=60, disable_sync_subtasks=False)
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_get_dimension_probes_and_caches(self):
        settings = _mock_settings()
        backend = LocalCeleryBackend(settings)

        mock_result = MagicMock()
        mock_result.get.return_value = [[0.1, 0.2, 0.3]]

        with patch("inference_core.celery.celery_main.celery_app") as mock_app:
            mock_app.send_task.return_value = mock_result

            dim1 = backend.get_dimension()
            dim2 = backend.get_dimension()

        assert dim1 == 3
        assert dim2 == 3
        # Only one probe call, second is cached
        assert mock_app.send_task.call_count == 1

    def test_get_dimension_fallback_on_error(self):
        settings = _mock_settings(vector_dim=768)
        backend = LocalCeleryBackend(settings)

        with patch("inference_core.celery.celery_main.celery_app") as mock_app:
            mock_app.send_task.side_effect = Exception("No worker")

            dim = backend.get_dimension()

        assert dim == 768

    @pytest.mark.asyncio
    async def test_aembed_texts_delegates_to_sync(self):
        settings = _mock_settings()
        backend = LocalCeleryBackend(settings)

        mock_result = MagicMock()
        mock_result.get.return_value = [[0.5, 0.6]]

        with patch("inference_core.celery.celery_main.celery_app") as mock_app:
            mock_app.send_task.return_value = mock_result

            result = await backend.aembed_texts(["test"])

        assert result == [[0.5, 0.6]]


# ---------------------------------------------------------------------------
# RemoteLangChainBackend
# ---------------------------------------------------------------------------


class TestRemoteLangChainBackend:
    def _make_mock_llm_config(self, embed_config):
        mock_config = MagicMock()
        mock_config.get_embedding_config.return_value = embed_config
        mock_config.providers = {
            "openai": {"api_key_env": "OPENAI_API_KEY"},
            "gemini": {"api_key_env": "GOOGLE_API_KEY"},
            "deepinfra": {"api_key_env": "DEEPINFRA_API_TOKEN"},
            "ollama": {"base_url": "http://localhost:11434"},
        }
        return mock_config

    @patch("inference_core.llm.config.get_llm_config")
    def test_raises_if_no_default_config(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.get_embedding_config.return_value = None
        mock_get_config.return_value = mock_config

        settings = _mock_settings(embedding_backend="remote")
        with pytest.raises(ValueError, match="no 'embeddings.default' section"):
            RemoteLangChainBackend(settings)

    @patch("inference_core.llm.config.get_llm_config")
    @patch("langchain_openai.OpenAIEmbeddings")
    def test_openai_provider(self, mock_oai_cls, mock_get_config):
        from inference_core.llm.config import EmbeddingConfig, EmbeddingProviderType

        embed_config = EmbeddingConfig(
            provider=EmbeddingProviderType.OPENAI,
            model="text-embedding-3-small",
            dimensions=1536,
        )
        mock_get_config.return_value = self._make_mock_llm_config(embed_config)

        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 1536]
        mock_oai_cls.return_value = mock_embeddings

        settings = _mock_settings(embedding_backend="remote")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = RemoteLangChainBackend(settings)
            result = backend.embed_texts(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 1536
        mock_oai_cls.assert_called_once()

    @patch("inference_core.llm.config.get_llm_config")
    def test_unsupported_provider_raises(self, mock_get_config):
        embed_config = MagicMock()
        embed_config.provider = "unknown_provider"
        embed_config.model = "some-model"
        embed_config.api_key_env = None
        embed_config.base_url = None
        embed_config.dimensions = None

        mock_config = MagicMock()
        mock_config.get_embedding_config.return_value = embed_config
        mock_config.providers = {}
        mock_get_config.return_value = mock_config

        settings = _mock_settings(embedding_backend="remote")
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            RemoteLangChainBackend(settings)

    @patch("inference_core.llm.config.get_llm_config")
    @patch("langchain_community.embeddings.deepinfra.DeepInfraEmbeddings")
    def test_deepinfra_provider(self, mock_di_cls, mock_get_config):
        from inference_core.llm.config import EmbeddingConfig, ModelProvider

        embed_config = EmbeddingConfig(
            provider=ModelProvider.DEEPINFRA,
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        mock_get_config.return_value = self._make_mock_llm_config(embed_config)

        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 384]
        mock_di_cls.return_value = mock_embeddings

        settings = _mock_settings(embedding_backend="remote")

        with patch.dict("os.environ", {"DEEPINFRA_API_TOKEN": "deepinfra-token"}):
            backend = RemoteLangChainBackend(settings)
            result = backend.embed_texts(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 384
        mock_di_cls.assert_called_once_with(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            deepinfra_api_token="deepinfra-token",
        )


# ---------------------------------------------------------------------------
# EmbeddingService (factory + methods)
# ---------------------------------------------------------------------------


class TestEmbeddingService:
    @patch("inference_core.services.embedding_service.get_settings")
    def test_creates_local_backend(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings(embedding_backend="local")
        service = EmbeddingService()
        assert isinstance(service._backend, LocalCeleryBackend)

    @patch("inference_core.services.embedding_service.get_settings")
    @patch("inference_core.llm.config.get_llm_config")
    @patch("langchain_openai.OpenAIEmbeddings")
    def test_creates_remote_backend(
        self, mock_oai_cls, mock_get_config, mock_get_settings
    ):
        from inference_core.llm.config import EmbeddingConfig, EmbeddingProviderType

        mock_get_settings.return_value = _mock_settings(embedding_backend="remote")

        embed_config = EmbeddingConfig(
            provider=EmbeddingProviderType.OPENAI,
            model="text-embedding-3-small",
        )
        mock_config = MagicMock()
        mock_config.get_embedding_config.return_value = embed_config
        mock_config.providers = {"openai": {"api_key_env": "OPENAI_API_KEY"}}
        mock_get_config.return_value = mock_config

        mock_oai_cls.return_value = MagicMock()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            service = EmbeddingService()

        assert isinstance(service._backend, RemoteLangChainBackend)

    @patch("inference_core.services.embedding_service.get_settings")
    def test_unknown_backend_raises(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings(embedding_backend="invalid")
        # Need to bypass Literal validation by direct attribute
        mock_get_settings.return_value.embedding_backend = "invalid"
        with pytest.raises(ValueError, match="Unknown EMBEDDING_BACKEND"):
            EmbeddingService()

    def test_get_embed_fn_returns_callable(self):
        mock_backend = MagicMock()
        mock_backend.embed_texts.return_value = [[1.0, 2.0]]
        service = EmbeddingService(backend=mock_backend)

        fn = service.get_embed_fn()
        result = fn(["test"])
        assert result == [[1.0, 2.0]]

    def test_embed_query(self):
        mock_backend = MagicMock()
        mock_backend.embed_texts.return_value = [[0.1, 0.2, 0.3]]
        service = EmbeddingService(backend=mock_backend)

        result = service.embed_query("test")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_aembed_query(self):
        mock_backend = MagicMock()
        mock_backend.aembed_texts = AsyncMock(return_value=[[0.4, 0.5]])
        service = EmbeddingService(backend=mock_backend)

        result = await service.aembed_query("test")
        assert result == [0.4, 0.5]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    @patch("inference_core.services.embedding_service.get_settings")
    def test_get_embedding_service_returns_same_instance(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings(embedding_backend="local")

        s1 = get_embedding_service()
        s2 = get_embedding_service()
        assert s1 is s2

    @patch("inference_core.services.embedding_service.get_settings")
    def test_clear_cache_resets_singleton(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings(embedding_backend="local")

        s1 = get_embedding_service()
        clear_embedding_service_cache()
        s2 = get_embedding_service()
        assert s1 is not s2
