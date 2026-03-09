"""Unit tests for the Celery embedding task."""

from unittest.mock import MagicMock, patch

import pytest


class TestGenerateEmbeddings:
    @patch("inference_core.celery.tasks.embedding_tasks._get_sentence_transformer")
    def test_generates_embeddings(self, mock_get_model):
        import numpy as np

        from inference_core.celery.tasks.embedding_tasks import generate_embeddings

        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_get_model.return_value = fake_model

        result = generate_embeddings(["hello", "world"])

        mock_get_model.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")
        fake_model.encode.assert_called_once_with(
            ["hello", "world"], convert_to_tensor=False
        )
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    @patch("inference_core.celery.tasks.embedding_tasks._get_sentence_transformer")
    def test_custom_model_name(self, mock_get_model):
        import numpy as np

        from inference_core.celery.tasks.embedding_tasks import generate_embeddings

        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[1.0, 2.0]])
        mock_get_model.return_value = fake_model

        result = generate_embeddings(["test"], model_name="custom/model")

        mock_get_model.assert_called_once_with("custom/model")
        assert result == [[1.0, 2.0]]


class TestModelCache:
    def test_model_cached_per_name(self):
        from inference_core.celery.tasks.embedding_tasks import (
            _get_sentence_transformer,
            _model_cache,
        )

        # Clear cache for this test
        _model_cache.clear()

        mock_model_a = MagicMock(name="model-a-instance")
        mock_model_b = MagicMock(name="model-b-instance")

        with patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=lambda name: (
                mock_model_a if name == "model-a" else mock_model_b
            ),
        ) as mock_cls:
            m1 = _get_sentence_transformer("model-a")
            m2 = _get_sentence_transformer("model-a")
            m3 = _get_sentence_transformer("model-b")

        assert m1 is m2  # Same model instance
        assert m1 is not m3  # Different model name
        assert mock_cls.call_count == 2  # Only 2 loads (model-a, model-b)

        _model_cache.clear()
