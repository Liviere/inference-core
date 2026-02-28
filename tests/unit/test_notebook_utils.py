"""Tests for Jupyter notebook utilities.

Covers: reset_for_jupyter, configure_for_jupyter.
"""

from unittest.mock import patch


class TestResetForJupyter:
    """Test reset_for_jupyter clears singletons."""

    @patch("inference_core.core.config.get_settings")
    @patch("inference_core.llm.usage_logging.reset_usage_logging_cache")
    @patch("inference_core.database.sql.connection.reset_database_for_new_event_loop")
    def test_calls_all_reset_functions(self, mock_db_reset, mock_usage_reset, mock_settings):
        """reset_for_jupyter calls all three reset points."""
        from inference_core.notebook_utils import reset_for_jupyter

        reset_for_jupyter()

        mock_db_reset.assert_called_once()
        mock_usage_reset.assert_called_once()
        mock_settings.cache_clear.assert_called_once()


class TestConfigureForJupyter:
    """Test configure_for_jupyter sets NullPool."""

    @patch("inference_core.database.sql.connection.reset_database_for_new_event_loop")
    @patch("inference_core.core.config.get_settings")
    @patch.dict("os.environ", {}, clear=False)
    def test_sets_null_pool_env(self, mock_settings, mock_db_reset):
        """configure_for_jupyter sets DATABASE_POOL_CLASS=null."""
        import os

        from inference_core.notebook_utils import configure_for_jupyter

        configure_for_jupyter()

        assert os.environ.get("DATABASE_POOL_CLASS") == "null"
        mock_settings.cache_clear.assert_called_once()
        mock_db_reset.assert_called_once()
