"""
Unit tests for inference_core.core.logging_config module

Tests logging configuration setup with different debug settings
and JSON formatter functionality.
"""

import logging
import os
from unittest.mock import MagicMock, call, patch

import pytest

from inference_core.core.logging_config import JsonFormatter, setup_logging


class TestJsonFormatter:
    """Test JsonFormatter functionality"""

    def setup_method(self, method):
        """Setup test environment"""
        self.formatter = JsonFormatter()

    def test_add_fields_with_timestamp(self):
        """Test JsonFormatter adds timestamp if not present"""
        log_record = {}
        record = MagicMock()
        record.created = 1640995200.0  # 2022-01-01 00:00:00 UTC
        record.levelname = "INFO"
        message_dict = {}

        self.formatter.add_fields(log_record, record, message_dict)

        assert log_record["timestamp"] == 1640995200.0
        assert log_record["level"] == "INFO"

    def test_add_fields_with_existing_timestamp(self):
        """Test JsonFormatter doesn't override existing timestamp"""
        log_record = {"timestamp": 1234567890.0}
        record = MagicMock()
        record.created = 1640995200.0
        record.levelname = "DEBUG"
        message_dict = {}

        self.formatter.add_fields(log_record, record, message_dict)

        # Should keep existing timestamp
        assert log_record["timestamp"] == 1234567890.0
        assert log_record["level"] == "DEBUG"

    def test_add_fields_with_existing_level(self):
        """Test JsonFormatter uses existing level if present"""
        log_record = {"level": "warning"}
        record = MagicMock()
        record.created = 1640995200.0
        record.levelname = "DEBUG"
        message_dict = {}

        self.formatter.add_fields(log_record, record, message_dict)

        # Should uppercase existing level
        assert log_record["level"] == "WARNING"

    def test_add_fields_uses_record_levelname(self):
        """Test JsonFormatter uses record.levelname when level not in log_record"""
        log_record = {}
        record = MagicMock()
        record.created = 1640995200.0
        record.levelname = "ERROR"
        message_dict = {}

        self.formatter.add_fields(log_record, record, message_dict)

        assert log_record["level"] == "ERROR"


class TestSetupLogging:
    """Test setup_logging function"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from inference_core.core.config import get_settings

        get_settings.cache_clear()

    @patch("inference_core.core.logging_config.dictConfig")
    def test_setup_logging_debug_mode(self, mock_dict_config):
        """Test setup_logging with debug=True sets DEBUG level"""
        # Patch the function as imported inside logging_config module
        with patch(
            "inference_core.core.logging_config.get_settings"
        ) as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.debug = True
            mock_get_settings.return_value = mock_settings

            setup_logging()

            # Verify dictConfig was called
            mock_dict_config.assert_called_once()
            config = mock_dict_config.call_args[0][0]

            # Check that log level is DEBUG
            assert config["loggers"]["uvicorn"]["level"] == "DEBUG"
            assert config["loggers"]["fastapi"]["level"] == "DEBUG"
            assert config["loggers"]["app"]["level"] == "DEBUG"
            assert config["root"]["level"] == "DEBUG"

    @patch("inference_core.core.logging_config.dictConfig")
    def test_setup_logging_production_mode(self, mock_dict_config):
        """Test setup_logging with debug=False sets INFO level"""
        # Since the logging config setup is already called, we need to isolate the test better
        with patch(
            "inference_core.core.logging_config.get_settings"
        ) as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.debug = False
            mock_get_settings.return_value = mock_settings

            setup_logging()

            # Verify dictConfig was called
            mock_dict_config.assert_called_once()
            config = mock_dict_config.call_args[0][0]

            # Check that log level is INFO
            assert config["loggers"]["uvicorn"]["level"] == "INFO"
            assert config["loggers"]["fastapi"]["level"] == "INFO"
            assert config["loggers"]["app"]["level"] == "INFO"
            assert config["root"]["level"] == "INFO"

    @patch("inference_core.core.logging_config.dictConfig")
    def test_setup_logging_config_structure(self, mock_dict_config):
        """Test setup_logging creates expected configuration structure.

        The dictConfig only contains the console handler and 'default' formatter.
        The file handler is added separately via _build_file_handler after dictConfig.
        """
        with patch("inference_core.core.config.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.debug = True
            mock_get_settings.return_value = mock_settings

            setup_logging()

            config = mock_dict_config.call_args[0][0]

            # Verify configuration structure
            assert config["version"] == 1
            assert config["disable_existing_loggers"] is False

            # Check formatters — only 'default' is in dictConfig
            assert "default" in config["formatters"]

            # Check handlers — only 'console' is in dictConfig
            assert "console" in config["handlers"]
            assert config["handlers"]["console"]["class"] == "logging.StreamHandler"

            # Check loggers
            required_loggers = ["uvicorn", "fastapi", "app"]
            for logger_name in required_loggers:
                assert logger_name in config["loggers"]
                assert "console" in config["loggers"][logger_name]["handlers"]
                assert config["loggers"][logger_name]["propagate"] is False

            # Check root logger
            assert "console" in config["root"]["handlers"]

    @patch("inference_core.core.logging_config._build_file_handler")
    @patch("inference_core.core.logging_config.dictConfig")
    def test_setup_logging_file_handler_config(self, mock_dict_config, mock_build):
        """Test file handler is built with correct parameters.

        The file handler is created via _build_file_handler and added to
        loggers after dictConfig is applied.
        """
        with patch("inference_core.core.config.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.debug = False
            mock_get_settings.return_value = mock_settings

            mock_handler = MagicMock()
            mock_handler.level = logging.INFO
            mock_build.return_value = mock_handler

            setup_logging()

            # Clean up mock handler from global loggers to prevent leaking
            for name in ("uvicorn", "fastapi", "app"):
                lgr = logging.getLogger(name)
                if mock_handler in lgr.handlers:
                    lgr.removeHandler(mock_handler)
            if mock_handler in logging.root.handlers:
                logging.root.removeHandler(mock_handler)

            mock_build.assert_called_once()
            call_args = mock_build.call_args
            log_file_path = call_args[0][0]
            log_level = call_args[0][1]

            assert log_file_path == os.path.join("logs", "app.log")
            assert log_level == "INFO"

    @patch("inference_core.core.logging_config.dictConfig")
    def test_setup_logging_console_handler_config(self, mock_dict_config):
        """Test console handler configuration in setup_logging"""
        import sys

        with patch("inference_core.core.config.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.debug = True
            mock_get_settings.return_value = mock_settings

            setup_logging()

            config = mock_dict_config.call_args[0][0]
            console_handler = config["handlers"]["console"]

            assert console_handler["class"] == "logging.StreamHandler"
            assert console_handler["formatter"] == "default"
            assert console_handler["stream"] == sys.stdout

    def test_integration_with_actual_logger(self, caplog):
        """Test integration with actual logging to verify handler works"""
        # Test that logging works - caplog captures at pytest level, not handler level
        test_logger = logging.getLogger("inference_core.test")

        # Test that logging works
        with caplog.at_level(logging.DEBUG):
            test_logger.debug("Test debug message")
            test_logger.info("Test info message")
            test_logger.error("Test error message")

        # Verify messages were captured by caplog
        assert any("Test debug message" in record.message for record in caplog.records)
        assert any("Test info message" in record.message for record in caplog.records)
        assert any("Test error message" in record.message for record in caplog.records)

    def test_json_formatter_integration(self):
        """Test JsonFormatter integration in actual logging"""
        formatter = JsonFormatter()

        # Create a test log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Format the record
        formatted = formatter.format(record)

        # Verify it's JSON-like format (contains expected fields)
        assert "timestamp" in formatted or str(record.created) in formatted
        assert "INFO" in formatted
        assert "Test message" in formatted
        # Note: logger name might not be in formatted output depending on format string
