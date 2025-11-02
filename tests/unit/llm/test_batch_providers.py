"""
Unit tests for batch provider base class and registry

Tests the interface contract using a mock subclass and validates
registry functionality for provider management.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

import pytest

from inference_core.llm.batch import (
    BaseBatchProvider,
    BatchProviderRegistry,
    PreparedSubmission,
    ProviderNotFoundError,
    ProviderPermanentError,
    ProviderResultRow,
    ProviderStatus,
    ProviderSubmitResult,
    ProviderTransientError,
    get_global_registry,
)
from inference_core.llm.batch.exceptions import ProviderRegistrationError


class MockBatchProvider(BaseBatchProvider):
    """Mock provider implementation for testing the interface contract."""

    PROVIDER_NAME = "mock_provider"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.submitted_batches = {}
        self.batch_statuses = {}
        self.batch_results = {}

    def supports_model(self, model: str, mode: str) -> bool:
        """Mock implementation - supports specific test models."""
        supported_models = ["test-model", "gpt-4o-mini"]
        supported_modes = ["chat", "completion"]
        return model in supported_models and mode in supported_modes

    def prepare_payloads(
        self,
        batch_items: List[dict],
        model: str,
        mode: str,
        config: Optional[dict] = None,
    ) -> PreparedSubmission:
        """Mock implementation - creates a prepared submission."""
        if not batch_items:
            raise ProviderPermanentError(
                "Batch items cannot be empty", self.PROVIDER_NAME
            )

        # Simulate provider-specific formatting
        formatted_items = []
        for i, item in enumerate(batch_items):
            formatted_items.append(
                {
                    "custom_id": f"item_{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": item.get("input_payload", {}),
                }
            )

        return PreparedSubmission(
            batch_job_id=uuid4(),
            provider=self.PROVIDER_NAME,
            model=model,
            mode=mode,
            items=formatted_items,
            config=config,
        )

    def submit(self, prepared_submission: PreparedSubmission) -> ProviderSubmitResult:
        """Mock implementation - simulates batch submission."""
        if prepared_submission.model == "unsupported-model":
            raise ProviderPermanentError("Unsupported model", self.PROVIDER_NAME)

        if prepared_submission.model == "rate-limited-model":
            raise ProviderTransientError(
                "Rate limit exceeded", self.PROVIDER_NAME, retry_after=60
            )

        provider_batch_id = f"batch_{len(self.submitted_batches)}"
        now = datetime.now()

        self.submitted_batches[provider_batch_id] = prepared_submission
        self.batch_statuses[provider_batch_id] = "submitted"

        return ProviderSubmitResult(
            provider_batch_id=provider_batch_id,
            status="submitted",
            submitted_at=now,
            item_count=len(prepared_submission.items),
        )

    def poll_status(self, provider_batch_id: str) -> ProviderStatus:
        """Mock implementation - returns batch status."""
        if provider_batch_id not in self.submitted_batches:
            raise ProviderPermanentError(
                f"Batch {provider_batch_id} not found", self.PROVIDER_NAME
            )

        status = self.batch_statuses.get(provider_batch_id, "submitted")

        return ProviderStatus(
            provider_batch_id=provider_batch_id,
            status=status,
            normalized_status=status,
            progress_info={
                "completed": 0,
                "total": len(self.submitted_batches[provider_batch_id].items),
            },
        )

    def fetch_results(self, provider_batch_id: str) -> List[ProviderResultRow]:
        """Mock implementation - returns mock results."""
        if provider_batch_id not in self.submitted_batches:
            raise ProviderPermanentError(
                f"Batch {provider_batch_id} not found", self.PROVIDER_NAME
            )

        if provider_batch_id not in self.batch_results:
            # Generate mock results
            submission = self.submitted_batches[provider_batch_id]
            results = []
            for i, item in enumerate(submission.items):
                results.append(
                    ProviderResultRow(
                        custom_id=item["custom_id"],
                        output_text=f"Mock output for item {i}",
                        output_data={"mock": True, "item_index": i},
                        raw_metadata={"provider": self.PROVIDER_NAME},
                        error_message=None,
                        is_success=True,
                    )
                )
            self.batch_results[provider_batch_id] = results

        return self.batch_results[provider_batch_id]

    def cancel(self, provider_batch_id: str) -> bool:
        """Mock implementation - simulates cancellation."""
        if provider_batch_id not in self.submitted_batches:
            raise ProviderPermanentError(
                f"Batch {provider_batch_id} not found", self.PROVIDER_NAME
            )

        self.batch_statuses[provider_batch_id] = "cancelled"
        return True


class InvalidProvider:
    """Invalid provider class that doesn't inherit from BaseBatchProvider."""

    PROVIDER_NAME = "invalid_provider"


class MissingNameProvider(BaseBatchProvider):
    """Provider without PROVIDER_NAME set."""

    def supports_model(self, model: str, mode: str) -> bool:
        return True

    def prepare_payloads(
        self,
        batch_items: List[dict],
        model: str,
        mode: str,
        config: Optional[dict] = None,
    ) -> PreparedSubmission:
        pass

    def submit(self, prepared_submission: PreparedSubmission) -> ProviderSubmitResult:
        pass

    def poll_status(self, provider_batch_id: str) -> ProviderStatus:
        pass

    def fetch_results(self, provider_batch_id: str) -> List[ProviderResultRow]:
        pass

    def cancel(self, provider_batch_id: str) -> bool:
        pass


class TestBaseBatchProvider:
    """Test the BaseBatchProvider interface contract."""

    def setup_method(self):
        """Setup test environment."""
        self.provider = MockBatchProvider()

    def test_provider_name_access(self):
        """Test provider name is accessible."""
        assert self.provider.get_provider_name() == "mock_provider"
        assert self.provider.PROVIDER_NAME == "mock_provider"

    def test_supports_model_contract(self):
        """Test supports_model method contract."""
        # Supported combinations
        assert self.provider.supports_model("test-model", "chat") is True
        assert self.provider.supports_model("gpt-4o-mini", "completion") is True

        # Unsupported combinations
        assert self.provider.supports_model("unsupported-model", "chat") is False
        assert self.provider.supports_model("test-model", "embedding") is False

    def test_prepare_payloads_contract(self):
        """Test prepare_payloads method contract."""
        batch_items = [
            {"input_payload": {"messages": [{"role": "user", "content": "Hello"}]}},
            {"input_payload": {"messages": [{"role": "user", "content": "World"}]}},
        ]

        result = self.provider.prepare_payloads(batch_items, "test-model", "chat")

        assert isinstance(result, PreparedSubmission)
        assert result.provider == "mock_provider"
        assert result.model == "test-model"
        assert result.mode == "chat"
        assert len(result.items) == 2
        assert result.get_item_count() == 2

    def test_prepare_payloads_empty_items_error(self):
        """Test prepare_payloads raises error for empty items."""
        with pytest.raises(ProviderPermanentError, match="Batch items cannot be empty"):
            self.provider.prepare_payloads([], "test-model", "chat")

    def test_submit_contract(self):
        """Test submit method contract."""
        # Prepare submission
        batch_items = [
            {"input_payload": {"messages": [{"role": "user", "content": "Test"}]}}
        ]
        prepared = self.provider.prepare_payloads(batch_items, "test-model", "chat")

        # Submit
        result = self.provider.submit(prepared)

        assert isinstance(result, ProviderSubmitResult)
        assert result.provider_batch_id.startswith("batch_")
        assert result.status == "submitted"
        assert isinstance(result.submitted_at, datetime)
        assert result.item_count == 1

    def test_submit_permanent_error(self):
        """Test submit raises permanent error for unsupported model."""
        batch_items = [
            {"input_payload": {"messages": [{"role": "user", "content": "Test"}]}}
        ]
        prepared = self.provider.prepare_payloads(
            batch_items, "unsupported-model", "chat"
        )
        prepared.model = "unsupported-model"  # Force unsupported model

        with pytest.raises(ProviderPermanentError, match="Unsupported model"):
            self.provider.submit(prepared)

    def test_submit_transient_error(self):
        """Test submit raises transient error for rate limits."""
        batch_items = [
            {"input_payload": {"messages": [{"role": "user", "content": "Test"}]}}
        ]
        prepared = self.provider.prepare_payloads(
            batch_items, "rate-limited-model", "chat"
        )
        prepared.model = "rate-limited-model"  # Force rate limit

        with pytest.raises(ProviderTransientError) as exc_info:
            self.provider.submit(prepared)

        assert exc_info.value.retry_after == 60

    def test_poll_status_contract(self):
        """Test poll_status method contract."""
        # Submit batch first
        batch_items = [
            {"input_payload": {"messages": [{"role": "user", "content": "Test"}]}}
        ]
        prepared = self.provider.prepare_payloads(batch_items, "test-model", "chat")
        submit_result = self.provider.submit(prepared)

        # Poll status
        status = self.provider.poll_status(submit_result.provider_batch_id)

        assert isinstance(status, ProviderStatus)
        assert status.provider_batch_id == submit_result.provider_batch_id
        assert status.status == "submitted"
        assert status.normalized_status == "submitted"
        assert status.progress_info is not None

    def test_poll_status_not_found_error(self):
        """Test poll_status raises error for non-existent batch."""
        with pytest.raises(ProviderPermanentError, match="Batch nonexistent not found"):
            self.provider.poll_status("nonexistent")

    def test_fetch_results_contract(self):
        """Test fetch_results method contract."""
        # Submit batch first
        batch_items = [
            {"input_payload": {"messages": [{"role": "user", "content": "Test 1"}]}},
            {"input_payload": {"messages": [{"role": "user", "content": "Test 2"}]}},
        ]
        prepared = self.provider.prepare_payloads(batch_items, "test-model", "chat")
        submit_result = self.provider.submit(prepared)

        # Fetch results
        results = self.provider.fetch_results(submit_result.provider_batch_id)

        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, ProviderResultRow)
            assert result.custom_id.startswith("item_")
            assert result.output_text is not None
            assert result.is_success is True

    def test_fetch_results_not_found_error(self):
        """Test fetch_results raises error for non-existent batch."""
        with pytest.raises(ProviderPermanentError, match="Batch nonexistent not found"):
            self.provider.fetch_results("nonexistent")

    def test_cancel_contract(self):
        """Test cancel method contract."""
        # Submit batch first
        batch_items = [
            {"input_payload": {"messages": [{"role": "user", "content": "Test"}]}}
        ]
        prepared = self.provider.prepare_payloads(batch_items, "test-model", "chat")
        submit_result = self.provider.submit(prepared)

        # Cancel batch
        result = self.provider.cancel(submit_result.provider_batch_id)

        assert result is True

        # Verify status changed
        status = self.provider.poll_status(submit_result.provider_batch_id)
        assert status.status == "cancelled"

    def test_cancel_not_found_error(self):
        """Test cancel raises error for non-existent batch."""
        with pytest.raises(ProviderPermanentError, match="Batch nonexistent not found"):
            self.provider.cancel("nonexistent")

    def test_validate_config_default(self):
        """Test default validate_config implementation."""
        assert self.provider.validate_config({}) is True
        assert self.provider.validate_config({"key": "value"}) is True


class TestBatchProviderRegistry:
    """Test the BatchProviderRegistry functionality."""

    def setup_method(self):
        """Setup test environment with fresh registry."""
        self.registry = BatchProviderRegistry()

    def test_register_valid_provider(self):
        """Test registering a valid provider."""
        self.registry.register(MockBatchProvider)

        assert self.registry.is_registered("mock_provider") is True
        assert "mock_provider" in self.registry.list()

    def test_register_invalid_provider_class(self):
        """Test registering invalid provider class raises error."""
        with pytest.raises(
            ProviderRegistrationError, match="must inherit from BaseBatchProvider"
        ):
            self.registry.register(InvalidProvider)

    def test_register_provider_without_name(self):
        """Test registering provider without PROVIDER_NAME raises error."""
        with pytest.raises(ProviderRegistrationError, match="must set PROVIDER_NAME"):
            self.registry.register(MissingNameProvider)

    def test_register_duplicate_provider_name(self):
        """Test registering duplicate provider name raises error."""
        self.registry.register(MockBatchProvider)

        with pytest.raises(ProviderRegistrationError, match="already registered"):
            self.registry.register(MockBatchProvider)

    def test_get_registered_provider(self):
        """Test getting a registered provider."""
        self.registry.register(MockBatchProvider)

        provider_class = self.registry.get("mock_provider")
        assert provider_class == MockBatchProvider

    def test_get_unregistered_provider_raises_error(self):
        """Test getting unregistered provider raises ProviderNotFoundError."""
        with pytest.raises(ProviderNotFoundError) as exc_info:
            self.registry.get("nonexistent_provider")

        assert exc_info.value.provider_name == "nonexistent_provider"
        assert "not registered" in str(exc_info.value)

    def test_list_empty_registry(self):
        """Test listing providers in empty registry."""
        assert self.registry.list() == []

    def test_list_multiple_providers(self):
        """Test listing multiple registered providers."""

        # Create another mock provider
        class AnotherMockProvider(BaseBatchProvider):
            PROVIDER_NAME = "another_provider"

            def supports_model(self, model: str, mode: str) -> bool:
                return True

            def prepare_payloads(
                self,
                batch_items: List[dict],
                model: str,
                mode: str,
                config: Optional[dict] = None,
            ) -> PreparedSubmission:
                pass

            def submit(
                self, prepared_submission: PreparedSubmission
            ) -> ProviderSubmitResult:
                pass

            def poll_status(self, provider_batch_id: str) -> ProviderStatus:
                pass

            def fetch_results(self, provider_batch_id: str) -> List[ProviderResultRow]:
                pass

            def cancel(self, provider_batch_id: str) -> bool:
                pass

        self.registry.register(MockBatchProvider)
        self.registry.register(AnotherMockProvider)

        provider_names = self.registry.list()
        assert len(provider_names) == 2
        assert "mock_provider" in provider_names
        assert "another_provider" in provider_names

    def test_is_registered(self):
        """Test is_registered method."""
        assert self.registry.is_registered("mock_provider") is False

        self.registry.register(MockBatchProvider)
        assert self.registry.is_registered("mock_provider") is True
        assert self.registry.is_registered("nonexistent") is False

    def test_create_provider_instance(self):
        """Test creating provider instance with config."""
        self.registry.register(MockBatchProvider)

        config = {"test_key": "test_value"}
        provider = self.registry.create_provider("mock_provider", config)

        assert isinstance(provider, MockBatchProvider)
        assert provider.config == config
        assert provider.get_provider_name() == "mock_provider"

    def test_create_provider_not_found(self):
        """Test creating provider instance for unregistered provider."""
        with pytest.raises(ProviderNotFoundError):
            self.registry.create_provider("nonexistent_provider")

    def test_unregister_provider(self):
        """Test unregistering a provider."""
        self.registry.register(MockBatchProvider)
        assert self.registry.is_registered("mock_provider") is True

        result = self.registry.unregister("mock_provider")
        assert result is True
        assert self.registry.is_registered("mock_provider") is False

    def test_unregister_nonexistent_provider(self):
        """Test unregistering non-existent provider."""
        result = self.registry.unregister("nonexistent")
        assert result is False

    def test_clear_registry(self):
        """Test clearing all providers."""
        self.registry.register(MockBatchProvider)
        assert len(self.registry.list()) == 1

        self.registry.clear()
        assert len(self.registry.list()) == 0
        assert self.registry.is_registered("mock_provider") is False


class TestGlobalRegistry:
    """Test the global registry instance."""

    def setup_method(self):
        """Clear global registry before each test."""
        registry = get_global_registry()
        registry.clear()

    def teardown_method(self):
        """Clear global registry after each test."""
        registry = get_global_registry()
        registry.clear()

    def test_global_registry_usage(self):
        """Test using the global registry instance."""
        # Register provider
        registry = get_global_registry()
        registry.register(MockBatchProvider)

        # Verify it's accessible
        assert registry.is_registered("mock_provider") is True
        provider_class = registry.get("mock_provider")
        assert provider_class == MockBatchProvider

        # Create instance
        provider = registry.create_provider("mock_provider")
        assert isinstance(provider, MockBatchProvider)
